"""
rship_agent.py — Scheduling + context agent for r.ship
Pattern: identical to agent_graph.py (Jarvis). Reuses tools_core.CalendarMapsTools.
New tools: r.ship DB (get_contact_context, get_pending_tasks, mark_task_done).
"""
import os, json, re, sqlite3
from datetime import datetime
import pytz
from langgraph.graph import StateGraph, START, END
from tools_core import CalendarMapsTools

IST              = pytz.timezone(os.getenv("TZ", "Asia/Kolkata"))
DB_PATH          = os.getenv("RSHIP_DB", "rship.db")
MAX_TOOL_CALLS   = 5
MAX_CONSEC_ERRORS = 2

# ── iCloud calendar (exact same instance as Jarvis uses) ──────────────────
_cal = CalendarMapsTools()

# ── r.ship DB tools ───────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def get_contact_context(contact_name: str) -> dict:
    """
    Return ALL logs + all tasks for the closest-matching contact.
    Call this BEFORE scheduling anything — the logs power the calendar description.
    """
    with _conn() as c:
        hits = [dict(r) for r in c.execute(
            "SELECT * FROM contacts WHERE LOWER(name) LIKE ?",
            (f"%{contact_name.lower()}%",)
        )]
        if not hits:
            return {"error": f"No contact found matching '{contact_name}'"}
        contact = hits[0]
        logs = [dict(r) for r in c.execute(
            "SELECT text, ai_summary, follow_up_prompt, follow_up_days, created_at "
            "FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 20",
            (contact["id"],)
        )]
        tasks = [dict(r) for r in c.execute(
            "SELECT text, due_date, status, category FROM tasks "
            "WHERE contact_id=? ORDER BY due_date ASC NULLS LAST",
            (contact["id"],)
        )]
        return {
            "contact":     contact["name"],
            "role":        contact.get("role") or "",
            "last_log_at": contact.get("last_log_at") or "never",
            "log_count":   len(logs),
            "logs":        logs,
            "tasks":       tasks,
        }

def get_pending_tasks(contact_name: str = "") -> dict:
    """
    Pending tasks — all contacts, or filtered by name.
    Call this when user asks 'what's pending' / 'anything to do'.
    """
    with _conn() as c:
        if contact_name:
            rows = [dict(r) for r in c.execute(
                "SELECT t.text, t.due_date, t.category, c.name as contact_name "
                "FROM tasks t LEFT JOIN contacts c ON c.id=t.contact_id "
                "WHERE t.status='pending' AND LOWER(c.name) LIKE ? "
                "ORDER BY t.due_date ASC NULLS LAST",
                (f"%{contact_name.lower()}%",)
            )]
        else:
            rows = [dict(r) for r in c.execute(
                "SELECT t.text, t.due_date, t.category, c.name as contact_name "
                "FROM tasks t LEFT JOIN contacts c ON c.id=t.contact_id "
                "WHERE t.status='pending' "
                "ORDER BY t.due_date ASC NULLS LAST LIMIT 30"
            )]
        return {"tasks": rows, "count": len(rows)}

def mark_task_done(task_text: str) -> dict:
    """Mark task(s) done by partial text match."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE tasks SET status='done' WHERE LOWER(text) LIKE ?",
            (f"%{task_text.lower()}%",)
        )
        c.commit()
        return {"updated": cur.rowcount, "matched_on": task_text}


def schedule_pending_task(task_text: str) -> dict:
    """Schedule a pending task by partial text match if it has remind_at or due_date."""
    with _conn() as c:
        task = c.execute(
            "SELECT t.*, c.name as contact_name FROM tasks t LEFT JOIN contacts c ON c.id=t.contact_id "
            "WHERE t.status='pending' AND LOWER(t.text) LIKE ? ORDER BY COALESCE(t.remind_at,t.due_date) ASC LIMIT 1",
            (f"%{task_text.lower()}%",),
        ).fetchone()
        if not task:
            return {"error": f"No pending task found matching '{task_text}'"}
        task = dict(task)
        if task.get("calendar_href"):
            return {"scheduled": False, "reason": "already_scheduled", "href": task.get("calendar_href")}
        start_iso = task.get("remind_at")
        if not start_iso and task.get("due_date"):
            start_iso = f"{task['due_date']}T09:00:00+05:30"
        if not start_iso:
            return {"scheduled": False, "reason": "task_has_no_due_date_or_remind_at"}
        title = f"{task.get('contact_name') or 'r.ship'} — {task.get('text','')[:52]}"[:80]
        result = _cal.create_event(
            title=title,
            start_iso=start_iso,
            duration_minutes=15,
            alert_minutes_before=int(task.get("alert_minutes_before") or 30),
            description=f"Task: {task.get('text')}\nCategory: {task.get('category')}\nPriority: {task.get('priority')}",
        )
        c.execute("UPDATE tasks SET calendar_href=?, scheduled_at=? WHERE id=?",
                  (result.get("href"), datetime.now(IST).isoformat(), task["id"]))
        c.commit()
        return {"scheduled": True, "title": title, "href": result.get("href")}

# ── Tool registry ─────────────────────────────────────────────────────────
TOOLS = {
    # iCloud CalDAV (from tools_core.py)
    "create_event":        _cal.create_event,
    "delete_event":        _cal.delete_event,
    "list_events_today":   _cal.list_events_today,
    "list_events_for_day": _cal.list_events_for_day,
    # r.ship relationship DB
    "get_contact_context": get_contact_context,
    "get_pending_tasks":   get_pending_tasks,
    "mark_task_done":      mark_task_done,
    "schedule_pending_task": schedule_pending_task,
}

# ── JSON helper ───────────────────────────────────────────────────────────
def _parse(text: str) -> dict:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$",          "", s)
    return json.loads(s.strip())

# ── LLM planner ───────────────────────────────────────────────────────────
def _llm_plan(messages: list) -> dict:
    SYSTEM = f"""You are r.ship Assistant — a personal relationship and calendar agent.
Current time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}

You have two categories of tools:

CALENDAR TOOLS (iCloud Apple Calendar via CalDAV):
- create_event(title, start_iso, duration_minutes=60, alert_minutes_before=15, description=None, location=None)
  → start_iso must be IST in ISO 8601, e.g. "2025-04-01T10:00:00+05:30"
- delete_event(href)
- list_events_today()
- list_events_for_day(date_iso)

RELATIONSHIP DB TOOLS:
- get_contact_context(contact_name)   → returns ALL logs + tasks for a person
- get_pending_tasks(contact_name="")  → returns pending tasks (blank = all contacts)
- mark_task_done(task_text)           → marks task done by partial text
- schedule_pending_task(task_text)     → schedules a pending task into Apple Calendar if it has due/remind time

RULES:
1. Output STRICT JSON only — no prose, no markdown:
   {{"action":"<tool>","args":{{...}}}}   call a tool
   {{"final":"<message>"}}               done

2. ALWAYS call get_contact_context(name) before creating a calendar event for a person.
   Use their logs to write a useful event description, e.g.
   "Ask about her Google interview result. She mentioned stress about the L5 bar."

3. Default event: 60 min, 15-min alert. All IST times as ISO 8601 with +05:30.

4. When user asks "what's pending" / "what should I do", call get_pending_tasks() first.
   When user asks to schedule/remind a pending task, call get_pending_tasks() first,
   then schedule_pending_task(task_text) for the selected task.

5. Return {{"final":"..."}} once when done. Be warm and specific.

6. Max {MAX_TOOL_CALLS} tool calls per turn.

7. Lines starting with TOOL_RESULT: contain JSON output from the last tool — read carefully."""

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=700,
            system=SYSTEM, messages=messages
        )
        return _parse(r.content[0].text)

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        conv = [{"role": "system", "content": SYSTEM}] + messages
        r = OpenAI().chat.completions.create(
            model=os.getenv("OAI_MODEL", "gpt-4o-mini"),
            messages=conv, temperature=0
        )
        return _parse(r.choices[0].message.content)

    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

# ── Graph nodes (identical pattern to agent_graph.py) ─────────────────────
def _sig(a: dict) -> str:
    return json.dumps({"a": a.get("action"), "k": sorted((a.get("args") or {}).items())})

def planner_node(state: dict) -> dict:
    if state.get("actions_taken", 0) >= MAX_TOOL_CALLS:
        state["done"] = True
        state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(
            {"final": f"Reached {MAX_TOOL_CALLS} tool calls. Ask again if you need more."})})
        return state

    if state.get("last_tool_result") is not None:
        state.setdefault("messages", []).append({
            "role":    "user",
            "content": "TOOL_RESULT: " + json.dumps(state["last_tool_result"], ensure_ascii=False)
        })
        state["last_tool_result"] = None

    plan = _llm_plan(state.get("messages", []))

    if "final" in plan:
        state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(plan)})
        state["done"] = True
        return state

    if plan.get("action") in TOOLS and isinstance(plan.get("args"), dict):
        sig = _sig(plan)
        if sig == state.get("last_action_sig"):
            state["done"] = True
            state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(
                {"final": "Stopping to avoid repeating the same action."})})
            return state
        state["planned_action"] = plan
        state["last_action_sig"] = sig
        state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(plan)})
        return state

    state["done"] = True
    state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(
        {"final": "Couldn't determine the next action. Please rephrase."})})
    return state

def executor_node(state: dict) -> dict:
    action = state.pop("planned_action", None) or {}
    tool, args = action.get("action"), action.get("args", {})
    try:
        result = TOOLS[tool](**args)
        state["last_tool_result"] = {"tool": tool, "result": result}
        state["consec_errors"] = 0
    except Exception as e:
        state["last_tool_result"] = {"tool": tool, "error": str(e)}
        state["consec_errors"] = state.get("consec_errors", 0) + 1
        if state["consec_errors"] >= MAX_CONSEC_ERRORS:
            state["done"] = True
            state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(
                {"final": f"Stopping after repeated errors: {e}"})})
    state["actions_taken"] = state.get("actions_taken", 0) + 1
    state["steps"] = state.get("steps", 0) + 1
    return state

# ── Graph assembly ────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(dict)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner",
        lambda s: "executor" if (not s.get("done") and s.get("planned_action")) else END,
        {"executor": "executor", END: END})
    g.add_conditional_edges("executor",
        lambda s: END if s.get("done") else "planner",
        {"planner": "planner", END: END})
    return g.compile()

def new_state() -> dict:
    return {
        "messages": [], "steps": 0, "done": False,
        "last_tool_result": None, "planned_action": None,
        "actions_taken": 0, "consec_errors": 0, "last_action_sig": None,
    }