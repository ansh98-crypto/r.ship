"""
rship_agent.py — Scheduling + context agent for r.ship

Enhanced version:
- Safer JSON parsing from LLM output
- Creates Apple Calendar events
- For person-specific reminders/events:
    1) creates/fetches contact
    2) creates relationship log
    3) creates linked pending task with remind_at/due_date/calendar_href
- Better pending-task lookup with today/overdue/upcoming filters
- Fixes OpenAI response parsing bug
"""

import os
import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, time
from typing import Optional, Any

import pytz
from langgraph.graph import StateGraph, START, END
from tools_core import CalendarMapsTools

# ── Config ────────────────────────────────────────────────────────────────
IST = pytz.timezone(os.getenv("TZ", "Asia/Kolkata"))
DB_PATH = os.getenv("RSHIP_DB", "rship.db")
MAX_TOOL_CALLS = 5
MAX_CONSEC_ERRORS = 2

AVATAR_COLORS = [
    "#5B7FE8", "#E8734A", "#52B788", "#B06AB3",
    "#F4A261", "#2EC4B6", "#E8487A", "#7B68EE"
]

PERSON_ADMIN_WORDS = {
    "dentist", "doctor", "bill", "electricity", "grocery", "groceries",
    "tax", "payment", "renew", "medicine", "gym", "passport", "bank",
    "emi", "rent", "insurance"
}

RELATIONSHIP_ACTION_WORDS = {
    "wish", "call", "message", "follow up", "ask", "tell", "meet",
    "check with", "birthday", "exam", "interview", "referral",
    "congratulate", "thank", "remind", "support", "apologize", "apologise"
}

# ── iCloud calendar ───────────────────────────────────────────────────────
_cal = CalendarMapsTools()

# ── DB helpers ────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def _now_iso() -> str:
    return datetime.now(IST).isoformat()


def _today_iso() -> str:
    return datetime.now(IST).date().isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return 0


def _dedupe_key(contact_id: str | None, source_log_id: str | None, text: str, due_date: str | None, remind_at: str | None = None) -> str:
    base = "|".join([
        contact_id or "",
        source_log_id or "",
        (text or "").strip().lower(),
        due_date or "",
        remind_at or "",
    ])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _parse_start_iso(start_iso: str) -> datetime:
    dt = datetime.fromisoformat(start_iso)
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(IST)


def _default_remind_at(due_date: str | None, priority: int = 3) -> str | None:
    if not due_date:
        return None
    try:
        d = datetime.fromisoformat(due_date).date()
        hour = 9 if priority <= 2 else 10
        return IST.localize(datetime.combine(d, time(hour=hour))).isoformat()
    except Exception:
        return None


def _choose_avatar_color(conn) -> str:
    try:
        n = conn.execute("SELECT COUNT(*) AS n FROM contacts").fetchone()["n"]
        return AVATAR_COLORS[n % len(AVATAR_COLORS)]
    except Exception:
        return AVATAR_COLORS[0]


def _embed_optional(text: str) -> Optional[str]:
    """
    Returns JSON-string embedding if OpenAI is available, else None.
    Keeping this local avoids importing app_v5_ and creating circular imports.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI
        import numpy as np

        model = os.getenv("OAI_EMBED_MODEL", "text-embedding-3-small")
        r = OpenAI(api_key=key).embeddings.create(input=text, model=model)
        vec = np.array(r.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec
        return json.dumps(vec.tolist())
    except Exception as e:
        print("Embedding skipped:", repr(e), flush=True)
        return None


def _get_or_create_contact(conn, contact_name: str) -> Optional[str]:
    name = (contact_name or "").strip()
    if not name:
        return None

    row = conn.execute(
        "SELECT * FROM contacts WHERE LOWER(name)=LOWER(?)",
        (name,),
    ).fetchone()

    if row:
        return row["id"]

    now = _now_iso()
    cid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO contacts VALUES (?,?,?,?,?,?)",
        (
            cid,
            name,
            None,
            _choose_avatar_color(conn),
            now,
            now,
        ),
    )
    return cid


def _create_log(conn, contact_id: str, text: str, summary: str | None = None) -> str:
    now = _now_iso()
    lid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO logs
        (id, contact_id, text, embedding, ai_summary, follow_up_days, follow_up_prompt, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            lid,
            contact_id,
            text.strip(),
            _embed_optional(text),
            summary or text.strip(),
            None,
            None,
            now,
        ),
    )
    conn.execute(
        "UPDATE contacts SET last_log_at=? WHERE id=?",
        (now, contact_id),
    )
    return lid


def _create_task(
    conn,
    text: str,
    contact_id: str | None = None,
    source_log_id: str | None = None,
    due_date: str | None = None,
    remind_at: str | None = None,
    category: str = "follow_up",
    priority: int = 2,
    auto_schedule: bool = True,
    alert_minutes_before: int = 15,
    calendar_href: str | None = None,
) -> str:
    now = _now_iso()
    tid = str(uuid.uuid4())
    priority = max(1, min(4, _safe_int(priority, 2)))
    remind_at = remind_at or _default_remind_at(due_date, priority)
    dedupe = _dedupe_key(contact_id, source_log_id, text, due_date, remind_at)

    conn.execute(
        """
        INSERT OR IGNORE INTO tasks
        (id, contact_id, source_log_id, text, due_date, remind_at, status,
         category, priority, auto_schedule, alert_minutes_before,
         calendar_href, scheduled_at, dedupe_key, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            tid,
            contact_id,
            source_log_id,
            text.strip(),
            due_date,
            remind_at,
            "pending",
            category or "other",
            priority,
            _coerce_bool(auto_schedule),
            _safe_int(alert_minutes_before, 15),
            calendar_href,
            now if calendar_href else None,
            dedupe,
            now,
        ),
    )

    row = conn.execute("SELECT id FROM tasks WHERE dedupe_key=?", (dedupe,)).fetchone()
    return row["id"] if row else tid


def _looks_relationship_specific(user_text: str, title: str, contact_name: str = "") -> bool:
    text = f"{user_text or ''} {title or ''} {contact_name or ''}".lower()

    if contact_name:
        return not any(w in text for w in PERSON_ADMIN_WORDS)

    if any(w in text for w in PERSON_ADMIN_WORDS):
        return False

    return any(w in text for w in RELATIONSHIP_ACTION_WORDS)


# ── r.ship DB tools ───────────────────────────────────────────────────────
def get_contact_context(contact_name: str) -> dict:
    """
    Return ALL recent logs + all tasks for the closest-matching contact.
    Call this BEFORE scheduling anything about a person.
    """
    with _conn() as c:
        hits = [dict(r) for r in c.execute(
            "SELECT * FROM contacts WHERE LOWER(name) LIKE ?",
            (f"%{contact_name.lower()}%",),
        )]

        if not hits:
            return {"error": f"No contact found matching '{contact_name}'"}

        contact = hits[0]

        logs = [dict(r) for r in c.execute(
            """
            SELECT text, ai_summary, follow_up_prompt, follow_up_days, created_at
            FROM logs
            WHERE contact_id=?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (contact["id"],),
        )]

        tasks = [dict(r) for r in c.execute(
            """
            SELECT text, due_date, remind_at, status, category, priority
            FROM tasks
            WHERE contact_id=?
            ORDER BY COALESCE(remind_at, due_date) ASC NULLS LAST
            """,
            (contact["id"],),
        )]

        return {
            "contact": contact["name"],
            "role": contact["role"] or "",
            "last_log_at": contact["last_log_at"] or "never",
            "log_count": len(logs),
            "logs": logs,
            "tasks": tasks,
        }


def get_pending_tasks(contact_name: str = "", filter: str = "") -> dict:
    """
    Pending tasks.
    filter can be: today, overdue, upcoming, all.
    """
    today = _today_iso()
    filter = (filter or "").strip().lower()

    where = ["t.status='pending'"]
    params: list[Any] = []

    if contact_name:
        where.append("LOWER(c.name) LIKE ?")
        params.append(f"%{contact_name.lower()}%")

    if filter == "today":
        where.append("(t.due_date=? OR substr(t.remind_at,1,10)=?)")
        params.extend([today, today])
    elif filter == "overdue":
        where.append("((t.due_date IS NOT NULL AND t.due_date < ?) OR (t.remind_at IS NOT NULL AND substr(t.remind_at,1,10) < ?))")
        params.extend([today, today])
    elif filter == "upcoming":
        where.append("((t.due_date IS NOT NULL AND t.due_date >= ?) OR (t.remind_at IS NOT NULL AND substr(t.remind_at,1,10) >= ?))")
        params.extend([today, today])

    query = f"""
        SELECT
            t.id,
            t.text,
            t.due_date,
            t.remind_at,
            t.category,
            t.priority,
            t.calendar_href,
            c.name as contact_name
        FROM tasks t
        LEFT JOIN contacts c ON c.id=t.contact_id
        WHERE {" AND ".join(where)}
        ORDER BY COALESCE(t.remind_at, t.due_date) ASC NULLS LAST, t.priority ASC
        LIMIT 50
    """

    with _conn() as c:
        rows = [dict(r) for r in c.execute(query, params)]

    return {
        "tasks": rows,
        "count": len(rows),
        "filter": filter or "all",
        "today": today,
    }


def mark_task_done(task_text: str) -> dict:
    """Mark task(s) done by partial text match."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE tasks SET status='done' WHERE status='pending' AND LOWER(text) LIKE ?",
            (f"%{task_text.lower()}%",),
        )
        c.commit()
        return {"updated": cur.rowcount, "matched_on": task_text}


def schedule_pending_task(task_text: str) -> dict:
    """Schedule a pending task by partial text match if it has remind_at or due_date."""
    with _conn() as c:
        task = c.execute(
            """
            SELECT t.*, c.name as contact_name
            FROM tasks t
            LEFT JOIN contacts c ON c.id=t.contact_id
            WHERE t.status='pending' AND LOWER(t.text) LIKE ?
            ORDER BY COALESCE(t.remind_at,t.due_date) ASC
            LIMIT 1
            """,
            (f"%{task_text.lower()}%",),
        ).fetchone()

        if not task:
            return {"error": f"No pending task found matching '{task_text}'"}

        task = dict(task)

        if task.get("calendar_href"):
            return {
                "scheduled": False,
                "reason": "already_scheduled",
                "href": task.get("calendar_href"),
            }

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
            alert_minutes_before=_safe_int(task.get("alert_minutes_before"), 30),
            description=(
                f"Task: {task.get('text')}\n"
                f"Category: {task.get('category')}\n"
                f"Priority: {task.get('priority')}"
            ),
        )

        c.execute(
            "UPDATE tasks SET calendar_href=?, scheduled_at=?, remind_at=? WHERE id=?",
            (result.get("href"), _now_iso(), start_iso, task["id"]),
        )
        c.commit()

        return {"scheduled": True, "title": title, "href": result.get("href"), "start_iso": start_iso}


def create_relationship_event(
    title: str,
    start_iso: str,
    contact_name: str = "",
    duration_minutes: int = 15,
    alert_minutes_before: int = 15,
    description: str = "",
    location: str | None = None,
    original_user_message: str = "",
    category: str = "follow_up",
) -> dict:
    """
    Calendar-first tool.
    If contact_name is present and this is relationship-specific, also writes:
    - contact
    - relationship log
    - linked pending task with remind_at and calendar_href
    """
    dt = _parse_start_iso(start_iso)
    start_iso = dt.isoformat()
    due_date = dt.date().isoformat()

    result = _cal.create_event(
        title=title,
        start_iso=start_iso,
        duration_minutes=duration_minutes,
        alert_minutes_before=alert_minutes_before,
        description=description or title,
        location=location,
    )

    relationship_log_created = False
    task_created = False
    contact_id = None
    log_id = None
    task_id = None

    if _looks_relationship_specific(original_user_message, title, contact_name) and contact_name:
        with _conn() as c:
            contact_id = _get_or_create_contact(c, contact_name)

            memory_text = description or title
            if original_user_message:
                memory_text = f"{contact_name}: {original_user_message.strip()}"

            log_id = _create_log(
                c,
                contact_id=contact_id,
                text=memory_text,
                summary=description or title,
            )

            task_text = title.strip() or f"Follow up with {contact_name}"
            task_id = _create_task(
                c,
                text=task_text,
                contact_id=contact_id,
                source_log_id=log_id,
                due_date=due_date,
                remind_at=start_iso,
                category=category or "follow_up",
                priority=2,
                auto_schedule=True,
                alert_minutes_before=alert_minutes_before,
                calendar_href=result.get("href"),
            )

            c.commit()
            relationship_log_created = True
            task_created = True

    return {
        "scheduled": True,
        "calendar": result,
        "title": title,
        "start_iso": start_iso,
        "relationship_log_created": relationship_log_created,
        "task_created": task_created,
        "contact_name": contact_name,
        "contact_id": contact_id,
        "log_id": log_id,
        "task_id": task_id,
    }


# ── Tool registry ─────────────────────────────────────────────────────────
TOOLS = {
    "create_event": create_relationship_event,
    "delete_event": _cal.delete_event,
    "list_events_today": _cal.list_events_today,
    "list_events_for_day": _cal.list_events_for_day,

    "get_contact_context": get_contact_context,
    "get_pending_tasks": get_pending_tasks,
    "mark_task_done": mark_task_done,
    "schedule_pending_task": schedule_pending_task,
}

# ── JSON helper ───────────────────────────────────────────────────────────
def _parse(text: str) -> dict:
    if not text:
        return {"final": "Agent returned an empty response."}

    s = str(text).strip()

    if not s:
        return {"final": "Agent returned a blank response."}

    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if block:
        s = block.group(1)
    else:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            s = match.group(0)

    try:
        return json.loads(s)
    except Exception as e:
        print("AGENT JSON PARSE FAILED:", repr(e), flush=True)
        print("RAW AGENT OUTPUT:", repr(text[:2000]), flush=True)
        return {"final": str(text).strip() or "Agent returned invalid JSON."}


# ── LLM planner ───────────────────────────────────────────────────────────
def _llm_plan(messages: list) -> dict:
    SYSTEM = f"""You are r.ship Assistant — a personal relationship and calendar agent.
Current time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}

You have two categories of tools:

CALENDAR TOOLS:
- create_event(title, start_iso, contact_name="", duration_minutes=15, alert_minutes_before=15, description="", location=null, original_user_message="", category="follow_up")
  → Creates Apple Calendar event.
  → If contact_name is provided, also creates a relationship log and linked pending task.
  → start_iso must be IST ISO 8601, e.g. "2026-05-18T21:30:00+05:30".

- delete_event(href)
- list_events_today()
- list_events_for_day(date_iso)

RELATIONSHIP DB TOOLS:
- get_contact_context(contact_name)
- get_pending_tasks(contact_name="", filter="")
  → filter can be "today", "overdue", "upcoming", or "" for all.
- mark_task_done(task_text)
- schedule_pending_task(task_text)

STRICT OUTPUT FORMAT:
Return only valid JSON. No prose. No markdown.

To call a tool:
{{"action":"<tool_name>","args":{{...}}}}

When done:
{{"final":"<message for user>"}}

RULES:
1. If the user asks what is pending / due / tasks today, call get_pending_tasks(filter="today").
2. If the user asks all pending tasks, call get_pending_tasks().
3. If the user asks to create a reminder/event for a specific person, call create_event directly.
4. For person-specific reminders, always include contact_name and original_user_message.
5. Do NOT include contact_name for personal admin events like dentist, bill, grocery, gym, tax, passport, rent, bank.
6. For relationship events, use a short action title:
   - "Wish Purushottam all the best"
   - "Follow up with Sachin about referral"
   - "Message Navleen about interview"
7. For relationship events, description should capture why it matters.
8. Default duration is 15 minutes for reminders and 60 minutes for meetings.
9. Default alert is 15 minutes before.
10. After a successful TOOL_RESULT, return a friendly final answer summarising what happened.
11. Max {MAX_TOOL_CALLS} tool calls per turn.

Examples:
User: "Wish Purushottam all the best today at 9.30 pm"
{{"action":"create_event","args":{{"title":"Wish Purushottam all the best","contact_name":"Purushottam","start_iso":"{_today_iso()}T21:30:00+05:30","duration_minutes":15,"alert_minutes_before":15,"description":"Wish Purushottam all the best.","original_user_message":"Wish Purushottam all the best today at 9.30 pm","category":"follow_up"}}}}

User: "Dentist appointment today at 6.30 pm"
{{"action":"create_event","args":{{"title":"Dentist appointment","start_iso":"{_today_iso()}T18:30:00+05:30","duration_minutes":60,"alert_minutes_before":15,"description":"Dentist appointment","original_user_message":"Dentist appointment today at 6.30 pm","category":"life_admin"}}}}

User: "pending tasks for today"
{{"action":"get_pending_tasks","args":{{"filter":"today"}}}}
"""

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        r = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=700,
            system=SYSTEM,
            messages=messages,
        )
        raw = r.content[0].text if r.content else ""
        print("LLM RAW:", repr(raw[:1000]), flush=True)
        return _parse(raw)

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        conv = [{"role": "system", "content": SYSTEM}] + messages
        r = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model=os.getenv("OAI_MODEL", "gpt-4o-mini"),
            messages=conv,
            temperature=0,
        )
        raw = r.choices[0].message.content or ""
        print("LLM RAW:", repr(raw[:1000]), flush=True)
        return _parse(raw)

    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


# ── Graph nodes ───────────────────────────────────────────────────────────
def _sig(a: dict) -> str:
    return json.dumps(
        {"a": a.get("action"), "k": sorted((a.get("args") or {}).items())},
        sort_keys=True,
        default=str,
    )


def planner_node(state: dict) -> dict:
    if state.get("actions_taken", 0) >= MAX_TOOL_CALLS:
        state["done"] = True
        state.setdefault("messages", []).append({
            "role": "assistant",
            "content": json.dumps({"final": f"Reached {MAX_TOOL_CALLS} tool calls. Ask again if you need more."}),
        })
        return state

    if state.get("last_tool_result") is not None:
        state.setdefault("messages", []).append({
            "role": "user",
            "content": "TOOL_RESULT: " + json.dumps(state["last_tool_result"], ensure_ascii=False, default=str),
        })
        state["last_tool_result"] = None

    plan = _llm_plan(state.get("messages", []))

    if "final" in plan:
        state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(plan)})
        state["done"] = True
        return state

    if plan.get("action") in TOOLS and isinstance(plan.get("args"), dict):
        sig = _sig(plan)

        # Allow finalisation after a repeated action; do not execute duplicate.
        if sig == state.get("last_action_sig"):
            state["done"] = True
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": json.dumps({"final": "I already handled that action, so I stopped to avoid duplicating it."}),
            })
            return state

        state["planned_action"] = plan
        state["last_action_sig"] = sig
        state.setdefault("messages", []).append({"role": "assistant", "content": json.dumps(plan)})
        return state

    state["done"] = True
    state.setdefault("messages", []).append({
        "role": "assistant",
        "content": json.dumps({"final": "Couldn't determine the next action. Please rephrase."}),
    })
    return state


def executor_node(state: dict) -> dict:
    action = state.pop("planned_action", None) or {}
    tool = action.get("action")
    args = action.get("args", {})

    try:
        if tool not in TOOLS:
            raise ValueError(f"Unknown tool: {tool}")

        result = TOOLS[tool](**args)
        state["last_tool_result"] = {"tool": tool, "args": args, "result": result}
        state["consec_errors"] = 0

    except Exception as e:
        print("TOOL EXECUTION ERROR:", repr(e), flush=True)
        state["last_tool_result"] = {"tool": tool, "args": args, "error": str(e)}
        state["consec_errors"] = state.get("consec_errors", 0) + 1

        if state["consec_errors"] >= MAX_CONSEC_ERRORS:
            state["done"] = True
            state.setdefault("messages", []).append({
                "role": "assistant",
                "content": json.dumps({"final": f"Stopping after repeated errors: {e}"}),
            })

    state["actions_taken"] = state.get("actions_taken", 0) + 1
    state["steps"] = state.get("steps", 0) + 1
    return state


# ── Graph assembly ────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(dict)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        lambda s: "executor" if (not s.get("done") and s.get("planned_action")) else END,
        {"executor": "executor", END: END},
    )
    g.add_conditional_edges(
        "executor",
        lambda s: END if s.get("done") else "planner",
        {"planner": "planner", END: END},
    )
    return g.compile()


def new_state() -> dict:
    return {
        "messages": [],
        "steps": 0,
        "done": False,
        "last_tool_result": None,
        "planned_action": None,
        "actions_taken": 0,
        "consec_errors": 0,
        "last_action_sig": None,
    }
