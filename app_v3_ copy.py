import os, json, sqlite3, uuid, tempfile, re, threading
from datetime import datetime, timedelta, timezone, time
from typing import Optional
from contextlib import contextmanager

import anthropic
import openai as _openai
import numpy as np
import pytz
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from icalendar import Calendar, Event, Alarm
from dateutil import parser as dtparse
from dotenv import load_dotenv

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
DB_PATH       = os.environ.get("RSHIP_DB", "rship.db")
TZ_LOCAL      = pytz.timezone(os.environ.get("TZ", "Asia/Kolkata"))
HAIKU         = "claude-haiku-4-5-20251001"
OAI_CHAT      = os.environ.get("OAI_MODEL", "gpt-4o-mini")
OAI_EMBED     = os.environ.get("OAI_EMBED_MODEL", "text-embedding-3-small")
print("OPENAI:", bool(OPENAI_KEY), flush=True)
print("ANTHROPIC:", bool(ANTHROPIC_KEY), flush=True)
AVATAR_COLORS = ["#5B7FE8","#E8734A","#52B788","#B06AB3",
                 "#F4A261","#2EC4B6","#E8487A","#7B68EE"]
THEME_PALETTE = ["#5B7FE8","#E8734A","#52B788","#B06AB3",
                 "#F4A261","#2EC4B6","#E8487A","#7B68EE",
                 "#A8C686","#F7B731","#C0392B","#16A085"]

DATE_SIGNAL = re.compile(
    r"\b(\d{1,2}(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"|next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|tomorrow|tonight|this\s+weekend|in\s+\d+\s+(days?|weeks?|months?)"
    r"|leaving|interview|exam|deadline|joining|moving|surgery|appointment"
    r"|offered|refer|connect|introduce|rejected|fired|broke\s*up|fight|argument)\b",
    re.IGNORECASE
)

# ── AI backend ─────────────────────────────────────────────────────────────
# Prefer Claude for chat if present; OpenAI is required for embeddings/STT.
USE_CLAUDE = bool(ANTHROPIC_KEY)
USE_OPENAI = bool(OPENAI_KEY)

if not USE_CLAUDE and not USE_OPENAI:
    raise RuntimeError("Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY.")
if not USE_OPENAI:
    raise RuntimeError("OPENAI_API_KEY is required for embeddings and Whisper STT.")

print(f"AI  : {'Claude Haiku' if USE_CLAUDE else f'OpenAI {OAI_CHAT}'}", flush=True)
print(f"EMB : OpenAI {OAI_EMBED}", flush=True)
print(f"CAL : iCloud CalDAV via tools_core.CalendarMapsTools", flush=True)

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if USE_CLAUDE else None
oai           = _openai.OpenAI(api_key=OPENAI_KEY)

# ── iCloud calendar (shared instance) ─────────────────────────────────────
try:
    from tools_core import CalendarMapsTools
    calendar_tools = CalendarMapsTools()
    CALENDAR_OK = True
    print("CAL : iCloud CalDAV connected ✓", flush=True)
except Exception as e:
    CALENDAR_OK = False
    print(f"CAL : not connected ({e}) — .ics fallback only", flush=True)

# ── Agent sessions (same pattern as Jarvis) ────────────────────────────────
AGENT_SESSIONS: dict = {}
AGENT_LOCK = threading.Lock()

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title="r.ship v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB ─────────────────────────────────────────────────────────────────────
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT,
            color TEXT, created_at TEXT NOT NULL, last_log_at TEXT
        );
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY, contact_id TEXT NOT NULL,
            text TEXT NOT NULL, embedding TEXT,
            ai_summary TEXT, follow_up_days INTEGER, follow_up_prompt TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS themes (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, color TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS log_themes (
            log_id TEXT NOT NULL, theme_id TEXT NOT NULL,
            PRIMARY KEY(log_id, theme_id),
            FOREIGN KEY(log_id)   REFERENCES logs(id)   ON DELETE CASCADE,
            FOREIGN KEY(theme_id) REFERENCES themes(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id             TEXT PRIMARY KEY,
            contact_id     TEXT,
            source_log_id  TEXT,
            text           TEXT NOT NULL,
            due_date       TEXT,
            remind_at      TEXT,
            status         TEXT DEFAULT 'pending',
            category       TEXT DEFAULT 'other',
            priority       INTEGER DEFAULT 3,
            auto_schedule  INTEGER DEFAULT 0,
            alert_minutes_before INTEGER DEFAULT 30,
            calendar_href  TEXT,
            scheduled_at   TEXT,
            dedupe_key     TEXT UNIQUE,
            created_at     TEXT NOT NULL,
            FOREIGN KEY(contact_id)    REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(source_log_id) REFERENCES logs(id)     ON DELETE SET NULL
        );
        """)
        # Lightweight migration for existing r.ship DBs. SQLite cannot easily alter
        # constraints, so we add only missing columns and enforce dedupe in app logic.
        existing = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        migrations = {
            "remind_at": "ALTER TABLE tasks ADD COLUMN remind_at TEXT",
            "priority": "ALTER TABLE tasks ADD COLUMN priority INTEGER DEFAULT 3",
            "auto_schedule": "ALTER TABLE tasks ADD COLUMN auto_schedule INTEGER DEFAULT 0",
            "alert_minutes_before": "ALTER TABLE tasks ADD COLUMN alert_minutes_before INTEGER DEFAULT 30",
            "scheduled_at": "ALTER TABLE tasks ADD COLUMN scheduled_at TEXT",
            "dedupe_key": "ALTER TABLE tasks ADD COLUMN dedupe_key TEXT",
        }
        for col, sql in migrations.items():
            if col not in existing:
                c.execute(sql)
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_dedupe_key ON tasks(dedupe_key)")

init_db()

# ── Helpers ────────────────────────────────────────────────────────────────
def r2d(row) -> dict:   return dict(row) if row else {}
def embed(text: str) -> list:
    """OpenAI text-embedding-3-small. This avoids local SentenceTransformer download/load hangs on Windows."""
    r = oai.embeddings.create(input=text, model=OAI_EMBED)
    vec = np.array(r.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()
def cosine(a, b) -> float:
    return float(np.dot(np.array(a), np.array(b)))
def days_ago(iso: str | None) -> int | None:
    if not iso: return None
    return (datetime.now() - datetime.fromisoformat(iso)).days

def get_or_create_theme(conn, name: str) -> str:
    row = conn.execute("SELECT id FROM themes WHERE LOWER(name)=LOWER(?)", (name,)).fetchone()
    if row: return row["id"]
    tid = str(uuid.uuid4())
    used = [r["color"] for r in conn.execute("SELECT color FROM themes")]
    color = next((c for c in THEME_PALETTE if c not in used), THEME_PALETTE[len(used) % len(THEME_PALETTE)])
    conn.execute("INSERT INTO themes VALUES (?,?,?)", (tid, name.strip().title(), color))
    return tid

def attach_themes(conn, log_id: str, names: list[str]):
    for name in names:
        if name.strip():
            conn.execute("INSERT OR IGNORE INTO log_themes VALUES (?,?)",
                         (log_id, get_or_create_theme(conn, name)))

def log_themes_for(conn, log_id: str) -> list[dict]:
    return [r2d(r) for r in conn.execute(
        "SELECT t.id,t.name,t.color FROM themes t "
        "JOIN log_themes lt ON lt.theme_id=t.id WHERE lt.log_id=?", (log_id,)
    )]

def enrich_log(conn, row) -> dict:
    d = r2d(row); d["themes"] = log_themes_for(conn, d["id"]); return d

# ── AI dispatch ────────────────────────────────────────────────────────────
def _ai(prompt: str, max_tokens: int = 500) -> str:
    if USE_CLAUDE:
        r = claude_client.messages.create(
            model=HAIKU, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.content[0].text.strip()
    r = oai.chat.completions.create(
        model=OAI_CHAT, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

def _parse_json(raw: str) -> dict | list:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$",          "", s)
    return json.loads(s.strip())

# ── AI functions ───────────────────────────────────────────────────────────
def ai_enhance_log(text: str, contact_name: str) -> dict:
    """
    Single Haiku call returns: summary + follow-up + themes + tasks.
    Tasks are auto-extracted here — zero extra API cost.
    """
    try:
        raw = _ai(f"""You are a personal relationship assistant. Analyse this log.

Contact: {contact_name}
Log: "{text}"
Today: {datetime.now().strftime('%Y-%m-%d')}

Reply ONLY with valid JSON (no markdown):
{{
  "summary": "one-sentence key fact",
  "follow_up_days": <integer or null>,
  "follow_up_prompt": "natural follow-up question or null",
  "suggested_themes": ["Theme1"],
  "tasks": [
    {{
      "text": "actionable task text",
      "due_date": "YYYY-MM-DD or null",
      "category": "gift|call|follow_up|life_admin|referral|other",
      "priority": 1,
      "remind_at": "YYYY-MM-DDTHH:MM:SS+05:30 or null",
      "auto_schedule": true,
      "alert_minutes_before": 30
    }}
  ]
}}

Theme labels: 1-3 short labels like Referrals, Career moves, Romantic, Meetups,
College friends, Health, Travel, Business, Family.
Tasks: ONLY extract clearly actionable items (e.g. "Buy birthday gift for X",
"Follow up about Amazon referral"). Most logs won't have tasks — return [] if none.
Priority: 1 = urgent/important, 2 = important, 3 = normal, 4 = low.
Set auto_schedule=true only when the task is time-sensitive and a reminder would clearly help.
Use remind_at for the best real-life reminder time; if unsure, set null.""", 500)
        return _parse_json(raw)
    except Exception as e:
        print(f"ai_enhance_log: {e}")
        return {"summary": None, "follow_up_days": None, "follow_up_prompt": None,
                "suggested_themes": [], "tasks": []}

def ai_scan_alerts(contact_name: str, logs: list[str]) -> list[dict]:
    logs_text = "\n".join(f"- {l}" for l in logs)
    try:
        raw = _ai(f"""Relationship assistant. Find time-sensitive signals in these logs.

Contact: {contact_name}
Logs:
{logs_text}
Today: {datetime.now().strftime('%Y-%m-%d')}

Triggers: explicit dates, implicit urgency (leaving/joining), opportunities (referrals),
emotional moments (rejection/conflict). For each, decide optimal alert timing.

Reply ONLY with valid JSON:
{{"triggers":[{{"type":"explicit_date|implicit_urgency|opportunity|emotional",
"description":"signal summary","alert_date":"YYYY-MM-DD","alert_time":"HH:MM",
"event_title":"{contact_name} — action prompt (max 60 chars)",
"log_snippet":"relevant excerpt"}}]}}
Return {{"triggers":[]}} if nothing time-sensitive.""", 600)
        return _parse_json(raw).get("triggers", [])
    except Exception as e:
        print(f"ai_scan_alerts: {e}")
        return []

def ai_parse_intent(query: str) -> dict:
    try:
        raw = _ai(f"""Parse this search query for a relationship logbook.
Query: "{query}"
Reply ONLY with valid JSON:
{{"intent":"what the user is looking for","enriched_query":"verbose rewrite for semantic matching",
"is_nudge_request":<bool>,"name_hint":"partial name or null","topic_hint":"key topic or null"}}""")
        return _parse_json(raw)
    except:
        return {"intent": query, "enriched_query": query,
                "is_nudge_request": False, "name_hint": None, "topic_hint": None}

def ai_master_ask(question: str, logs: list, tasks: list, contact_names: list) -> str:
    """Full-context Q&A — reads ALL logs for mentioned contacts."""
    logs_text = "\n".join(
        f"[{l.get('contact_name','?')}, {l.get('created_at','')[:10]}]: {l['text']}"
        + (f"\n  ↩ Follow-up: {l['follow_up_prompt']}" if l.get("follow_up_prompt") else "")
        for l in logs[:60]
    ) or "No logs."

    tasks_text = "\n".join(
        f"- [{t.get('status','pending')}] {t['text']}"
        + (f" (due {t['due_date']})" if t.get("due_date") else "")
        + (f" — {t.get('contact_name','')}" if t.get("contact_name") else "")
        for t in tasks
    ) or "No tasks."

    about = f"about {', '.join(contact_names)}" if contact_names else "across all contacts"
    return _ai(f"""You are a thoughtful personal relationship assistant.

CONTEXT {about.upper()}:

LOG HISTORY:
{logs_text}

PENDING TASKS:
{tasks_text}

Answer this question helpfully, specifically, and warmly.
Reference actual details from the logs where relevant.

Question: {question}""", 800)

# ── STT: OpenAI Whisper ────────────────────────────────────────────────────
# NOTE: Swap only _transcribe() body when WisprFlow API is available.
def _transcribe(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    suffix = ".webm" if "webm" in mime else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes); f.flush()
        with open(f.name, "rb") as af:
            return oai.audio.transcriptions.create(
                model="whisper-1", file=af, language="en"
            ).text

# ── iCal fallback ─────────────────────────────────────────────────────────
def _to_utc(date_str: str, time_str: str) -> datetime:
    try:
        dt = TZ_LOCAL.localize(datetime.fromisoformat(f"{date_str}T{time_str}:00"))
        return dt.astimezone(timezone.utc).replace(microsecond=0)
    except:
        return datetime.now(timezone.utc)

def build_ical(triggers: list[dict]) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//r.ship//EN"); cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN"); cal.add("x-wr-calname", "r.ship Alerts")
    for t in triggers:
        try:
            dt = _to_utc(t["alert_date"], t.get("alert_time","09:00"))
            ev = Event()
            ev["uid"] = f"{uuid.uuid4()}@r.ship"
            ev.add("summary", t["event_title"])
            ev.add("dtstart", dt); ev.add("dtend", dt + timedelta(minutes=15))
            ev.add("dtstamp", datetime.now(timezone.utc))
            if t.get("log_snippet"): ev.add("description", t["log_snippet"])
            al = Alarm(); al.add("action","DISPLAY")
            al.add("description", t["event_title"])
            al.add("trigger", timedelta(minutes=-30))
            ev.add_component(al); cal.add_component(ev)
        except Exception as e:
            print(f"iCal event error: {e}")
    return cal.to_ical()


# ── Task alert scheduling helpers ─────────────────────────────────────────
def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def _coerce_bool(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return 0

def _dedupe_key(contact_id: str | None, source_log_id: str | None, text: str, due_date: str | None) -> str:
    base = "|".join([(contact_id or ""), (source_log_id or ""), text.strip().lower(), (due_date or "")])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

def _default_remind_at(due_date: str | None, priority: int = 3) -> str | None:
    if not due_date:
        return None
    try:
        d = dtparse.isoparse(due_date).date()
        hour = 9 if priority <= 2 else 10
        local_dt = TZ_LOCAL.localize(datetime.combine(d, time(hour=hour, minute=0)))
        return local_dt.isoformat()
    except Exception:
        return None

def _task_title(task: dict) -> str:
    who = task.get("contact_name") or "r.ship"
    text = re.sub(r"\s+", " ", task.get("text", "")).strip()
    return f"{who} — {text[:52]}"[:80]

def _schedule_one_task(conn, task: dict) -> dict:
    """Create one calendar alert for a pending task, with hard dedupe."""
    if task.get("calendar_href"):
        return {"status": "already_scheduled", "task_id": task["id"], "href": task.get("calendar_href")}
    if not CALENDAR_OK:
        return {"status": "calendar_unavailable", "task_id": task["id"]}

    priority = _safe_int(task.get("priority"), 3)
    start_iso = task.get("remind_at") or _default_remind_at(task.get("due_date"), priority)
    if not start_iso:
        return {"status": "no_reminder_time", "task_id": task["id"]}

    title = _task_title(task)
    desc = (
        f"Task: {task.get('text','')}\n"
        f"Category: {task.get('category','other')}\n"
        f"Priority: {priority}\n"
        f"Due: {task.get('due_date') or 'not set'}\n"
        f"Contact: {task.get('contact_name') or ''}\n"
        f"Source log: {task.get('source_log_id') or ''}"
    )
    result = calendar_tools.create_event(
        title=title,
        start_iso=start_iso,
        duration_minutes=15,
        alert_minutes_before=_safe_int(task.get("alert_minutes_before"), 30),
        description=desc,
    )
    href = result.get("href")
    conn.execute(
        "UPDATE tasks SET calendar_href=?, scheduled_at=?, remind_at=? WHERE id=?",
        (href, datetime.now().isoformat(), start_iso, task["id"]),
    )
    return {"status": "scheduled", "task_id": task["id"], "title": title, "href": href}

def _pending_schedulable_tasks(conn, auto_only: bool = True) -> list[dict]:
    q = ("SELECT t.*, c.name as contact_name FROM tasks t "
         "LEFT JOIN contacts c ON c.id=t.contact_id "
         "WHERE t.status='pending' AND t.calendar_href IS NULL "
         "AND (t.remind_at IS NOT NULL OR t.due_date IS NOT NULL)")
    if auto_only:
        q += " AND COALESCE(t.auto_schedule,0)=1"
    q += " ORDER BY COALESCE(t.remind_at, t.due_date) ASC, t.priority ASC LIMIT 100"
    return [r2d(r) for r in conn.execute(q)]

# ── Pydantic ───────────────────────────────────────────────────────────────
class ContactCreate(BaseModel):
    name: str; role: Optional[str] = None

class LogCreate(BaseModel):
    text: str

class ThemePatch(BaseModel):
    add: list[str] = []; remove: list[str] = []

class SearchQ(BaseModel):
    query: str

class AskQ(BaseModel):
    question: str

class TaskCreate(BaseModel):
    text: str
    contact_id: Optional[str] = None
    due_date: Optional[str] = None
    remind_at: Optional[str] = None
    category: Optional[str] = "other"
    priority: Optional[int] = 3
    auto_schedule: Optional[bool] = False
    alert_minutes_before: Optional[int] = 30

class TaskPatch(BaseModel):
    status: Optional[str] = None
    due_date: Optional[str] = None
    remind_at: Optional[str] = None
    priority: Optional[int] = None
    auto_schedule: Optional[bool] = None
    alert_minutes_before: Optional[int] = None
    calendar_href: Optional[str] = None

class AgentMessage(BaseModel):
    message: str
    session_id: str = "default"

# ── Routes: Static ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()

@app.get("/api/config")
def config():
    return {
        "llm": "claude-haiku" if USE_CLAUDE else OAI_CHAT,
        "calendar": CALENDAR_OK,
        "tz": str(TZ_LOCAL),
    }

# ── Routes: Contacts ───────────────────────────────────────────────────────
@app.get("/api/contacts")
def list_contacts():
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts ORDER BY name")]
        for ct in contacts:
            ct["themes"] = [r2d(r) for r in c.execute(
                "SELECT DISTINCT t.id,t.name,t.color FROM themes t "
                "JOIN log_themes lt ON lt.theme_id=t.id "
                "JOIN logs l ON l.id=lt.log_id WHERE l.contact_id=?", (ct["id"],)
            )]
        return contacts

@app.post("/api/contacts", status_code=201)
def create_contact(data: ContactCreate):
    with db() as c:
        n = c.execute("SELECT COUNT(*) as n FROM contacts").fetchone()["n"]
        cid = str(uuid.uuid4())
        c.execute("INSERT INTO contacts VALUES (?,?,?,?,?,?)",
                  (cid, data.name.strip(), data.role, AVATAR_COLORS[n % len(AVATAR_COLORS)],
                   datetime.now().isoformat(), None))
        return r2d(c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone())

@app.get("/api/contacts/{cid}")
def get_contact(cid: str):
    with db() as c:
        contact = c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact: raise HTTPException(404)
        logs  = [enrich_log(c, r) for r in
                 c.execute("SELECT * FROM logs WHERE contact_id=? ORDER BY created_at DESC", (cid,))]
        tasks = [r2d(r) for r in
                 c.execute("SELECT * FROM tasks WHERE contact_id=? ORDER BY due_date ASC NULLS LAST, created_at DESC", (cid,))]
        cd = r2d(contact)
        cd["themes"] = [r2d(r) for r in c.execute(
            "SELECT DISTINCT t.id,t.name,t.color FROM themes t "
            "JOIN log_themes lt ON lt.theme_id=t.id "
            "JOIN logs l ON l.id=lt.log_id WHERE l.contact_id=?", (cid,)
        )]
        return {"contact": cd, "logs": logs, "tasks": tasks}

@app.delete("/api/contacts/{cid}", status_code=204)
def delete_contact(cid: str):
    with db() as c:
        c.execute("DELETE FROM contacts WHERE id=?", (cid,))

# ── Routes: Logs ───────────────────────────────────────────────────────────
@app.post("/api/contacts/{cid}/logs", status_code=201)
def add_log(cid: str, data: LogCreate):
    with db() as c:
        contact = c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact: raise HTTPException(404)

        ai  = ai_enhance_log(data.text, contact["name"])
        emb = embed(data.text)
        lid = str(uuid.uuid4())
        now = datetime.now().isoformat()

        c.execute("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?)",
                  (lid, cid, data.text.strip(), json.dumps(emb),
                   ai.get("summary"), ai.get("follow_up_days"),
                   ai.get("follow_up_prompt"), now))
        attach_themes(c, lid, ai.get("suggested_themes", []))
        c.execute("UPDATE contacts SET last_log_at=? WHERE id=?", (now, cid))

        # Auto-extract tasks from this log
        saved_tasks = []
        for task_data in ai.get("tasks", []):
            if not task_data.get("text"): continue
            tid = str(uuid.uuid4())
            text = task_data["text"].strip()
            due_date = task_data.get("due_date")
            priority = max(1, min(4, _safe_int(task_data.get("priority"), 3)))
            remind_at = task_data.get("remind_at") or _default_remind_at(due_date, priority)
            auto_schedule = _coerce_bool(task_data.get("auto_schedule"))
            dedupe = _dedupe_key(cid, lid, text, due_date)
            c.execute("""
                INSERT OR IGNORE INTO tasks
                (id, contact_id, source_log_id, text, due_date, remind_at, status,
                 category, priority, auto_schedule, alert_minutes_before,
                 calendar_href, scheduled_at, dedupe_key, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (tid, cid, lid, text, due_date, remind_at, "pending",
                  task_data.get("category","other"), priority, auto_schedule,
                  _safe_int(task_data.get("alert_minutes_before"), 30),
                  None, None, dedupe, now))
            saved_tasks.append(task_data)

        log = enrich_log(c, c.execute("SELECT * FROM logs WHERE id=?", (lid,)).fetchone())
        return {"log": log, "ai": ai, "tasks_created": len(saved_tasks)}

@app.delete("/api/logs/{lid}", status_code=204)
def delete_log(lid: str):
    with db() as c:
        c.execute("DELETE FROM logs WHERE id=?", (lid,))

@app.patch("/api/logs/{lid}/themes")
def patch_log_themes(lid: str, data: ThemePatch):
    with db() as c:
        attach_themes(c, lid, data.add)
        for name in data.remove:
            row = c.execute("SELECT id FROM themes WHERE LOWER(name)=LOWER(?)", (name,)).fetchone()
            if row:
                c.execute("DELETE FROM log_themes WHERE log_id=? AND theme_id=?", (lid, row["id"]))
        return {"themes": log_themes_for(c, lid)}

# ── Routes: Themes ─────────────────────────────────────────────────────────
@app.get("/api/themes")
def list_themes():
    with db() as c:
        themes = [r2d(r) for r in c.execute("SELECT * FROM themes ORDER BY name")]
        for t in themes:
            t["contact_count"] = c.execute(
                "SELECT COUNT(DISTINCT l.contact_id) as n FROM logs l "
                "JOIN log_themes lt ON lt.log_id=l.id WHERE lt.theme_id=?", (t["id"],)
            ).fetchone()["n"]
        return themes

@app.get("/api/themes/{tid}/contacts")
def theme_contacts(tid: str):
    with db() as c:
        theme = c.execute("SELECT * FROM themes WHERE id=?", (tid,)).fetchone()
        if not theme: raise HTTPException(404)
        contacts = [r2d(r) for r in c.execute(
            "SELECT DISTINCT c.* FROM contacts c "
            "JOIN logs l ON l.contact_id=c.id "
            "JOIN log_themes lt ON lt.log_id=l.id "
            "WHERE lt.theme_id=? ORDER BY c.name", (tid,)
        )]
        for ct in contacts:
            latest = c.execute(
                "SELECT l.text,l.ai_summary FROM logs l "
                "JOIN log_themes lt ON lt.log_id=l.id "
                "WHERE l.contact_id=? AND lt.theme_id=? "
                "ORDER BY l.created_at DESC LIMIT 1", (ct["id"], tid)
            ).fetchone()
            ct["theme_log_text"]    = latest["text"]       if latest else None
            ct["theme_log_summary"] = latest["ai_summary"] if latest else None
        return {"theme": r2d(theme), "contacts": contacts}

# ── Routes: Tasks ──────────────────────────────────────────────────────────
@app.get("/api/tasks")
def list_tasks(status: str = "", contact_id: str = ""):
    with db() as c:
        q = ("SELECT t.*, c.name as contact_name, c.color as contact_color "
             "FROM tasks t LEFT JOIN contacts c ON c.id=t.contact_id WHERE 1=1")
        params = []
        if status:      q += " AND t.status=?";     params.append(status)
        if contact_id:  q += " AND t.contact_id=?"; params.append(contact_id)
        q += " ORDER BY t.due_date ASC NULLS LAST, t.created_at DESC"
        return [r2d(r) for r in c.execute(q, params)]

@app.post("/api/tasks", status_code=201)
def create_task(data: TaskCreate):
    with db() as c:
        tid = str(uuid.uuid4())
        text = data.text.strip()
        priority = max(1, min(4, data.priority or 3))
        remind_at = data.remind_at or _default_remind_at(data.due_date, priority)
        dedupe = _dedupe_key(data.contact_id, None, text, data.due_date)
        c.execute("""
            INSERT OR IGNORE INTO tasks
            (id, contact_id, source_log_id, text, due_date, remind_at, status,
             category, priority, auto_schedule, alert_minutes_before,
             calendar_href, scheduled_at, dedupe_key, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (tid, data.contact_id, None, text, data.due_date, remind_at, "pending",
              data.category or "other", priority, _coerce_bool(data.auto_schedule),
              data.alert_minutes_before or 30, None, None, dedupe, datetime.now().isoformat()))
        return r2d(c.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())

@app.patch("/api/tasks/{tid}")
def patch_task(tid: str, data: TaskPatch):
    with db() as c:
        task = c.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if not task: raise HTTPException(404)
        if data.status is not None:
            c.execute("UPDATE tasks SET status=? WHERE id=?", (data.status, tid))
        if data.due_date is not None:
            c.execute("UPDATE tasks SET due_date=? WHERE id=?", (data.due_date, tid))
        if data.remind_at is not None:
            c.execute("UPDATE tasks SET remind_at=? WHERE id=?", (data.remind_at, tid))
        if data.priority is not None:
            c.execute("UPDATE tasks SET priority=? WHERE id=?", (max(1, min(4, data.priority)), tid))
        if data.auto_schedule is not None:
            c.execute("UPDATE tasks SET auto_schedule=? WHERE id=?", (_coerce_bool(data.auto_schedule), tid))
        if data.alert_minutes_before is not None:
            c.execute("UPDATE tasks SET alert_minutes_before=? WHERE id=?", (data.alert_minutes_before, tid))
        if data.calendar_href is not None:
            c.execute("UPDATE tasks SET calendar_href=? WHERE id=?", (data.calendar_href, tid))
        return r2d(c.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())

@app.delete("/api/tasks/{tid}", status_code=204)
def delete_task(tid: str):
    with db() as c:
        c.execute("DELETE FROM tasks WHERE id=?", (tid,))

# ── Routes: Transcribe ─────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not oai: raise HTTPException(503, "OpenAI key required for STT")
    text = _transcribe(await audio.read(), audio.content_type or "audio/webm")
    return {"transcript": text}


@app.post("/api/tasks/schedule_pending")
def schedule_pending_tasks(auto_only: bool = True):
    """Schedule pending task reminders into Apple Calendar with dedupe."""
    scheduled, skipped, errors = [], [], []
    with db() as c:
        tasks = _pending_schedulable_tasks(c, auto_only=auto_only)
        for task in tasks:
            try:
                res = _schedule_one_task(c, task)
                if res.get("status") == "scheduled":
                    scheduled.append(res)
                else:
                    skipped.append(res)
            except Exception as e:
                errors.append({"task_id": task.get("id"), "error": str(e)})
    return {"scheduled": len(scheduled), "skipped": skipped, "errors": errors, "events": scheduled}

# ── Routes: Alerts ─────────────────────────────────────────────────────────
@app.get("/api/alerts/preview")
def preview_alerts():
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_triggers = []
        for contact in contacts:
            logs = [r2d(r) for r in c.execute(
                "SELECT text FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 5",
                (contact["id"],)
            )]
            if not logs: continue
            if not DATE_SIGNAL.search(" ".join(l["text"] for l in logs)): continue
            triggers = ai_scan_alerts(contact["name"], [l["text"] for l in logs])
            for t in triggers:
                t["contact_name"] = contact["name"]
            all_triggers.extend(triggers)
    return {"triggers": all_triggers, "scanned": len(contacts)}

@app.post("/api/alerts/schedule")
def schedule_alerts():
    """
    Push auto task reminders + relationship alerts directly to Apple Calendar.
    Falls back to .ics download for AI-scanned relationship triggers if CalDAV is unavailable.
    """
    task_schedule_result = {"scheduled": 0, "events": [], "skipped": [], "errors": []}
    with db() as c:
        for task in _pending_schedulable_tasks(c, auto_only=True):
            try:
                res = _schedule_one_task(c, task)
                if res.get("status") == "scheduled":
                    task_schedule_result["events"].append(res)
                    task_schedule_result["scheduled"] += 1
                else:
                    task_schedule_result["skipped"].append(res)
            except Exception as e:
                task_schedule_result["errors"].append({"task_id": task.get("id"), "error": str(e)})
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_triggers = []
        for contact in contacts:
            logs = [r2d(r) for r in c.execute(
                "SELECT text FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 5",
                (contact["id"],)
            )]
            if not logs: continue
            if not DATE_SIGNAL.search(" ".join(l["text"] for l in logs)): continue
            triggers = ai_scan_alerts(contact["name"], [l["text"] for l in logs])
            for t in triggers:
                t["contact_name"] = contact["name"]
                t["contact_id"]   = contact["id"]
            all_triggers.extend(triggers)

    if not all_triggers:
        return {"message": "No time-sensitive relationship signals found.", "scheduled": task_schedule_result["scheduled"], "task_alerts": task_schedule_result}

    if CALENDAR_OK:
        # Push directly to Apple Calendar
        scheduled, errors = [], []
        for t in all_triggers:
            try:
                dt_utc  = _to_utc(t["alert_date"], t.get("alert_time", "09:00"))
                dt_ist  = dt_utc.astimezone(TZ_LOCAL)
                iso_ist = dt_ist.strftime("%Y-%m-%dT%H:%M:%S+05:30")
                desc    = f"{t.get('log_snippet','')}\n\nType: {t.get('type','')}\nContact: {t.get('contact_name','')}"
                result  = calendar_tools.create_event(
                    title              = t["event_title"],
                    start_iso          = iso_ist,
                    duration_minutes   = 15,
                    alert_minutes_before = 30,
                    description        = desc,
                )
                scheduled.append({"event_title": t["event_title"], "href": result.get("href")})
            except Exception as e:
                errors.append({"event_title": t["event_title"], "error": str(e)})
        return {
            "mode":      "apple_calendar",
            "scheduled": len(scheduled) + task_schedule_result["scheduled"],
            "task_alerts": task_schedule_result,
            "relationship_alerts_scheduled": len(scheduled),
            "errors":    errors + task_schedule_result["errors"],
            "events":    scheduled,
        }
    else:
        # Fallback: return .ics
        ical_bytes = build_ical(all_triggers)
        return Response(
            content   = ical_bytes,
            media_type = "text/calendar",
            headers   = {"Content-Disposition": "attachment; filename=rship_alerts.ics"}
        )

# ── Routes: Master Ask ─────────────────────────────────────────────────────
@app.post("/api/ask")
def master_ask(data: AskQ):
    """
    AI Q&A over FULL log history — not just last 5.
    Powers: 'What to gift Navleen?', 'Anything pending?', 'Who should I call?'
    """
    question = data.question.strip()
    if not question: raise HTTPException(400, "Question required")

    # Step 1: Who is this question about?
    task_words = re.search(r"\b(pending|due|task|tasks|to[- ]?do|follow up|remind|anything left)\b", question, re.I)
    try:
        parse = _parse_json(_ai(f"""Parse this question for a relationship assistant.
Question: "{question}"
Reply ONLY with valid JSON:
{{"contact_names":["names mentioned"],"is_tasks_query":<bool>,"is_nudge_query":<bool>}}"""))
    except:
        parse = {"contact_names": [], "is_tasks_query": False, "is_nudge_query": False}

    with db() as c:
        context_logs, context_tasks, found_names = [], [], []

        # Fetch ALL logs for mentioned contacts
        for name in parse.get("contact_names", []):
            hits = [r2d(r) for r in c.execute(
                "SELECT * FROM contacts WHERE LOWER(name) LIKE ?", (f"%{name.lower()}%",)
            )]
            for contact in hits:
                found_names.append(contact["name"])
                context_logs.extend([r2d(r) for r in c.execute(
                    "SELECT l.text, l.ai_summary, l.follow_up_prompt, l.created_at, "
                    "c.name as contact_name FROM logs l "
                    "JOIN contacts c ON c.id=l.contact_id "
                    "WHERE l.contact_id=? ORDER BY l.created_at DESC", (contact["id"],)
                )])
                context_tasks.extend([r2d(r) for r in c.execute(
                    "SELECT t.*, c.name as contact_name FROM tasks t "
                    "LEFT JOIN contacts c ON c.id=t.contact_id WHERE t.contact_id=?",
                    (contact["id"],)
                )])

        # For generic questions — pull global tasks + follow-up prompts
        if not found_names or parse.get("is_tasks_query") or task_words:
            context_tasks = [r2d(r) for r in c.execute(
                "SELECT t.*, c.name as contact_name FROM tasks t "
                "LEFT JOIN contacts c ON c.id=t.contact_id "
                "WHERE t.status='pending' ORDER BY t.due_date ASC NULLS LAST LIMIT 50"
            )]
            context_logs = [r2d(r) for r in c.execute(
                "SELECT l.text, l.follow_up_prompt, l.follow_up_days, l.created_at, "
                "c.name as contact_name FROM logs l "
                "JOIN contacts c ON c.id=l.contact_id "
                "WHERE l.follow_up_prompt IS NOT NULL "
                "ORDER BY l.created_at DESC LIMIT 30"
            )]

    answer = ai_master_ask(question, context_logs, context_tasks, found_names)
    return {
        "answer":        answer,
        "contacts_used": found_names,
        "logs_read":     len(context_logs),
        "tasks_checked": len(context_tasks),
    }

# ── Routes: Agent (LangGraph — same pattern as Jarvis) ─────────────────────
@app.post("/api/agent")
def run_agent(data: AgentMessage):
    """
    Scheduling agent with iCloud calendar access + r.ship DB context.
    Try: 'Schedule a catch-up with Rahul next Tuesday'
         'What's on my calendar tomorrow?'
         'Anything pending for Priya?'
    """
    from rship_agent import build_agent, new_state

    with AGENT_LOCK:
        state = AGENT_SESSIONS.get(data.session_id) or new_state()

    state["messages"].append({"role": "user", "content": data.message})

    try:
        state = build_agent().invoke(state, config={"recursion_limit": 40})
    except Exception as e:
        return {"response": f"Agent error: {e}", "actions_taken": 0}

    final_msg = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)
    try:
        payload = json.loads(final_msg["content"]) if final_msg else {"final": "(no response)"}
    except:
        payload = {"final": "Agent returned non-JSON."}

    resp = {
        "response":      payload.get("final", "(no final)"),
        "actions_taken": state.get("actions_taken", 0),
    }

    # Reset per-turn state, keep conversation memory
    state.update({"last_tool_result": None, "steps": 0, "done": False,
                  "actions_taken": 0, "planned_action": None})

    with AGENT_LOCK:
        AGENT_SESSIONS[data.session_id] = state

    return resp

@app.delete("/api/agent/session/{session_id}", status_code=204)
def clear_agent_session(session_id: str):
    """Clear agent conversation history."""
    with AGENT_LOCK:
        AGENT_SESSIONS.pop(session_id, None)

# ── Routes: Search ─────────────────────────────────────────────────────────
def _is_master_answer_query(question: str, intent: dict | None = None) -> bool:
    """Questions that need a synthesized answer, not only ranked contact retrieval."""
    q = (question or "").strip().lower()
    if not q:
        return False
    return bool(re.search(
        r"\b(gift|suggest|suggestion|recommend|recommendation|what should|what to|ideas?|pending|due|task|tasks|to[- ]?do|follow up|remind|anything left|who should i|summary|summarise|summarize|advice)\b",
        q,
        re.I,
    )) or bool((intent or {}).get("is_nudge_request"))


def _find_contacts_for_question(conn, question: str, intent: dict | None = None) -> list[dict]:
    """Find mentioned contacts using both AI name_hint and literal contact-name matching."""
    q = (question or "").lower()
    contacts = [r2d(r) for r in conn.execute("SELECT * FROM contacts ORDER BY name")]
    hits, seen = [], set()

    name_hint = ((intent or {}).get("name_hint") or "").strip().lower()
    for ct in contacts:
        name = (ct.get("name") or "").lower()
        if not name:
            continue
        parts = [p for p in re.split(r"\s+", name) if len(p) >= 3]
        literal_hit = name in q or any(p in q for p in parts)
        hint_hit = bool(name_hint and (name_hint in name or name in name_hint))
        if literal_hit or hint_hit:
            if ct["id"] not in seen:
                hits.append(ct)
                seen.add(ct["id"])
    return hits


def _load_master_context(conn, contacts: list[dict], generic_tasks: bool = False) -> tuple[list, list, list]:
    """Return logs, tasks, names for master-answer mode."""
    context_logs, context_tasks, names = [], [], []
    for contact in contacts:
        names.append(contact["name"])
        context_logs.extend([r2d(r) for r in conn.execute(
            "SELECT l.text, l.ai_summary, l.follow_up_prompt, l.created_at, "
            "c.name as contact_name FROM logs l "
            "JOIN contacts c ON c.id=l.contact_id "
            "WHERE l.contact_id=? ORDER BY l.created_at DESC",
            (contact["id"],),
        )])
        context_tasks.extend([r2d(r) for r in conn.execute(
            "SELECT t.*, c.name as contact_name FROM tasks t "
            "LEFT JOIN contacts c ON c.id=t.contact_id WHERE t.contact_id=? "
            "ORDER BY t.status ASC, t.due_date ASC NULLS LAST",
            (contact["id"],),
        )])

    if generic_tasks or not contacts:
        context_tasks = [r2d(r) for r in conn.execute(
            "SELECT t.*, c.name as contact_name FROM tasks t "
            "LEFT JOIN contacts c ON c.id=t.contact_id "
            "WHERE t.status='pending' ORDER BY t.due_date ASC NULLS LAST LIMIT 50"
        )]
        context_logs = [r2d(r) for r in conn.execute(
            "SELECT l.text, l.follow_up_prompt, l.follow_up_days, l.created_at, "
            "c.name as contact_name FROM logs l "
            "JOIN contacts c ON c.id=l.contact_id "
            "WHERE l.follow_up_prompt IS NOT NULL "
            "ORDER BY l.created_at DESC LIMIT 30"
        )]
    return context_logs, context_tasks, names


@app.post("/api/search")
def search(data: SearchQ):
    """
    Master search:
    - For lookup queries, returns ranked contacts as before.
    - For recommendation / pending / advice questions, ALSO returns a synthesized answer.
      Frontend should display `answer` above `results` when present.
    """
    question = data.query.strip()
    if not question:
        return {"mode": "empty", "answer": None, "results": [], "intent": None}

    intent = ai_parse_intent(question)
    q_emb  = embed(intent.get("enriched_query", question))
    answer = None
    answer_contacts = []

    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_logs = [r2d(r) for r in c.execute(
            "SELECT id,contact_id,text,embedding,ai_summary,created_at FROM logs ORDER BY created_at DESC"
        )]

        mentioned_contacts = _find_contacts_for_question(c, question, intent)
        is_task_query = bool(re.search(r"\b(pending|due|task|tasks|to[- ]?do|follow up|remind|anything left)\b", question, re.I))
        is_answer_query = _is_master_answer_query(question, intent)

        if is_answer_query:
            logs_ctx, tasks_ctx, answer_contacts = _load_master_context(
                c,
                mentioned_contacts,
                generic_tasks=(is_task_query and not mentioned_contacts),
            )
            answer = ai_master_ask(question, logs_ctx, tasks_ctx, answer_contacts)

    logs_by = {}
    for log in all_logs:
        cid = log["contact_id"]
        if len(logs_by.get(cid, [])) < 8:
            logs_by.setdefault(cid, []).append(log)

    mentioned_ids = {ct["id"] for ct in mentioned_contacts} if 'mentioned_contacts' in locals() else set()
    results = []
    for ct in contacts:
        best_score, best_log = 0.0, None
        for log in logs_by.get(ct["id"], []):
            if not log.get("embedding"):
                continue
            try:
                score = cosine(q_emb, json.loads(log["embedding"]))
            except Exception:
                continue
            if score > best_score:
                best_score, best_log = score, log

        name_hint = intent.get("name_hint") or ""
        if name_hint and name_hint.lower() in ct["name"].lower():
            best_score += 0.2
        if ct["id"] in mentioned_ids:
            best_score += 0.35
        if intent.get("is_nudge_request") and ct.get("last_log_at"):
            best_score += min(0.25, (days_ago(ct["last_log_at"]) or 0) * 0.008)

        # In answer mode with a specific person, avoid showing random low-quality matches.
        threshold = 0.10 if answer else 0.12
        if answer and mentioned_ids and ct["id"] not in mentioned_ids and best_score < 0.45:
            continue

        if best_score > threshold:
            results.append({"contact": ct, "score": round(best_score, 3),
                            "matched_log": best_log, "intent": intent})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "mode": "answer" if answer else "search",
        "answer": answer,
        "answer_contacts": answer_contacts,
        "results": results[:8],
        "intent": intent,
    }


@app.post("/api/master_search")
def master_search(data: SearchQ):
    """Explicit alias for the frontend if you want to separate Q&A from normal search later."""
    return search(data)

# ── Routes: Feed ───────────────────────────────────────────────────────────
@app.get("/api/feed")
def feed():
    with db() as c:
        logs = [r2d(r) for r in c.execute(
            "SELECT l.id, l.contact_id, l.text, l.ai_summary, l.created_at, "
            "c.name as contact_name, c.color as contact_color "
            "FROM logs l JOIN contacts c ON c.id=l.contact_id "
            "ORDER BY l.created_at DESC LIMIT 20"
        )]
        for log in logs:
            log["themes"] = log_themes_for(c, log["id"])
        return logs

# ── Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting r.ship v3 on http://127.0.0.1:8000", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")