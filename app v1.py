"""
r.ship v2 — Relationship Logbook
STT: OpenAI Whisper (swap to WisprFlow when available — change _transcribe() only)
AI:  Claude Haiku (primary) or OpenAI GPT (fallback) — auto-selected at startup
EMB: OpenAI text-embedding-3-small (replaces sentence-transformers; no local GPU needed)
CAL: icalendar — generates .ics for Apple Calendar (UTC-correct, IST-aware)
"""

import os, json, sqlite3, uuid, tempfile, re
from datetime import datetime, timedelta, timezone
from typing import Optional
from contextlib import contextmanager
from urllib.parse import urlsplit

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
DB_PATH       = "rship.db"
TZ_LOCAL      = pytz.timezone(os.environ.get("TZ", "Asia/Kolkata"))

# Claude model IDs
HAIKU   = "claude-haiku-4-5-20251001"
SONNET  = "claude-sonnet-4-5"          # upgrade path

# OpenAI model IDs (used when Claude key is absent)
OAI_CHAT  = os.environ.get("OAI_MODEL", "gpt-4o-mini")
OAI_EMBED = "text-embedding-3-small"

AVATAR_COLORS = ["#5B7FE8","#E8734A","#52B788","#B06AB3",
                 "#F4A261","#2EC4B6","#E8487A","#7B68EE"]
THEME_PALETTE = ["#5B7FE8","#E8734A","#52B788","#B06AB3",
                 "#F4A261","#2EC4B6","#E8487A","#7B68EE",
                 "#A8C686","#F7B731","#C0392B","#16A085"]

# Pre-filter regex — contacts whose logs match this get AI alert scan
DATE_SIGNAL = re.compile(
    r"\b(\d{1,2}(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"|next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|tomorrow|tonight|this\s+weekend|in\s+\d+\s+(days?|weeks?|months?)"
    r"|leaving|interview|exam|deadline|joining|moving|surgery|appointment"
    r"|offered|refer|connect|introduce"
    r"|rejected|fired|broke\s*up|fight|argument)\b",
    re.IGNORECASE
)

# ── AI Backend Selection ────────────────────────────────────────────────────
# Prefer Claude if key is set; fall back to OpenAI.
# Embeddings always use OpenAI (text-embedding-3-small) — no local model needed.

_forced = os.environ.get("LLM_BACKEND", "").lower()
USE_CLAUDE = (_forced == "claude") or (not _forced and bool(ANTHROPIC_KEY))
USE_OPENAI = bool(OPENAI_KEY)

if not USE_CLAUDE and not USE_OPENAI:
    raise RuntimeError(
        "No AI backend configured. Set ANTHROPIC_API_KEY (for Claude) "
        "and/or OPENAI_API_KEY (for OpenAI chat + embeddings)."
    )

if not USE_OPENAI:
    raise RuntimeError(
        "OPENAI_API_KEY is required for embeddings (text-embedding-3-small). "
        "Chat can still run on Claude alone, but embeddings need OpenAI."
    )

_backend = "Claude (Haiku)" if USE_CLAUDE else f"OpenAI ({OAI_CHAT})"
print(f"AI backend : {_backend}")
print(f"Embeddings : OpenAI {OAI_EMBED}")

# ── Init clients ───────────────────────────────────────────────────────────
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if USE_CLAUDE else None
oai           = _openai.OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="r.ship")
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
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            role        TEXT,
            color       TEXT,
            created_at  TEXT NOT NULL,
            last_log_at TEXT
        );
        CREATE TABLE IF NOT EXISTS logs (
            id               TEXT PRIMARY KEY,
            contact_id       TEXT NOT NULL,
            text             TEXT NOT NULL,
            embedding        TEXT,
            ai_summary       TEXT,
            follow_up_days   INTEGER,
            follow_up_prompt TEXT,
            created_at       TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS themes (
            id    TEXT PRIMARY KEY,
            name  TEXT NOT NULL UNIQUE,
            color TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS log_themes (
            log_id   TEXT NOT NULL,
            theme_id TEXT NOT NULL,
            PRIMARY KEY(log_id, theme_id),
            FOREIGN KEY(log_id)   REFERENCES logs(id)   ON DELETE CASCADE,
            FOREIGN KEY(theme_id) REFERENCES themes(id) ON DELETE CASCADE
        );
        """)

init_db()

# ── Helpers ────────────────────────────────────────────────────────────────
def r2d(row) -> dict:
    return dict(row) if row else {}

def embed(text: str) -> list:
    """OpenAI text-embedding-3-small — returns a normalised float list."""
    r = oai.embeddings.create(input=text, model=OAI_EMBED)
    vec = r.data[0].embedding
    # Normalise so cosine similarity == dot product (same as before)
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()

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

def attach_themes_to_log(conn, log_id: str, theme_names: list[str]):
    for name in theme_names:
        if not name.strip(): continue
        tid = get_or_create_theme(conn, name)
        conn.execute("INSERT OR IGNORE INTO log_themes VALUES (?,?)", (log_id, tid))

def log_themes_for(conn, log_id: str) -> list[dict]:
    return [r2d(r) for r in conn.execute(
        "SELECT t.id, t.name, t.color FROM themes t "
        "JOIN log_themes lt ON lt.theme_id=t.id WHERE lt.log_id=?", (log_id,)
    )]

def enrich_log(conn, log_row) -> dict:
    d = r2d(log_row)
    d["themes"] = log_themes_for(conn, d["id"])
    return d

# ── AI Dispatch ────────────────────────────────────────────────────────────
def _ai_call(prompt: str, max_tokens: int = 400) -> str:
    """
    Route a plain-text prompt to whichever LLM backend is configured.
    Claude is preferred; OpenAI is the fallback.
    """
    if USE_CLAUDE:
        r = claude_client.messages.create(
            model=HAIKU, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.content[0].text.strip()

    # OpenAI fallback
    r = oai.chat.completions.create(
        model=OAI_CHAT, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

# ── AI Calls ───────────────────────────────────────────────────────────────
def ai_enhance_log(text: str, contact_name: str) -> dict:
    """Single AI call: key fact + follow-up + theme suggestions."""
    try:
        raw = _ai_call(f"""You are a personal relationship assistant. Analyse this log entry.

Contact: {contact_name}
Log: "{text}"

Reply ONLY with valid JSON (no markdown fences):
{{
  "summary": "one sentence key fact about this person from this log",
  "follow_up_days": <integer or null>,
  "follow_up_prompt": "natural follow-up question, or null",
  "suggested_themes": ["Theme1", "Theme2"]
}}

Theme guidelines: pick 1-3 short labels like "Referrals", "Career moves", "Romantic",
"Meetups", "College friends", "Health", "Travel", "Business", "Family". Only suggest
themes clearly implied by the log content.""")
        return json.loads(raw)
    except Exception:
        return {"summary": None, "follow_up_days": None, "follow_up_prompt": None, "suggested_themes": []}


def ai_scan_for_alerts(contact_name: str, logs: list[str]) -> list[dict]:
    """AI analyses last 5 logs and returns calendar trigger events."""
    logs_text = "\n".join(f"- {l}" for l in logs)
    try:
        raw = _ai_call(f"""You are a relationship assistant helping someone stay thoughtful about people they care about.

Contact: {contact_name}
Their last few log entries:
{logs_text}

Today's date: {datetime.now().strftime('%Y-%m-%d')}

Identify any time-sensitive signals that warrant a reminder: explicit dates (exams, interviews,
appointments), implicit urgency (leaving soon, joining next week), opportunities (referrals,
introductions offered), or emotional moments (rejection, conflict, big news).

For each signal, decide the optimal reminder date/time (AI-decided: morning of the event,
or 1-2 days before, or a few days after an emotional moment for a check-in).

Reply ONLY with valid JSON (no markdown fences):
{{
  "triggers": [
    {{
      "type": "explicit_date|implicit_urgency|opportunity|emotional",
      "description": "brief description of the signal",
      "alert_date": "YYYY-MM-DD",
      "alert_time": "HH:MM",
      "event_title": "{contact_name} — [action prompt, max 60 chars]",
      "log_snippet": "the relevant log excerpt"
    }}
  ]
}}

If no time-sensitive signals found, return {{"triggers": []}}""", max_tokens=600)
        return json.loads(raw).get("triggers", [])
    except Exception:
        return []


def ai_parse_intent(query: str) -> dict:
    """AI parses search query for semantic enrichment."""
    try:
        raw = _ai_call(f"""Parse this search query for a personal relationship logbook app.

Query: "{query}"

Reply ONLY with valid JSON (no markdown fences):
{{
  "intent": "brief description of what the user is looking for",
  "enriched_query": "rewritten query optimised for semantic similarity — expand abbreviations, add synonyms, make it verbose",
  "is_nudge_request": <true if asking who to call/reconnect/check-in with>,
  "name_hint": "partial name if mentioned, else null",
  "topic_hint": "key topic/context if clear, else null"
}}""")
        return json.loads(raw)
    except Exception:
        return {"intent": query, "enriched_query": query, "is_nudge_request": False, "name_hint": None, "topic_hint": None}


# ── STT (Whisper) ──────────────────────────────────────────────────────────
# NOTE: Using OpenAI Whisper API. Swap _transcribe() body only when WisprFlow is available.

def _transcribe(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    suffix = ".webm" if "webm" in mime else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = oai.audio.transcriptions.create(
                model="whisper-1", file=af, language="en"
            )
    return result.text


# ── iCal ──────────────────────────────────────────────────────────────────
# Improved iCal building, mirroring the UTC-correct approach in tools_core.py:
#   - All datetimes stored/emitted as UTC
#   - Alert_time string interpreted in local TZ (TZ_LOCAL) before converting to UTC
#   - Alarm uses timedelta trigger (not string) — compatible with Apple Calendar

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _to_utc(iso: str) -> datetime:
    """
    Parse an ISO datetime string. If timezone-naive, assume TZ_LOCAL (default Asia/Kolkata).
    Returns a UTC datetime with microseconds stripped.
    """
    dt = dtparse.isoparse(iso)
    if dt.tzinfo is None:
        dt = TZ_LOCAL.localize(dt)
    return dt.astimezone(timezone.utc).replace(microsecond=0)

def _build_trigger_datetime(alert_date: str, alert_time: str) -> datetime:
    """
    Combine YYYY-MM-DD date + HH:MM time, interpret in TZ_LOCAL, return UTC datetime.
    Falls back to 09:00 local if alert_time is missing or malformed.
    """
    time_str = alert_time if alert_time else "09:00"
    try:
        iso = f"{alert_date}T{time_str}:00"
        return _to_utc(iso)
    except Exception:
        # Last-resort: midnight UTC on that date
        return _to_utc(f"{alert_date}T00:00:00")

def build_ical(all_triggers: list[dict]) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//r.ship//relationship-logbook//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", "r.ship Alerts")

    for t in all_triggers:
        try:
            dt_start = _build_trigger_datetime(
                t["alert_date"],
                t.get("alert_time", "09:00"),
            )
            dt_end = dt_start + timedelta(minutes=15)

            description_parts = []
            if t.get("log_snippet"):
                description_parts.append(t["log_snippet"])
            if t.get("type"):
                description_parts.append(f"Type: {t['type']}")
            if t.get("description"):
                description_parts.append(f"Signal: {t['description']}")
            if t.get("contact_name"):
                description_parts.append(f"Contact: {t['contact_name']}")
            description = "\n\n".join(description_parts)

            event = Event()
            event["uid"] = f"{uuid.uuid4()}@r.ship"
            event.add("summary", t["event_title"])
            event.add("dtstart", dt_start)
            event.add("dtend", dt_end)
            event.add("dtstamp", _utcnow())
            if description:
                event.add("description", description)

            # 30-minute pre-alert — uses timedelta (Apple Calendar / RFC 5545 compatible)
            alarm = Alarm()
            alarm.add("action", "DISPLAY")
            alarm.add("description", t["event_title"])
            alarm.add("trigger", timedelta(minutes=-30))
            event.add_component(alarm)

            cal.add_component(event)

        except Exception as e:
            print(f"iCal event error for '{t.get('event_title', '?')}': {e}")

    return cal.to_ical()


# ── Pydantic models ────────────────────────────────────────────────────────
class ContactCreate(BaseModel):
    name: str
    role: Optional[str] = None

class LogCreate(BaseModel):
    text: str

class ThemePatch(BaseModel):
    add: list[str] = []
    remove: list[str] = []

class SearchQ(BaseModel):
    query: str


# ── Routes: Static ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()


# ── Routes: Contacts ───────────────────────────────────────────────────────
@app.get("/api/contacts")
def list_contacts():
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts ORDER BY name")]
        for contact in contacts:
            contact["themes"] = [r2d(r) for r in c.execute(
                "SELECT DISTINCT t.id, t.name, t.color FROM themes t "
                "JOIN log_themes lt ON lt.theme_id=t.id "
                "JOIN logs l ON l.id=lt.log_id "
                "WHERE l.contact_id=?", (contact["id"],)
            )]
        return contacts

@app.post("/api/contacts", status_code=201)
def create_contact(data: ContactCreate):
    with db() as c:
        count = c.execute("SELECT COUNT(*) as n FROM contacts").fetchone()["n"]
        cid   = str(uuid.uuid4())
        color = AVATAR_COLORS[count % len(AVATAR_COLORS)]
        c.execute("INSERT INTO contacts VALUES (?,?,?,?,?,?)",
                  (cid, data.name.strip(), data.role, color, datetime.now().isoformat(), None))
        return r2d(c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone())

@app.get("/api/contacts/{cid}")
def get_contact(cid: str):
    with db() as c:
        contact = c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact: raise HTTPException(404)
        logs = [enrich_log(c, r) for r in
                c.execute("SELECT * FROM logs WHERE contact_id=? ORDER BY created_at DESC", (cid,))]
        contact_dict = r2d(contact)
        contact_dict["themes"] = [r2d(r) for r in c.execute(
            "SELECT DISTINCT t.id,t.name,t.color FROM themes t "
            "JOIN log_themes lt ON lt.theme_id=t.id "
            "JOIN logs l ON l.id=lt.log_id WHERE l.contact_id=?", (cid,)
        )]
        return {"contact": contact_dict, "logs": logs}

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

        ai   = ai_enhance_log(data.text, contact["name"])
        emb  = embed(data.text)
        lid  = str(uuid.uuid4())
        now  = datetime.now().isoformat()

        c.execute("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?)",
                  (lid, cid, data.text.strip(), json.dumps(emb),
                   ai.get("summary"), ai.get("follow_up_days"),
                   ai.get("follow_up_prompt"), now))

        attach_themes_to_log(c, lid, ai.get("suggested_themes", []))
        c.execute("UPDATE contacts SET last_log_at=? WHERE id=?", (now, cid))

        log = enrich_log(c, c.execute("SELECT * FROM logs WHERE id=?", (lid,)).fetchone())
        return {"log": log, "ai": ai}

@app.delete("/api/logs/{lid}", status_code=204)
def delete_log(lid: str):
    with db() as c:
        c.execute("DELETE FROM logs WHERE id=?", (lid,))

@app.patch("/api/logs/{lid}/themes", status_code=200)
def patch_log_themes(lid: str, data: ThemePatch):
    with db() as c:
        attach_themes_to_log(c, lid, data.add)
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
        for contact in contacts:
            latest = c.execute(
                "SELECT l.text, l.ai_summary FROM logs l "
                "JOIN log_themes lt ON lt.log_id=l.id "
                "WHERE l.contact_id=? AND lt.theme_id=? "
                "ORDER BY l.created_at DESC LIMIT 1",
                (contact["id"], tid)
            ).fetchone()
            contact["theme_log_text"]    = latest["text"]       if latest else None
            contact["theme_log_summary"] = latest["ai_summary"] if latest else None
        return {"theme": r2d(theme), "contacts": contacts}


# ── Routes: Transcribe ─────────────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = _transcribe(audio_bytes, audio.content_type or "audio/webm")
    return {"transcript": text}


# ── Routes: Alerts ─────────────────────────────────────────────────────────
@app.post("/api/alerts/scan")
def scan_alerts():
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_triggers = []

        for contact in contacts:
            logs = [r2d(r) for r in c.execute(
                "SELECT text FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 5",
                (contact["id"],)
            )]
            if not logs: continue

            combined = " ".join(l["text"] for l in logs)
            if not DATE_SIGNAL.search(combined):
                continue

            log_texts = [l["text"] for l in logs]
            triggers  = ai_scan_for_alerts(contact["name"], log_texts)

            for t in triggers:
                t["contact_name"] = contact["name"]
                t["contact_id"]   = contact["id"]
            all_triggers.extend(triggers)

    if not all_triggers:
        return Response(
            content=json.dumps({"message": "No time-sensitive signals found across your contacts."}),
            media_type="application/json"
        )

    ical_bytes = build_ical(all_triggers)
    return Response(
        content=ical_bytes,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=rship_alerts.ics"}
    )

@app.get("/api/alerts/preview")
def preview_alerts():
    """Returns triggers as JSON for display in UI before download."""
    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_triggers = []

        for contact in contacts:
            logs = [r2d(r) for r in c.execute(
                "SELECT text FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 5",
                (contact["id"],)
            )]
            if not logs: continue
            combined = " ".join(l["text"] for l in logs)
            if not DATE_SIGNAL.search(combined):
                continue
            log_texts = [l["text"] for l in logs]
            triggers  = ai_scan_for_alerts(contact["name"], log_texts)
            for t in triggers:
                t["contact_name"] = contact["name"]
            all_triggers.extend(triggers)

    return {"triggers": all_triggers, "scanned": len(contacts)}


# ── Routes: Search ─────────────────────────────────────────────────────────
@app.post("/api/search")
def search(data: SearchQ):
    if not data.query.strip(): return {"results": [], "intent": None}

    intent = ai_parse_intent(data.query)
    q_emb  = embed(intent.get("enriched_query", data.query))

    with db() as c:
        contacts = [r2d(r) for r in c.execute("SELECT * FROM contacts")]
        all_logs = [r2d(r) for r in c.execute(
            "SELECT id,contact_id,text,embedding,ai_summary,created_at "
            "FROM logs ORDER BY created_at DESC"
        )]

    logs_by_contact: dict[str, list] = {}
    for log in all_logs:
        cid = log["contact_id"]
        if len(logs_by_contact.get(cid, [])) < 5:
            logs_by_contact.setdefault(cid, []).append(log)

    results = []
    for contact in contacts:
        clogs = logs_by_contact.get(contact["id"], [])
        best_score, best_log = 0.0, None

        for log in clogs:
            if not log.get("embedding"): continue
            score = cosine(q_emb, json.loads(log["embedding"]))
            if score > best_score:
                best_score, best_log = score, log

        name_hint = intent.get("name_hint") or ""
        if name_hint and name_hint.lower() in contact["name"].lower():
            best_score += 0.2

        if intent.get("is_nudge_request") and contact.get("last_log_at"):
            d = days_ago(contact["last_log_at"]) or 0
            best_score += min(0.25, d * 0.008)

        if best_score > 0.12:
            results.append({
                "contact":     contact,
                "score":       round(best_score, 3),
                "matched_log": best_log,
                "intent":      intent,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:8], "intent": intent}


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


# ── Routes: Config (debug) ─────────────────────────────────────────────────
@app.get("/api/config")
def config_info():
    """Returns active backend info — useful for debugging deployments."""
    return {
        "llm_backend":       "claude" if USE_CLAUDE else "openai",
        "llm_model":         HAIKU if USE_CLAUDE else OAI_CHAT,
        "embedding_model":   OAI_EMBED,
        "local_timezone":    str(TZ_LOCAL),
    }