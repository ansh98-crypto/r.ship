import os, json, sqlite3, uuid, numpy as np
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DB_PATH = "rship.db"
HAIKU = "claude-haiku-4-5-20251001"
AVATAR_COLORS = ['#5B7FE8','#E8734A','#52B788','#B06AB3','#F4A261','#2EC4B6','#E8487A','#7B68EE']

# ── Init ──────────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
claude  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
app     = FastAPI(title="Rship")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Database ──────────────────────────────────────────────────────────────────
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT,
            cadence_days INTEGER,
            last_contacted_at TEXT,
            color TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT,
            ai_summary TEXT,
            follow_up_days INTEGER,
            follow_up_prompt TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        """)

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────
def row_to_dict(row) -> dict:
    return dict(row) if row else {}

def embed(text: str) -> list[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()

def cosine(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b))   # already normalized → dot == cosine

def days_since(iso: str | None) -> int | None:
    if not iso:
        return None
    return (datetime.now() - datetime.fromisoformat(iso)).days

# ── AI functions ──────────────────────────────────────────────────────────────
def ai_enhance_log(text: str, contact_name: str) -> dict:
    """Call Haiku to extract key fact + follow-up suggestion from a log."""
    try:
        resp = claude.messages.create(
            model=HAIKU, max_tokens=300,
            messages=[{"role": "user", "content":
                f"""You are a relationship assistant. Extract structured info from this log entry.

Contact: {contact_name}
Log: \"{text}\"

Reply with ONLY valid JSON — no markdown, no explanation:
{{
  "summary": "one-sentence key fact about this person",
  "follow_up_days": <integer days until follow-up makes sense, or null>,
  "follow_up_prompt": "natural question to ask on follow-up, or null"
}}"""
            }]
        )
        return json.loads(resp.content[0].text.strip())
    except Exception as e:
        print(f"AI enhance error: {e}")
        return {"summary": None, "follow_up_days": None, "follow_up_prompt": None}


def ai_understand_intent(query: str) -> dict:
    """Call Haiku to parse the search query into structured intent."""
    try:
        resp = claude.messages.create(
            model=HAIKU, max_tokens=250,
            messages=[{"role": "user", "content":
                f"""You are a search assistant for a personal relationship logbook app.
Parse this search query and return ONLY valid JSON:

Query: \"{query}\"

{{
  "intent": "brief description of what the user is looking for",
  "enriched_query": "rewritten query optimised for semantic similarity search",
  "is_nudge_request": <true if asking who to call / reconnect / check-in with>,
  "filters": {{
    "name_hint": "partial name if mentioned, else null",
    "topic_hint": "topic keyword if clear, else null"
  }}
}}"""
            }]
        )
        return json.loads(resp.content[0].text.strip())
    except Exception as e:
        print(f"AI intent error: {e}")
        return {"intent": query, "enriched_query": query, "is_nudge_request": False, "filters": {}}


def semantic_search_core(query: str, contacts: list[dict], logs_by_contact: dict) -> list[dict]:
    intent   = ai_understand_intent(query)
    search_q = intent.get("enriched_query", query)
    q_emb    = embed(search_q)

    results = []
    for contact in contacts:
        cid   = contact["id"]
        clogs = logs_by_contact.get(cid, [])   # already capped to last 5

        # Score 1: semantic similarity across logs
        best_score = 0.0
        best_log   = None
        for log in clogs:
            if log.get("embedding"):
                l_emb = json.loads(log["embedding"])
                score = cosine(q_emb, l_emb)
                if score > best_score:
                    best_score = score
                    best_log   = log

        # Score 2: name fuzzy boost
        name_hint = (intent.get("filters") or {}).get("name_hint") or ""
        if name_hint and name_hint.lower() in contact["name"].lower():
            best_score += 0.25

        # Score 3: nudge boost — overdue contacts surface higher
        if intent.get("is_nudge_request") and contact.get("cadence_days"):
            d = days_since(contact.get("last_contacted_at"))
            overdue = (d or contact["cadence_days"]) - contact["cadence_days"]
            if overdue >= 0:
                best_score += min(0.3, overdue * 0.01)

        # Only include if there's any signal
        if best_score > 0.10 or (intent.get("is_nudge_request") and contact.get("cadence_days")):
            results.append({
                "contact":    contact,
                "score":      round(best_score, 3),
                "matched_log": row_to_dict(best_log) if best_log else None,
                "intent":     intent,
            })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:8]

# ── Pydantic models ───────────────────────────────────────────────────────────
class ContactCreate(BaseModel):
    name: str
    role: Optional[str] = None
    cadence_days: Optional[int] = None

class LogCreate(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

# -- Contacts --
@app.get("/api/contacts")
def list_contacts():
    with db() as c:
        return [row_to_dict(r) for r in c.execute("SELECT * FROM contacts ORDER BY name")]

@app.post("/api/contacts", status_code=201)
def create_contact(data: ContactCreate):
    with db() as c:
        count = c.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        cid   = str(uuid.uuid4())
        c.execute(
            "INSERT INTO contacts VALUES (?,?,?,?,?,?,?)",
            (cid, data.name.strip(), data.role, data.cadence_days,
             None, AVATAR_COLORS[count % len(AVATAR_COLORS)], datetime.now().isoformat())
        )
        return row_to_dict(c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone())

@app.get("/api/contacts/{cid}")
def get_contact(cid: str):
    with db() as c:
        contact = c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact:
            raise HTTPException(404, "Contact not found")
        logs = c.execute(
            "SELECT id, contact_id, text, ai_summary, follow_up_days, follow_up_prompt, created_at "
            "FROM logs WHERE contact_id=? ORDER BY created_at DESC", (cid,)
        ).fetchall()
        return {"contact": row_to_dict(contact), "logs": [row_to_dict(l) for l in logs]}

@app.delete("/api/contacts/{cid}", status_code=204)
def delete_contact(cid: str):
    with db() as c:
        c.execute("DELETE FROM logs WHERE contact_id=?", (cid,))
        c.execute("DELETE FROM contacts WHERE id=?", (cid,))

@app.patch("/api/contacts/{cid}/contacted", status_code=200)
def mark_contacted(cid: str):
    with db() as c:
        c.execute("UPDATE contacts SET last_contacted_at=? WHERE id=?",
                  (datetime.now().isoformat(), cid))
        return {"ok": True}

# -- Logs --
@app.post("/api/contacts/{cid}/logs", status_code=201)
def add_log(cid: str, data: LogCreate):
    with db() as c:
        contact = c.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact:
            raise HTTPException(404, "Contact not found")

        log_embedding = embed(data.text)
        ai = ai_enhance_log(data.text, contact["name"])
        lid = str(uuid.uuid4())

        c.execute(
            "INSERT INTO logs VALUES (?,?,?,?,?,?,?,?)",
            (lid, cid, data.text.strip(), json.dumps(log_embedding),
             ai.get("summary"), ai.get("follow_up_days"), ai.get("follow_up_prompt"),
             datetime.now().isoformat())
        )
        c.execute("UPDATE contacts SET last_contacted_at=? WHERE id=?",
                  (datetime.now().isoformat(), cid))

        return {
            "log": row_to_dict(c.execute(
                "SELECT id,contact_id,text,ai_summary,follow_up_days,follow_up_prompt,created_at "
                "FROM logs WHERE id=?", (lid,)).fetchone()),
            "ai": ai
        }

@app.delete("/api/logs/{lid}", status_code=204)
def delete_log(lid: str):
    with db() as c:
        c.execute("DELETE FROM logs WHERE id=?", (lid,))

# -- Nudges --
@app.get("/api/nudges")
def get_nudges():
    with db() as c:
        contacts = [row_to_dict(r) for r in
                    c.execute("SELECT * FROM contacts WHERE cadence_days IS NOT NULL")]
        nudges = []
        for contact in contacts:
            d = days_since(contact.get("last_contacted_at"))
            overdue = (d if d is not None else contact["cadence_days"]) - contact["cadence_days"]
            if overdue >= 0:
                last_log = row_to_dict(c.execute(
                    "SELECT text FROM logs WHERE contact_id=? ORDER BY created_at DESC LIMIT 1",
                    (contact["id"],)).fetchone())
                nudges.append({**contact, "days_overdue": overdue, "last_log_text": last_log.get("text")})
        return sorted(nudges, key=lambda x: x["days_overdue"], reverse=True)

# -- Search --
@app.post("/api/search")
def search(data: SearchQuery):
    if not data.query.strip():
        return {"results": [], "intent": None}

    with db() as c:
        contacts = [row_to_dict(r) for r in c.execute("SELECT * FROM contacts")]
        all_logs = [row_to_dict(r) for r in
                    c.execute("SELECT * FROM logs ORDER BY created_at DESC")]

    logs_by_contact: dict[str, list] = {}
    for log in all_logs:
        cid = log["contact_id"]
        if len(logs_by_contact.get(cid, [])) < 5:
            logs_by_contact.setdefault(cid, []).append(log)

    results = semantic_search_core(data.query, contacts, logs_by_contact)
    return {"results": results}