# tools_core.py
import os, uuid, json, requests, pytz, caldav
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
from urllib.parse import urlsplit, quote_plus

from icalendar import Calendar, Event, Alarm
from caldav.elements import dav
from dateutil import parser as dtparse

IST = pytz.timezone(os.getenv("TZ","Asia/Kolkata"))
from dotenv import load_dotenv

# Load environment variables from a local .env file into os.environ (if present).
# This lets you run the script without exporting variables in every shell session.
load_dotenv()

class CalendarMapsTools:
    """
    One clean class your agent can call directly.
    Later (or in parallel), wrap this with your MCP server (thin layer).
    """

    # ---------- iCloud helpers ----------
    def _principal(self):
        return caldav.DAVClient(
            url="https://caldav.icloud.com/",
            username=os.environ["ICLOUD_APPLE_ID"],
            password=os.environ["ICLOUD_APP_PASSWORD"],
        ).principal()

    def _event_cals(self):
        cals = []
        for c in self._principal().calendars():
            try:
                comps = c.get_supported_components() or []
            except Exception:
                comps = []
            if "VEVENT" in comps:
                cals.append(c)
        if not cals:
            raise RuntimeError("No VEVENT calendars found.")
        return cals

    def _pick_cal(self):
        want = (os.getenv("ICLOUD_CALENDAR_NAME","").strip().lower())
        for c in self._event_cals():
            try:
                props = c.get_properties([dav.DisplayName()])
                name = str(props.get("{DAV:}displayname","") or props.get("{DAV:}displayname$","")).strip()
            except Exception:
                name = ""
            if want and name.lower()==want:
                return c
        return self._event_cals()[0]

    def _obj_from_href(self, href:str):
        parts = urlsplit(href)
        client = caldav.DAVClient(
            url=f"{parts.scheme}://{parts.netloc}/",
            username=os.environ["ICLOUD_APPLE_ID"],
            password=os.environ["ICLOUD_APP_PASSWORD"],
        )
        return caldav.CalendarObjectResource(client=client, url=href)

    def _utcnow(self): 
        return datetime.now(timezone.utc).replace(microsecond=0)

    def _to_utc(self, iso:str)->datetime:
        dt = dtparse.isoparse(iso)
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        return dt.astimezone(timezone.utc).replace(microsecond=0)

    # ---------- ICS helpers ----------
    def _build_event_ics(self, title:str, start_utc:datetime, duration_min:int,
                         alert_before_min:int, description:Optional[str], location:Optional[str],
                         rrule:Optional[str])->bytes:
        end_utc = start_utc + timedelta(minutes=duration_min)
        cal = Calendar(); cal.add("prodid","-//Jarvis//EN"); cal.add("version","2.0")
        ev = Event()
        ev["uid"] = f"{uuid.uuid4()}@jarvis"
        ev.add("summary", title)
        if description: ev.add("description", description)
        if location:    ev.add("location", location)
        ev.add("dtstamp", self._utcnow())
        ev.add("dtstart", start_utc)
        ev.add("dtend", end_utc)
        if rrule: ev.add("rrule", rrule)
        al = Alarm(); al.add("action","DISPLAY"); al.add("description", title)
        al.add("trigger", timedelta(minutes=-alert_before_min))  # 0 => at start
        ev.add_component(al)
        cal.add_component(ev)
        return cal.to_ical()

    # ---------- Calendar API ----------
    def create_event(self, title:str, start_iso:str, duration_minutes:int=15,
                     alert_minutes_before:int=0, description:str|None=None,
                     location:str|None=None, rrule:str|None=None) -> Dict:
        start_utc = self._to_utc(start_iso)
        ics = self._build_event_ics(title, start_utc, duration_minutes, alert_minutes_before, description, location, rrule)
        created = self._pick_cal().add_event(ics)
        return {"href": str(created.url)}

    def delete_event(self, href:str) -> Dict:
        self._obj_from_href(href).delete()
        return {"deleted": True}

    def delete_by_search(self, title_contains:str, start_iso:str|None=None, end_iso:str|None=None, dry_run:bool=True)->Dict:
        start = self._to_utc(start_iso) if start_iso else self._utcnow()-timedelta(days=30)
        end   = self._to_utc(end_iso)   if end_iso   else self._utcnow()+timedelta(days=365)
        matches, deleted = [], 0
        for cal in self._event_cals():
            try: events = cal.date_search(start, end)
            except: continue
            for ev in events:
                v = getattr(ev, "vobject_instance", None)
                if not (v and hasattr(v,"vevent")): continue
                summary = getattr(v.vevent.summary, "value", "") or ""
                if title_contains.lower() in summary.lower():
                    href = str(ev.url)
                    matches.append(href)
                    if not dry_run:
                        try: ev.delete(); deleted += 1
                        except: pass
        return {"matches": matches, "deleted_count": deleted, "dry_run": dry_run}

    def skip_occurrence(self, href:str, occurrence_start_iso:str)->Dict:
        obj = self._obj_from_href(href)
        raw = obj.data
        if isinstance(raw,str): raw = raw.encode("utf-8",errors="replace")
        cal = Calendar.from_ical(raw)
        masters = [c for c in cal.walk("VEVENT") if c.get("RRULE") and not c.get("RECURRENCE-ID")]
        if not masters: raise ValueError("Not a recurring series (no RRULE).")
        master = masters[0]
        occ_utc = self._to_utc(occurrence_start_iso)
        master.add("exdate", occ_utc)
        obj.set_data(cal.to_ical())
        return {"href": href, "exdate_added_utc": occ_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}

    def reschedule(self, href:str, occurrence_start_iso:str, new_start_iso:str, new_duration_minutes:int=15)->Dict:
        obj = self._obj_from_href(href)
        raw = obj.data
        if isinstance(raw,str): raw = raw.encode("utf-8",errors="replace")
        cal = Calendar.from_ical(raw)
        masters = [c for c in cal.walk("VEVENT") if c.get("RRULE") and not c.get("RECURRENCE-ID")]
        if not masters: raise ValueError("Not a recurring series (no RRULE).")
        master = masters[0]; uid = str(master.get("UID"))
        old_start = self._to_utc(occurrence_start_iso)
        new_start = self._to_utc(new_start_iso)
        new_end   = new_start + timedelta(minutes=new_duration_minutes)
        ex = Event()
        ex.add("uid", uid)
        ex.add("recurrence-id", old_start)
        ex.add("dtstart", new_start)
        ex.add("dtend", new_end)
        if master.get("SUMMARY"):     ex.add("summary", master.get("SUMMARY"))
        if master.get("DESCRIPTION"): ex.add("description", master.get("DESCRIPTION"))
        if master.get("LOCATION"):    ex.add("location", master.get("LOCATION"))
        ex.add("dtstamp", self._utcnow())
        cal.add_component(ex)
        obj.set_data(cal.to_ical())
        return {
            "href": href,
            "recurrence_id_utc": old_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "new_start_utc": new_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "new_end_utc": new_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # ---------- Day views ----------
    def _day_bounds_utc(self, date_iso: str | None = None):
        """
        Returns (start_utc, end_utc) for the given date in IST (Asia/Kolkata).
        If date_iso is None -> today.
        """
        if date_iso:
            # Accept "YYYY-MM-DD" or any ISO-ish string; interpret as IST midnight if date-only.
            dt_local = dtparse.isoparse(date_iso)
            if dt_local.tzinfo is None:
                # Treat date-only as IST midnight
                dt_local = IST.localize(datetime(dt_local.year, dt_local.month, dt_local.day))
            else:
                # Convert whatever was provided to IST, then clamp to day
                dt_local = dt_local.astimezone(IST)
                dt_local = IST.localize(datetime(dt_local.year, dt_local.month, dt_local.day))
        else:
            now_ist = datetime.now(IST)
            dt_local = IST.localize(datetime(now_ist.year, now_ist.month, now_ist.day))

        start_ist = dt_local
        end_ist   = start_ist + timedelta(days=1) - timedelta(microseconds=1)  # inclusive day-end
        # Convert to UTC (strip microseconds like your other helpers do)
        start_utc = start_ist.astimezone(timezone.utc).replace(microsecond=0)
        end_utc   = end_ist.astimezone(timezone.utc).replace(microsecond=0)
        return start_utc, end_utc

    def _ical_dt_to_utc(self, vdt) -> datetime:
        """
        Normalize VEVENT dtstart/dtend (could be date or datetime, tz-aware or naive) to UTC.
        - DATE (all-day) -> interpret as local IST midnight unless end date, which is exclusive per RFC 5545.
        """
        # vdt may be date or datetime from icalendar/vobject
        if isinstance(vdt, datetime):
            if vdt.tzinfo is None:
                # Assume IST for naive datetimes
                vdt = IST.localize(vdt)
            return vdt.astimezone(timezone.utc).replace(microsecond=0)

        # All-day (DATE): interpret as IST midnight
        if hasattr(vdt, "year") and hasattr(vdt, "month") and hasattr(vdt, "day"):
            dt_local = IST.localize(datetime(vdt.year, vdt.month, vdt.day))
            return dt_local.astimezone(timezone.utc).replace(microsecond=0)

        # Fallback: now
        return self._utcnow()

    def _overlaps(self, a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        """True if [a_start, a_end] intersects [b_start, b_end]."""
        return not (a_end < b_start or b_end < a_start)

    def list_events_for_day(self, date_iso: str | None = None) -> Dict:
        """
        Return all events that overlap the given IST day (default: today).
        Output is sorted by start time (IST), and each item includes both UTC and IST strings.
        """
        day_start_utc, day_end_utc = self._day_bounds_utc(date_iso)
        items = []

        for cal in self._event_cals():
            try:
                # date_search returns events that intersect the window (and expands recurrences on many servers)
                evs = cal.date_search(day_start_utc, day_end_utc)
            except Exception:
                continue

            # Try to get a calendar display name (best-effort)
            try:
                props = cal.get_properties([dav.DisplayName()])
                cal_name = str(props.get("{DAV:}displayname", "") or props.get("{DAV:}displayname$", "")).strip()
            except Exception:
                cal_name = ""

            for ev in evs:
                v = getattr(ev, "vobject_instance", None)
                if not (v and hasattr(v, "vevent")):
                    continue
                ve = v.vevent

                # Summary/Location
                title = getattr(ve.summary, "value", "") if hasattr(ve, "summary") else ""
                location = getattr(ve.location, "value", "") if hasattr(ve, "location") else ""

                # Start/End handling (DATE or DATE-TIME)
                raw_start = getattr(ve, "dtstart", None)
                raw_end = getattr(ve, "dtend", None)

                if raw_start is None:
                    continue  # invalid
                start_utc = self._ical_dt_to_utc(raw_start.value)

                # If DTEND missing, derive from DTSTART + DURATION or treat as 1-hour default
                if raw_end is not None:
                    end_utc = self._ical_dt_to_utc(raw_end.value)
                else:
                    # RFC: If only DATE (all-day) start exists, treat as that single day
                    if hasattr(raw_start.value, "year") and not isinstance(raw_start.value, datetime):
                        # all-day single date -> exclusive end = start + 1 day
                        end_utc = (start_utc + timedelta(days=1))
                    else:
                        # fallback timed event default 1h
                        end_utc = start_utc + timedelta(hours=1)

                # Some servers give all-day with exclusive DTEND; normalize item span to overlap logic
                # Ensure start <= end
                if end_utc < start_utc:
                    # swap if malformed
                    start_utc, end_utc = end_utc, start_utc

                # Only include if overlaps our day window
                if not self._overlaps(start_utc, end_utc, day_start_utc, day_end_utc):
                    continue

                # Flags
                is_all_day = hasattr(raw_start.value, "year") and not isinstance(raw_start.value, datetime)

                # IST strings for UI
                start_ist = start_utc.astimezone(IST)
                end_ist = end_utc.astimezone(IST)

                items.append({
                    "title": title,
                    "location": location,
                    "href": str(ev.url),
                    "calendar": cal_name,
                    "all_day": bool(is_all_day),
                    "start_utc": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_utc": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "start_ist": start_ist.strftime("%Y-%m-%d %H:%M"),
                    "end_ist": end_ist.strftime("%Y-%m-%d %H:%M"),
                })

        # Sort by IST start
        items.sort(key=lambda x: (x["start_utc"], x["title"]))
        return {
            "date_ist": (day_start_utc.astimezone(IST)).strftime("%Y-%m-%d"),
            "tz": "Asia/Kolkata",
            "count": len(items),
            "events": items,
        }

    def list_events_today(self) -> Dict:
        """Convenience wrapper for today's IST date."""
        return self.list_events_for_day(None)

