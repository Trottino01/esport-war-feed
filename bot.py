#!/usr/bin/env python3
"""
Esports War Feed
================
Inoltra in un canale Discord i video YouTube delle guerre di Clash of Clans
giocate in modalità Esports (World Championship, Monthly Final, Town Hall Cup,
scrim, ecc.), così puoi analizzarle senza perderne nessuna.

Fonti:
  1. Feed RSS dei canali e delle playlist elencati in config.toml
     (gratis, nessuna chiave necessaria).
  2. Facoltativa: ricerca su tutto YouTube con la YouTube Data API v3,
     attiva solo se imposti la variabile YOUTUBE_API_KEY.

Uso:
  python bot.py              un controllo e poi esce (GitHub Actions, cron)
  python bot.py --loop 15    resta acceso e controlla ogni 15 minuti
  python bot.py --dry-run    mostra cosa pubblicherebbe, senza inviare né salvare
  python bot.py --test       invia un messaggio di prova nel canale

Richiede Python 3.11 o successivo. Nessuna libreria da installare.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    sys.exit("Serve Python 3.11 o successivo (manca il modulo tomllib).")

BASE_DIR = Path(__file__).resolve().parent
FEED = "https://www.youtube.com/feeds/videos.xml"
API = "https://www.googleapis.com/youtube/v3"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
DISCORD_UA = "DiscordBot (https://github.com, 1.0) EsportsWarFeed"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}
SEEN_RETENTION_DAYS = 400   # dopo quanto dimenticare un video già visto
HEARTBEAT_DAYS = 7          # aggiorna lo stato almeno ogni tanto (tiene vivo il cron di GitHub)

VERBOSE = False


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def debug(msg: str) -> None:
    if VERBOSE:
        log(msg)


# ───────────────────────────── utilità ─────────────────────────────

def parse_dt(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_duration_seconds(s: str | None) -> int | None:
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", s or "")
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def env_value(name: str) -> str:
    """Legge una variabile d'ambiente, oppure il file .env accanto a bot.py."""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    p = BASE_DIR / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, val = line.split("=", 1)
            if k.strip() == name:
                return val.strip().strip('"').strip("'")
    return ""


def http_get(url: str, *, retries: int = 3, timeout: int = 20) -> bytes:
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": "CONSENT=YES+1; SOCS=CAI",   # evita la pagina dei cookie di YouTube
    }
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403):       # inutile riprovare
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def api_get(endpoint: str, params: dict, key: str) -> dict:
    q = dict(params)
    q["key"] = key
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(q)}"
    try:
        return json.loads(http_get(url, retries=2))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        reason = ""
        try:
            reason = json.loads(body)["error"]["errors"][0]["reason"]
        except Exception:
            pass
        hint = " (quota giornaliera esaurita: riprova domani o riduci le query)" \
            if reason in ("quotaExceeded", "dailyLimitExceeded") else ""
        raise RuntimeError(f"YouTube API '{endpoint}': HTTP {e.code} {reason}{hint}") from None


# ───────────────────────────── modello ─────────────────────────────

@dataclass
class Video:
    id: str
    title: str
    description: str
    channel: str
    channel_id: str
    published: datetime
    source: str
    trusted_clash: bool = False     # canale dedicato a CoC: non serve la parola "clash"
    forward_all: bool = False       # inoltra tutto da questa fonte
    live: str = ""                  # "live" / "upcoming" / "" (noto solo con la API)
    duration: int | None = None     # secondi (noto solo con la API)
    tags: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"

    def to_pending(self) -> dict:
        return {
            "id": self.id, "title": self.title, "channel": self.channel,
            "published": iso(self.published), "matched": self.matched,
            "live": self.live, "source": self.source,
        }


# ───────────────────────────── fonti ─────────────────────────────

def parse_feed(xml_bytes: bytes, source: str, trusted: bool = False,
               forward_all: bool = False) -> list[Video]:
    root = ET.fromstring(xml_bytes)
    feed_author = root.findtext("atom:author/atom:name", "", NS)
    out = []
    for e in root.findall("atom:entry", NS):
        vid = e.findtext("yt:videoId", "", NS)
        if not vid:
            continue
        out.append(Video(
            id=vid,
            title=(e.findtext("atom:title", "", NS) or "").strip(),
            description=e.findtext("media:group/media:description", "", NS) or "",
            channel=e.findtext("atom:author/atom:name", "", NS) or feed_author,
            channel_id=e.findtext("yt:channelId", "", NS) or "",
            published=parse_dt(e.findtext("atom:published", "", NS)),
            source=source,
            trusted_clash=trusted,
            forward_all=forward_all,
        ))
    return out


def resolve_handle(handle: str, state: dict, api_key: str) -> str | None:
    """Trasforma @nomecanale nell'ID del canale (UC...), con cache in state.json."""
    handle = handle.strip()
    if not handle.startswith("@"):
        handle = "@" + handle
    cache = state.setdefault("channel_ids", {})
    if handle.lower() in cache:
        return cache[handle.lower()]
    cid = None
    if api_key:
        try:
            data = api_get("channels", {"part": "id", "forHandle": handle}, api_key)
            if data.get("items"):
                cid = data["items"][0]["id"]
        except Exception as e:
            debug(f"API forHandle fallita per {handle}: {e}")
    if not cid:
        try:
            page = http_get(f"https://www.youtube.com/{urllib.parse.quote(handle)}").decode("utf-8", "replace")
            for pat in (r'"externalId":"(UC[\w-]{22})"',
                        r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"',
                        r'feeds/videos\.xml\?channel_id=(UC[\w-]{22})'):
                m = re.search(pat, page)
                if m:
                    cid = m.group(1)
                    break
        except Exception as e:
            debug(f"Pagina del canale {handle} non raggiungibile: {e}")
    if cid:
        cache[handle.lower()] = cid
    return cid


def collect_rss(cfg: dict, state: dict, api_key: str, skip_shorts: bool):
    videos: list[Video] = []
    shorts: set[str] = set()
    flags: dict[str, tuple[bool, bool]] = {}   # channel_id -> (clash_only, forward_all)

    for ch in cfg.get("channels", []):
        name = ch.get("name") or ch.get("handle") or ch.get("id") or "?"
        cid = ch.get("id") or (resolve_handle(ch["handle"], state, api_key) if ch.get("handle") else None)
        if not cid:
            log(f"⚠️  Canale '{name}': non trovo l'ID, lo salto. Controlla 'handle' o metti 'id' in config.toml")
            continue
        trusted = bool(ch.get("clash_only", True))
        fwd = bool(ch.get("forward_all", False))
        flags[cid] = (trusted, fwd)
        src = f"canale {name}"
        got: dict[str, Video] = {}
        try:
            for v in parse_feed(http_get(f"{FEED}?channel_id={cid}"), src, trusted, fwd):
                got[v.id] = v
        except Exception as e:
            log(f"⚠️  Feed di '{name}' non raggiungibile ({e}); riprovo al prossimo giro")
        if cid.startswith("UC"):
            # Video lunghi (senza Shorts): allarga la copertura oltre gli ultimi 15 caricamenti
            try:
                for v in parse_feed(http_get(f"{FEED}?playlist_id=UULF{cid[2:]}", retries=1), src, trusted, fwd):
                    got.setdefault(v.id, v)
            except Exception:
                pass
            if skip_shorts:
                try:
                    shorts |= {v.id for v in parse_feed(
                        http_get(f"{FEED}?playlist_id=UUSH{cid[2:]}", retries=1), src)}
                except Exception:
                    pass
        videos.extend(got.values())
        debug(f"📡 {name}: {len(got)} video nei feed")

    for pl in cfg.get("playlists", []):
        pid = pl.get("id", "").strip()
        if not pid:
            continue
        name = pl.get("name") or pid
        try:
            videos.extend(parse_feed(
                http_get(f"{FEED}?playlist_id={pid}"), f"playlist {name}",
                bool(pl.get("clash_only", True)), bool(pl.get("forward_all", True))))
        except Exception as e:
            log(f"⚠️  Playlist '{name}' non raggiungibile ({e})")

    return videos, shorts, flags


def search_api(cfg: dict, state: dict, api_key: str, now: datetime, max_age: timedelta) -> list[Video]:
    s = cfg.get("search", {})
    if not api_key or not s.get("enabled", True):
        return []
    every = timedelta(hours=float(s.get("every_hours", 3)))
    last = state.get("last_search")
    if last and now - parse_dt(last) < every - timedelta(minutes=5):
        return []
    since = max(parse_dt(last) - timedelta(hours=6), now - max_age) if last else now - max_age
    out: list[Video] = []
    ok = True
    event_names = {"completed": "dirette concluse", "live": "dirette in corso", "upcoming": "dirette programmate"}
    items_all = []
    for entry in s.get("queries", []):
        # Una query può essere una stringa oppure {q = "...", event = "completed"|"live", pages = 2}
        if isinstance(entry, dict):
            q, event, pages = str(entry.get("q", "")), entry.get("event", ""), int(entry.get("pages", 1))
        else:
            q, event, pages = str(entry), "", 1
        label = f"'{q}'" + (f" ({event_names.get(event, event)})" if event else "")
        params = {"part": "snippet", "q": q, "type": "video", "order": "date",
                  "maxResults": 50, "publishedAfter": iso(since)}
        if event:
            params["eventType"] = event
        if s.get("relevance_language"):
            params["relevanceLanguage"] = s["relevance_language"]
        n_before = len(items_all)
        for _ in range(max(1, pages)):
            try:
                data = api_get("search", params, api_key)
            except Exception as e:
                log(f"⚠️  Ricerca {label} non riuscita: {e}")
                ok = False
                break
            items_all += [(it, label) for it in data.get("items", [])]
            if not data.get("nextPageToken"):
                break
            params["pageToken"] = data["nextPageToken"]
        log(f"🔎 {label}: {len(items_all) - n_before} risultati")
    for it, label in items_all:
        vid = (it.get("id") or {}).get("videoId")
        sn = it.get("snippet") or {}
        if not vid:
            continue
        live = sn.get("liveBroadcastContent", "none")
        out.append(Video(
            id=vid,
            title=html.unescape(sn.get("title", "")),
            description=html.unescape(sn.get("description", "")),
            channel=html.unescape(sn.get("channelTitle", "")),
            channel_id=sn.get("channelId", ""),
            published=parse_dt(sn.get("publishedAt")),
            source=f"ricerca {label}",
            live="" if live == "none" else live,
        ))
    if ok:
        state["last_search"] = iso(now)
    log(f"🔎 Ricerca YouTube: {len(out)} risultati in totale (dal {since.strftime('%d/%m %H:%M')} UTC)")
    return out


def enrich_with_api(videos: list[Video], api_key: str) -> None:
    """Descrizione completa, tag, durata e stato live (1 unità di quota ogni 50 video)."""
    info = {}
    ids = [v.id for v in videos]
    for i in range(0, len(ids), 50):
        data = api_get("videos", {"part": "snippet,contentDetails",
                                  "id": ",".join(ids[i:i + 50]), "maxResults": 50}, api_key)
        for it in data.get("items", []):
            info[it["id"]] = it
    for v in videos:
        it = info.get(v.id)
        if not it:
            continue
        sn = it.get("snippet") or {}
        v.title = sn.get("title", v.title)
        v.description = sn.get("description", v.description)
        v.tags = sn.get("tags", []) or []
        live = sn.get("liveBroadcastContent", "none")
        v.live = "" if live == "none" else live
        v.duration = iso_duration_seconds((it.get("contentDetails") or {}).get("duration"))


def merge(videos: list[Video]) -> list[Video]:
    by_id: dict[str, Video] = {}
    for v in videos:
        o = by_id.get(v.id)
        if o is None:
            by_id[v.id] = v
            continue
        o.trusted_clash |= v.trusted_clash
        o.forward_all |= v.forward_all
        o.live = o.live or v.live
        if len(v.description) > len(o.description):
            o.description = v.description
    return list(by_id.values())


# ───────────────────────────── filtro ─────────────────────────────

def compile_keywords(words) -> list[tuple[str, re.Pattern]]:
    out = []
    for w in words or []:
        w = str(w).strip().lower()
        if w:
            # Parola intera per gli alfabeti con spazi (latino, cirillico); nessun confine per
            # cinese/giapponese/coreano, dove le parole sono attaccate (es. "クラクラTH19").
            b = "0-9a-z\u00c0-\u024f\u0400-\u04ff"
            pre = f"(?<![{b}])" if re.match(f"[{b}]", w[0]) else ""
            post = f"(?![{b}])" if re.match(f"[{b}]", w[-1]) else ""
            out.append((w, re.compile(pre + re.escape(w) + post, re.IGNORECASE)))
    return out


class Matcher:
    def __init__(self, fcfg: dict, match_description: bool):
        self.esports = compile_keywords(fcfg.get("esports_keywords"))
        self.clash = compile_keywords(fcfg.get("clash_keywords"))
        self.exclude = compile_keywords(fcfg.get("exclude_in_title"))
        self.match_description = match_description

    def check(self, v: Video) -> tuple[bool, list[str], str]:
        parts = [v.title, " ".join(v.tags)]
        if self.match_description:
            parts.append(v.description)
        text = "\n".join(parts)
        bad = [w for w, p in self.exclude if p.search(v.title)]
        if bad:
            return False, [], f"escluso da '{bad[0]}'"
        hits = [w for w, p in self.esports if p.search(text)]
        hits = [h for h in hits if not any(h != o and h in o for o in hits)]  # niente doppioni
        if v.forward_all:
            return True, hits, ""
        if not hits:
            return False, [], "nessuna parola chiave esports"
        if not v.trusted_clash and not any(p.search(text) for _, p in self.clash):
            return False, hits, "non sembra un video di Clash of Clans"
        return True, hits, ""


# ───────────────────────────── Discord ─────────────────────────────

def safe(s: str) -> str:
    s = re.sub(r"([*_~|`>\\])", r"\\\1", s)
    return s.replace("@", "@​")   # niente ping involontari dai titoli


def discord_send(webhook: str, payload: dict, retries: int = 4) -> None:
    url = webhook + ("&" if "?" in webhook else "?") + "wait=true"
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json", "User-Agent": DISCORD_UA})
        try:
            with urllib.request.urlopen(req, timeout=20):
                return
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 429:
                try:
                    wait = float(json.loads(body).get("retry_after", 2))
                except Exception:
                    wait = 2.0
                time.sleep(min(wait, 60) + 0.5)
                continue
            if e.code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            hint = ""
            if e.code == 404:
                hint = " → il webhook non esiste più: creane uno nuovo e aggiorna DISCORD_WEBHOOK_URL"
            elif "thread_name" in body or "forum" in body.lower():
                hint = " → il canale è un Forum: metti forum_mode = true in config.toml"
            raise RuntimeError(f"Discord HTTP {e.code}: {body[:300]}{hint}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Discord non risponde dopo vari tentativi")


def build_payload(item: dict, dcfg: dict) -> dict:
    ts = int(parse_dt(item["published"]).timestamp())
    live = {"live": "  🔴 IN DIRETTA", "upcoming": "  ⏰ diretta programmata"}.get(item.get("live", ""), "")
    lines = []
    mention = (dcfg.get("mention") or "").strip()
    if mention:
        lines.append(mention)
    lines.append(f"🎥 **{safe(item['title'])}**{live}")
    lines.append(f"📺 {safe(item['channel'])} · 🗓️ <t:{ts}:f>")
    if item.get("matched"):
        lines.append("🏷️ " + ", ".join(item["matched"][:5]))
    lines.append(f"https://www.youtube.com/watch?v={item['id']}")
    payload = {
        "content": "\n".join(lines)[:2000],
        "allowed_mentions": {"parse": ["roles", "users", "everyone"]} if mention else {"parse": []},
    }
    if dcfg.get("username"):
        payload["username"] = str(dcfg["username"])[:80]
    if dcfg.get("avatar_url"):
        payload["avatar_url"] = dcfg["avatar_url"]
    if dcfg.get("forum_mode"):
        payload["thread_name"] = (item["title"] or item["id"])[:100]
    return payload


# ───────────────────────────── stato ─────────────────────────────

def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("⚠️  state.json illeggibile: riparto da zero")
    return {"seen": {}, "pending": {}, "channel_ids": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ───────────────────────────── ciclo principale ─────────────────────────────

def run_once(cfg: dict, state_path: Path, dry_run: bool = False) -> int:
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    state = load_state(state_path)
    seen: dict = state.setdefault("seen", {})
    pending: dict = state.setdefault("pending", {})
    beh = cfg.get("behavior", {})
    dcfg = cfg.get("discord", {})
    max_age = timedelta(days=float(beh.get("max_age_days", 14)))
    max_posts = int(beh.get("max_posts_per_run", 25))
    skip_shorts = bool(beh.get("skip_shorts", True))
    min_dur = int(float(beh.get("min_duration_minutes", 4)) * 60)
    matcher = Matcher(cfg.get("filter", {}), bool(beh.get("match_description", True)))

    # Se filtro o ricerche sono cambiati, rivaluta i video scartati prima e rifà la ricerca da capo
    fhash = hashlib.sha1(json.dumps([cfg.get("filter", {}), cfg.get("search", {}),
                                     beh.get("max_age_days")], sort_keys=True).encode()).hexdigest()[:12]
    if state.get("filter_hash") != fhash:
        dropped = [k for k, val in seen.items() if isinstance(val, list) and val[0] in ("skip", "old")]
        for k in dropped:
            del seen[k]
        had_search = state.pop("last_search", None)
        if pending:   # la coda va rifatta con il nuovo filtro
            log(f"♻️  Svuoto la coda ({len(pending)} video): verranno ricontrollati con il nuovo filtro")
            pending.clear()
        if dropped or had_search:
            log(f"♻️  Configurazione cambiata: rifaccio la ricerca e ricontrollo {len(dropped)} video scartati prima")
        state["filter_hash"] = fhash

    api_key = env_value("YOUTUBE_API_KEY")
    webhook = env_value("DISCORD_WEBHOOK_URL")
    if not webhook and not dry_run:
        log("❌ Manca DISCORD_WEBHOOK_URL (segreto su GitHub oppure file .env). Mi fermo.")
        return 2

    # 1) raccolta
    found, shorts, flags = collect_rss(cfg, state, api_key, skip_shorts)
    try:
        found += search_api(cfg, state, api_key, now, max_age)
    except Exception as e:
        log(f"⚠️  Ricerca YouTube saltata: {e}")
    for v in found:
        if v.channel_id in flags:
            t, f = flags[v.channel_id]
            v.trusted_clash |= t
            v.forward_all |= f
    videos = merge(found)

    # 2) scarta già visti, troppo vecchi, Shorts
    def mark(v: Video, status: str) -> None:
        seen[v.id] = [status, today]

    fresh = []
    for v in videos:
        if v.id in seen or v.id in pending:
            continue
        if now - v.published > max_age:
            mark(v, "old")
        elif skip_shorts and v.id in shorts:
            mark(v, "short")
        else:
            fresh.append(v)

    if api_key and fresh:
        try:
            enrich_with_api(fresh, api_key)
        except Exception as e:
            log(f"⚠️  Dettagli video dalla API non disponibili: {e}")

    # 3) filtro esports
    accepted = 0
    for v in fresh:
        if (v.duration is not None and v.live not in ("live", "upcoming")
                and v.duration < min_dur):
            mark(v, "short")
            log(f"   scartato (sotto {min_dur // 60} min): {v.title}  [{v.channel}]")
            continue
        ok, hits, why = matcher.check(v)
        if ok:
            v.matched = hits
            pending[v.id] = v.to_pending()
            accepted += 1
            log(f"✅ {v.title}  [{v.channel}]  ← {', '.join(hits) or 'fonte sempre inoltrata'}")
        else:
            mark(v, "skip")
            log(f"   scartato ({why}): {v.title}  [{v.channel}]")

    # 4) invio (dal più vecchio al più recente)
    queue = sorted(pending.values(), key=lambda d: d["published"])
    posted, failed = 0, False
    for item in queue[:max_posts]:
        if dry_run:
            log(f"[prova] pubblicherei: {item['title']} — https://youtu.be/{item['id']}")
            posted += 1
            continue
        try:
            discord_send(webhook, build_payload(item, dcfg))
        except Exception as e:
            log(f"❌ Invio a Discord non riuscito: {e}")
            failed = True
            break
        pending.pop(item["id"], None)
        seen[item["id"]] = ["posted", today]
        posted += 1
        save_state(state_path, state)   # salva subito: niente doppioni se qualcosa si interrompe
        time.sleep(1.2)

    # 5) pulizia e salvataggio
    cutoff = (now - timedelta(days=SEEN_RETENTION_DAYS)).date().isoformat()
    for vid in [k for k, val in seen.items() if isinstance(val, list) and len(val) > 1 and val[1] < cutoff]:
        del seen[vid]
    hb = state.get("heartbeat")
    if not hb or hb < (now - timedelta(days=HEARTBEAT_DAYS)).date().isoformat():
        state["heartbeat"] = today
    if not dry_run:
        save_state(state_path, state)

    left = max(0, len(queue) - posted)
    log(f"Fatto: {len(videos)} video controllati, {accepted} nuovi esports, "
        f"{posted} {'da pubblicare' if dry_run else 'pubblicati'}"
        + (f", {left} in coda per il prossimo giro" if left else "")
        + ("" if api_key else " (ricerca globale spenta: nessuna YOUTUBE_API_KEY)"))
    return 1 if failed else 0


def send_test(cfg: dict) -> int:
    webhook = env_value("DISCORD_WEBHOOK_URL")
    if not webhook:
        log("❌ Manca DISCORD_WEBHOOK_URL")
        return 2
    dcfg = cfg.get("discord", {})
    payload = {"content": "✅ **Esports War Feed** è collegato a questo canale. "
                          "Qui arriveranno i video delle guerre in modalità Esports.",
               "allowed_mentions": {"parse": []}}
    if dcfg.get("username"):
        payload["username"] = dcfg["username"]
    if dcfg.get("avatar_url"):
        payload["avatar_url"] = dcfg["avatar_url"]
    if dcfg.get("forum_mode"):
        payload["thread_name"] = "Test Esports War Feed"
    discord_send(webhook, payload)
    log("Messaggio di prova inviato.")
    return 0


def load_config(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        sys.exit(f"Non trovo {path}")
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"Errore in {path.name}: {e}")


def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description="Inoltra su Discord i video YouTube delle guerre CoC in modalità Esports")
    ap.add_argument("--config", default=str(BASE_DIR / "config.toml"))
    ap.add_argument("--state", default=str(BASE_DIR / "state.json"))
    ap.add_argument("--loop", type=float, metavar="MINUTI", help="resta acceso e controlla ogni N minuti")
    ap.add_argument("--dry-run", action="store_true", help="mostra cosa farebbe, senza inviare né salvare")
    ap.add_argument("--test", action="store_true", help="invia un messaggio di prova al canale")
    ap.add_argument("-v", "--verbose", action="store_true", help="spiega anche perché un video è stato scartato")
    args = ap.parse_args()
    VERBOSE = args.verbose

    cfg_path, state_path = Path(args.config), Path(args.state)
    if args.test:
        return send_test(load_config(cfg_path))
    if not args.loop:
        return run_once(load_config(cfg_path), state_path, args.dry_run)

    log(f"Modalità continua: controllo ogni {args.loop:g} minuti (Ctrl+C per fermare)")
    while True:
        try:
            run_once(load_config(cfg_path), state_path, args.dry_run)   # rilegge la config a ogni giro
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"❌ Errore imprevisto: {e}")
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nFermato.")
