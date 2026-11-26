#!/usr/bin/env python3
"""
Cerca Base — trova il link di una base di Clash of Clans partendo da uno screenshot.

Come funziona:
  1. Legge i canali "archivio" del server Discord (messaggi o post del forum con
     screenshot + link della base) e ne costruisce un indice.
  2. Quando qualcuno posta uno screenshot nel canale di ricerca, lo confronta con
     tutte le basi dell'archivio e risponde con le più simili e il loro link.

Il confronto guarda la DISPOSIZIONE degli edifici, quindi funziona anche se:
  • lo screenshot ha zoom, taglio, scenario o grafica di una live diversi;
  • la base è leggermente modificata (alcuni edifici spostati);
  • la base è ruotata di 90/180/270 gradi o specchiata.

Uso:
  python cerca_base.py                    aggiorna l'archivio e risponde alle richieste
  python cerca_base.py --image foto.png   prova locale sull'archivio già costruito

Richiede: Python 3.11+, opencv-python-headless, numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═════════════════════════════ Confronto immagini ═════════════════════════════
LONG = 1200
RATIOS = (0.70, 0.75, 0.80)        # rapporto altezza/larghezza di un riquadro isometrico


def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = LONG / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def feats(img, n=3000):
    g = prep(img)
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, None)
    if d is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), np.zeros(0, np.float32)
    return (np.float32([k.pt for k in kp]), np.clip(d, 0, 255).astype(np.uint8),
            np.float32([k.size for k in kp]))


def _matches(q, b, k=3):
    (qp, qd, qs), (bp, bd, bs) = q, b
    if len(qd) < 10 or len(bd) < 10:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd.astype(np.float32), bd.astype(np.float32), k=k)
    qi, bi, rk = [], [], []
    for row in m:
        for r, mm in enumerate(row):
            qi.append(mm.queryIdx); bi.append(mm.trainIdx); rk.append(r)
    qi = np.array(qi); bi = np.array(bi); rk = np.array(rk)
    return qp[qi], bp[bi], bi, qs[qi] / np.maximum(bs[bi], 1e-3), rk


def _scale(ratio):
    r = np.log(ratio[(ratio > 0.2) & (ratio < 5)])
    if len(r) < 30:
        return None
    hist, edges = np.histogram(r, bins=120, range=(np.log(0.2), np.log(5)))
    k = int(hist.argmax())
    sel = r[(r >= edges[max(0, k - 1)]) & (r <= edges[min(len(edges) - 1, k + 2)])]
    return float(np.exp(np.median(sel))) if len(sel) else None


def _count_cells(pts, cell=22.0):
    return len({(int(x // cell), int(y // cell)) for x, y in pts})


# ───────── Metodo A ─────────
def _score_affine(Q, B, rk, s):
    sel = rk < 2
    if sel.sum() < 8:
        return 0
    M, inl = cv2.estimateAffine2D(B[sel], Q[sel], method=cv2.RANSAC,
                                  ransacReprojThreshold=6.0 * max(s, 0.5), maxIters=4000, confidence=0.995)
    if M is None:
        return 0
    lin = M[:, :2]
    sv = np.linalg.svd(lin, compute_uv=False)
    if sv[1] < 1e-6 or sv[0] / sv[1] > 1.6:           # deve essere quasi una similitudine
        return 0
    _score_affine.mirrored = bool(np.linalg.det(lin) < 0)
    return _count_cells(B[sel][inl.ravel().astype(bool)])


# ───────── Metodo B ─────────
def _instances(Q, B, bidx, s, vtol=4.0, rad=40.0):
    V = Q - s * B
    cells = {}
    for i, v in enumerate(V):
        cells.setdefault((int(v[0] // vtol), int(v[1] // vtol)), []).append(i)
    out = []
    done = set()
    for key in list(cells):
        if key in done:
            continue
        done.add(key)
        group = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                group += cells.get((key[0] + dx, key[1] + dy), [])
        if len(group) < 3:
            continue
        group = np.array(sorted(set(group)))
        pts = B[group]
        lab = -np.ones(len(group), int); c = 0
        for a in range(len(group)):
            if lab[a] >= 0:
                continue
            stack = [a]; lab[a] = c
            while stack:
                u = stack.pop()
                near = np.where((np.abs(pts[:, 0] - pts[u, 0]) < rad) & (np.abs(pts[:, 1] - pts[u, 1]) < rad) & (lab < 0))[0]
                lab[near] = c; stack.extend(near.tolist())
            c += 1
        for cc in range(c):
            mem = group[lab == cc]
            if len(set(bidx[mem].tolist())) < 3:
                continue
            bm = B[mem]
            w = max(float(np.ptp(bm[:, 0])), 8.0)
            low = mem[np.argmax(bm[:, 1])]
            out.append((B[low], Q[low], w, bm.mean(0)))
    # elimina doppioni (stessa coppia trovata da celle vicine)
    uniq = {}
    for bpt, qpt, w, cen in out:
        key = (round(bpt[0] / 3), round(bpt[1] / 3), round(qpt[0] / 3), round(qpt[1] / 3))
        uniq.setdefault(key, (bpt, qpt, w, cen))
    return list(uniq.values())


def _sym_mats(r):
    P = np.array([[1.0, -1.0], [r, r]]); Pi = np.linalg.inv(P)
    mats = []
    for mirror in (False, True):
        for rot in range(4):
            if not mirror and rot == 0:
                continue                     # identità: metodo A
            S = np.eye(2)
            if mirror:
                S = np.array([[0.0, 1.0], [1.0, 0.0]]) @ S
            for _ in range(rot):
                S = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ S
            mats.append(P @ S @ Pi)
    return mats


def _score_symmetric(inst, s, max_hyp=400, rng=None):
    if len(inst) < 6:
        return 0
    cb = np.float32([i[0] for i in inst]); cq = np.float32([i[1] for i in inst])
    w = np.float32([i[2] for i in inst]); cen = np.float32([i[3] for i in inst])
    n = len(inst)
    hyp = np.arange(n) if n <= max_hyp else (rng or np.random.default_rng(0)).choice(n, max_hyp, replace=False)
    best = 0
    tol = 5.0
    for r in RATIOS:
        for L in _sym_mats(r):
            A = np.eye(2) - L
            U, S, Vt = np.linalg.svd(A)
            # ipotesi di traslazione: ogni edificio con offset tipico d0 = (0, 0.25 w)
            d0 = np.stack([np.zeros(n), 0.25 * w], 1)
            T = cq[hyp] - s * (cb[hyp] @ L.T) - s * (d0[hyp] @ A.T)          # (h,2)
            # e = (cq - t)/s - L cb    per tutte le coppie ipotesi×istanze
            E = (cq[None, :, :] - T[:, None, :]) / s - (cb @ L.T)[None, :, :]  # (h,n,2)
            Eu = E @ U                                                        # componenti nei vettori singolari
            if S[1] > 0.3:
                D = (Eu / S) @ Vt                                             # d = A^-1 e
                ok = ((np.abs(D[..., 0]) <= 0.4 * w + tol) &
                      (D[..., 1] >= -0.15 * w - tol) & (D[..., 1] <= 0.6 * w + tol))
            else:                                                             # riflessione: A di rango 1
                tight = np.abs(Eu[..., 1]) <= 2.0 * tol
                base_d = (Eu[..., :1] / S[0]) * Vt[0]                          # (h,n,2)
                nv = Vt[1]
                # esiste lambda con base_d + lambda*nv dentro la scatola?
                lo = np.full(E.shape[:2], -1e9); hi = np.full(E.shape[:2], 1e9)
                bounds = [(-0.4 * w - tol, 0.4 * w + tol), (-0.15 * w - tol, 0.6 * w + tol)]
                for ax in (0, 1):
                    a_lo, a_hi = bounds[ax]
                    if abs(nv[ax]) < 1e-6:
                        inside = (base_d[..., ax] >= a_lo) & (base_d[..., ax] <= a_hi)
                        lo = np.where(inside, lo, 1e9)
                    else:
                        l1 = (a_lo - base_d[..., ax]) / nv[ax]; l2 = (a_hi - base_d[..., ax]) / nv[ax]
                        lo = np.maximum(lo, np.minimum(l1, l2)); hi = np.minimum(hi, np.maximum(l1, l2))
                ok = tight & (lo <= hi)
            counts = ok.sum(1)
            h = int(counts.argmax())
            if counts[h] <= best:
                continue
            c = _count_cells(cen[ok[h]])
            best = max(best, c)
    return best


def score(q, b):
    """Punteggio di somiglianza (edifici allineati) e metodo: "A" stesso orientamento, "B" ruotata/specchiata."""
    m = _matches(q, b)
    if m is None:
        return 0, "A"
    Q, B, bidx, ratio, rk = m
    s = _scale(ratio)
    if s is None:
        return 0, "A"
    _score_affine.mirrored = False
    a = _score_affine(Q, B, rk, s)
    ka = "M" if _score_affine.mirrored else "A"
    if a >= 60:                      # già evidente: inutile provare le rotazioni
        return a, ka
    bsc = _score_symmetric(_instances(Q, B, bidx, s), s)
    return (a, ka) if a >= bsc else (bsc, "B")


# ═════════════════════════════ Discord ═════════════════════════════

DISCORD_API = "https://discord.com/api/v10"
UA = "DiscordBot (https://github.com, 1.0) CercaBase"
LINK_RE = re.compile(r"https?://link\.clashofclans\.com/[^\s<>()\[\]\"']*OpenLayout[^\s<>()\[\]\"']*", re.I)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")
TEXT_TYPES = (0, 5)          # testo, annunci
THREAD_PARENT_TYPES = (15, 16)   # forum, media


class DiscordError(Exception):
    def __init__(self, code, body):
        super().__init__(f"Discord HTTP {code}: {body[:200]}")
        self.code = code


def dapi(method, path, token, body=None, query=None, retries=6):
    url = DISCORD_API + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bot {token}", "User-Agent": UA, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            if e.code == 429:
                try:
                    wait = float(json.loads(txt).get("retry_after", 2))
                except Exception:
                    wait = 2.0
                time.sleep(min(wait, 60) + 0.3)
                continue
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise DiscordError(e.code, txt) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise DiscordError(429, "troppi tentativi")


def download(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 200:
        return None
    return img


def message_images(m):
    out = []
    for a in m.get("attachments", []):
        ct = (a.get("content_type") or "").lower()
        if ct.startswith("image/") or a.get("filename", "").lower().endswith(IMG_EXT):
            out.append((f"{m['id']}_{a['id']}", a["url"]))
    for k, e in enumerate(m.get("embeds", [])):
        for key in ("image", "thumbnail"):
            u = (e.get(key) or {}).get("url")
            if u and e.get("type") in ("image", "rich", "article") and u.lower().split("?")[0].endswith(IMG_EXT):
                out.append((f"{m['id']}_e{k}", u))
                break
    return out


def message_links(m):
    text = m.get("content", "") + " " + " ".join(
        (e.get("url") or "") + " " + (e.get("description") or "") for e in m.get("embeds", []))
    seen, out = set(), []
    for l in LINK_RE.findall(text):
        l = l.rstrip(".,;:!*_~|")
        if l not in seen:
            seen.add(l); out.append(l)
    return out


def th_of(link):
    m = re.search(r"id=([TB]H)(\d+)", urllib.parse.unquote(link), re.I)
    return f"{m.group(1).upper()}{m.group(2)}" if m else None


def snowflake_time(sid):
    return ((int(sid) >> 22) + 1420070400000) / 1000.0


# ═════════════════════════════ Archivio ═════════════════════════════

class Index:
    def __init__(self, root: Path):
        self.root = root
        self.fdir = root / "feats"
        self.fdir.mkdir(parents=True, exist_ok=True)
        p = root / "meta.json"
        self.meta = json.loads(p.read_text()) if p.exists() else {"version": 1, "entries": {}, "channels": {}}
        self.changed = False
        self._cache = None

    def save(self):
        tmp = self.root / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False))
        tmp.replace(self.root / "meta.json")
        if self.changed:
            (self.root / "CHANGED").write_text("1")

    def add(self, key, img, link, jump, ts):
        if key in self.meta["entries"]:
            return False
        pts, desc, sizes = feats(img)
        if len(pts) < 50:
            return False
        np.savez_compressed(self.fdir / f"{key}.npz", p=pts, d=desc, s=sizes.astype(np.float16))
        self.meta["entries"][key] = {"link": link, "th": th_of(link), "jump": jump, "ts": ts}
        self.changed = True
        self._cache = None
        return True

    def load_all(self):
        if self._cache is None:
            self._cache = {}
            for key in self.meta["entries"]:
                f = self.fdir / f"{key}.npz"
                if f.exists():
                    z = np.load(f)
                    self._cache[key] = (z["p"], z["d"], z["s"].astype(np.float32))
        return self._cache


def harvest(msgs, guild_id, channel_id, pending, single_link=None):
    """Da una lista di messaggi (in ordine) ricava le coppie immagine → link."""
    out = []
    for m in msgs:
        imgs, links = message_images(m), message_links(m)
        author = (m.get("author") or {}).get("id", "?")
        ts = snowflake_time(m["id"])
        jump = f"https://discord.com/channels/{guild_id}/{channel_id}/{m['id']}"
        if imgs and links:
            for k, (key, url) in enumerate(imgs):
                out.append((key, url, links[k] if len(links) == len(imgs) else links[0], jump))
        elif imgs:
            if single_link:
                out += [(key, url, single_link, jump) for key, url in imgs]
            else:
                pending[author] = {"ts": ts, "jump": jump, "imgs": imgs}
        elif links:
            p = pending.pop(author, None)
            if p and ts - p["ts"] <= 20 * 60:
                out += [(key, url, links[0], p["jump"]) for key, url in p["imgs"]]
    for a in [a for a, p in pending.items() if time.time() - p["ts"] > 3600]:
        del pending[a]
    return out


def fetch_after(token, channel_id, after=None, cap=20000):
    """Messaggi in ordine cronologico, dopo 'after' (o tutta la storia)."""
    msgs = []
    if after:
        cur = after
        while len(msgs) < cap:
            page = dapi("GET", f"/channels/{channel_id}/messages", token, query={"after": cur, "limit": 100})
            if not page:
                break
            page.sort(key=lambda m: int(m["id"]))
            msgs += page
            cur = page[-1]["id"]
            if len(page) < 100:
                break
    else:
        cur = None
        while len(msgs) < cap:
            q = {"limit": 100}
            if cur:
                q["before"] = cur
            page = dapi("GET", f"/channels/{channel_id}/messages", token, query=q)
            if not page:
                break
            msgs += page
            cur = min(page, key=lambda m: int(m["id"]))["id"]
            if len(page) < 100:
                break
        msgs.sort(key=lambda m: int(m["id"]))
    return msgs


def list_threads(token, guild_id, parent_id):
    threads = {}
    try:
        act = dapi("GET", f"/guilds/{guild_id}/threads/active", token) or {}
        for t in act.get("threads", []):
            if t.get("parent_id") == parent_id:
                threads[t["id"]] = t
    except DiscordError as e:
        log(f"⚠️  Thread attivi non leggibili: {e}")
    before = None
    for _ in range(200):
        q = {"limit": 100}
        if before:
            q["before"] = before
        page = dapi("GET", f"/channels/{parent_id}/threads/archived/public", token, query=q) or {}
        ts = page.get("threads", [])
        for t in ts:
            threads.setdefault(t["id"], t)
        if not page.get("has_more") or not ts:
            break
        before = ts[-1].get("thread_metadata", {}).get("archive_timestamp")
    return list(threads.values())


def update_index(idx: Index, token, archive_ids, deadline):
    added = 0
    for cid in archive_ids:
        cid = str(cid)
        try:
            ch = dapi("GET", f"/channels/{cid}", token)
        except DiscordError as e:
            log(f"⚠️  Canale archivio {cid} non leggibile ({e}). Il bot è nel server e vede il canale?")
            continue
        st = idx.meta["channels"].setdefault(cid, {"last_id": None, "threads": {}, "pending": {}})
        gid = ch.get("guild_id")
        name = ch.get("name", cid)
        containers = []
        if ch.get("type") in THREAD_PARENT_TYPES:
            for t in list_threads(token, gid, cid):
                last = t.get("last_message_id")
                done = st["threads"].get(t["id"])
                if done and last and int(last) <= int(done):
                    continue
                containers.append((t["id"], done, True))
        else:
            containers.append((cid, st["last_id"], False))
        for cont_id, after, is_thread in containers:
            if time.time() > deadline:
                log("⏱️  Tempo quasi finito: continuo l'archivio al prossimo giro")
                return added
            try:
                msgs = fetch_after(token, cont_id, after)
            except DiscordError as e:
                log(f"⚠️  Messaggi di {cont_id} non leggibili ({e})")
                continue
            if not msgs:
                continue
            single = None
            if is_thread:
                all_links = {l for m in msgs for l in message_links(m)}
                single = next(iter(all_links)) if len(all_links) == 1 else None
            pending = st["pending"] if not is_thread else {}
            pairs = harvest(msgs, gid, cont_id, pending, single)
            for key, url, link, jump in pairs:
                if key in idx.meta["entries"]:
                    continue
                try:
                    img = decode_image(download(url))
                except Exception as e:
                    log(f"⚠️  Immagine non scaricabile ({e})")
                    continue
                if img is not None and idx.add(key, img, link, jump, snowflake_time(key.split("_")[0])):
                    added += 1
            last = msgs[-1]["id"]
            if is_thread:
                st["threads"][cont_id] = last
            else:
                st["last_id"] = last
            idx.changed = True
            idx.save()
        log(f"📚 #{name}: archivio aggiornato")
    return added



# ═════════════════════════════ Fonte YouTube ═════════════════════════════
# Molti creator mettono il link della base nella descrizione del video (o in un
# commento del canale stesso) e mostrano la base nella miniatura: il bot li usa
# per riempire l'archivio da solo.

YT_API = "https://www.googleapis.com/youtube/v3"


def layout_id(link):
    q = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
    return (q.get("id") or [None])[0]


def yt_get(endpoint, params, key):
    q = dict(params); q["key"] = key
    try:
        return json.loads(download(f"{YT_API}/{endpoint}?{urllib.parse.urlencode(q)}", retries=2))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"YouTube API {endpoint}: HTTP {e.code}") from None


class Catalog:
    """Elenco persistente dei video YouTube già esaminati (non si rifà ogni settimana)."""
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        p = root / "catalog.json"
        self.data = json.loads(p.read_text()) if p.exists() else {"items": {}, "seen": {}, "last_run": None}
        self.changed = False
        (root / "CHANGED").unlink(missing_ok=True)

    def save(self):
        (self.root / "catalog.json.tmp").write_text(json.dumps(self.data, ensure_ascii=False))
        (self.root / "catalog.json.tmp").replace(self.root / "catalog.json")
        if self.changed:
            (self.root / "CHANGED").write_text("1")


def best_thumb(thumbs):
    for k in ("maxres", "standard", "high", "medium", "default"):
        if (thumbs or {}).get(k, {}).get("url"):
            return thumbs[k]["url"]
    return None


def youtube_collect(ycfg: dict, cat: Catalog, key: str):
    if not ycfg.get("enabled", True):
        return 0
    if not key:
        log("ℹ️  Raccolta da YouTube spenta: manca YOUTUBE_API_KEY_BASI (o YOUTUBE_API_KEY)")
        return 0
    now = time.time()
    last = cat.data.get("last_run")
    if last and now - last < float(ycfg.get("every_hours", 24)) * 3600 - 300:
        return 0
    first = last is None
    since = (now - float(ycfg.get("backfill_days", 365)) * 86400) if first else last - 2 * 86400
    since_iso = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pages = int(ycfg.get("backfill_pages" if first else "pages", 4 if first else 2))
    th_ok = {t.upper() for t in ycfg.get("town_halls", ["TH19"])}
    seen, items = cat.data["seen"], cat.data["items"]
    new_ids, quota = [], 0
    orders = ["date", "relevance"] if first else ["date"]
    for qtext in ycfg.get("queries", []):
        for order in orders:
            params = {"part": "id", "q": qtext, "type": "video", "order": order, "maxResults": 50}
            if order == "date":
                params["publishedAfter"] = since_iso
            for _ in range(max(1, pages)):
                try:
                    data = yt_get("search", params, key)
                except Exception as e:
                    log(f"⚠️  Ricerca YouTube '{qtext}' non riuscita: {e}")
                    break
                quota += 100
                for it in data.get("items", []):
                    vid = (it.get("id") or {}).get("videoId")
                    if vid and vid not in seen and vid not in new_ids:
                        new_ids.append(vid)
                if not data.get("nextPageToken"):
                    break
                params["pageToken"] = data["nextPageToken"]
    added = 0
    for i in range(0, len(new_ids), 50):
        try:
            data = yt_get("videos", {"part": "snippet", "id": ",".join(new_ids[i:i + 50])}, key)
        except Exception as e:
            log(f"⚠️  Dettagli video non disponibili: {e}")
            break
        quota += 1
        for v in data.get("items", []):
            sn = v.get("snippet") or {}
            text = sn.get("title", "") + "\n" + sn.get("description", "")
            links = LINK_RE.findall(text)
            if not links and ycfg.get("check_creator_comments", True):
                try:
                    cm = yt_get("commentThreads", {"part": "snippet", "videoId": v["id"], "maxResults": 20,
                                                   "order": "relevance", "textFormat": "plainText"}, key)
                    quota += 1
                    for c in cm.get("items", []):
                        top = ((c.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {}
                        if (top.get("authorChannelId") or {}).get("value") == sn.get("channelId"):
                            links += LINK_RE.findall(top.get("textDisplay", "") + " " + top.get("textOriginal", ""))
                except Exception:
                    pass                      # commenti disattivati
            uniq = {}
            for l in links:
                lid = layout_id(l.rstrip(".,;:!*_~|"))
                if lid:
                    uniq.setdefault(lid, l.rstrip(".,;:!*_~|"))
            seen[v["id"]] = "multi" if len(uniq) > 1 else ("none" if not uniq else "ok")
            if len(uniq) != 1:
                continue                      # 0 link, oppure video con tante basi: la miniatura non dice quale
            link = next(iter(uniq.values()))
            th = th_of(link)
            if th_ok and th not in th_ok:
                seen[v["id"]] = "th"
                continue
            thumb = best_thumb(sn.get("thumbnails"))
            if not thumb:
                continue
            items[v["id"]] = {"link": link, "th": th, "thumb": thumb, "title": sn.get("title", ""),
                              "channel": sn.get("channelTitle", ""), "ts": sn.get("publishedAt", "")}
            added += 1
    for vid in new_ids:
        seen.setdefault(vid, "skip")
    cat.data["last_run"] = now
    cat.changed = True
    log(f"▶️  YouTube: {len(new_ids)} video nuovi esaminati, {added} basi con link aggiunte "
        f"(quota usata ≈ {quota} unità)")
    return added


def index_youtube(idx: Index, cat: Catalog, deadline):
    added = 0
    for vid, it in cat.data["items"].items():
        key = f"yt_{vid}"
        if key in idx.meta["entries"] or key in idx.meta.setdefault("failed", {}):
            continue
        if time.time() > deadline:
            log("⏱️  Tempo quasi finito: continuo con le miniature al prossimo giro")
            break
        try:
            img = decode_image(download(it["thumb"]))
        except Exception:
            img = None
        if img is None or not idx.add(key, img, it["link"], f"https://www.youtube.com/watch?v={vid}", time.time()):
            idx.meta["failed"][key] = 1
            idx.changed = True
            continue
        added += 1
        if added % 25 == 0:
            idx.save()
    return added

# ═════════════════════════════ Ricerca ═════════════════════════════

def search(idx: Index, qf, th=None, top=3):
    feats_all = idx.load_all()
    best_by_link = {}
    for key, bf in feats_all.items():
        e = idx.meta["entries"][key]
        if th and e.get("th") and e["th"] != th:
            continue
        sc, how = score(qf, bf)
        cur = best_by_link.get(e["link"])
        if cur is None or sc > cur[0]:
            best_by_link[e["link"]] = (sc, how, e)
    ranked = sorted(best_by_link.values(), key=lambda x: -x[0])
    return ranked[:top], (ranked[top][0] if len(ranked) > top else 0), len(feats_all)


def confidence(results, cfg):
    if not results:
        return "none"
    s1 = results[0][0]
    s2 = results[1][0] if len(results) > 1 else 0
    if s1 < cfg.get("min_score", 12):
        return "none"
    if (results[0][1] in ("A", "M") and s1 >= 60) or s1 >= 2.0 * max(s2, 1):
        return "high"
    if s1 >= 1.25 * max(s2, 1) and s1 >= 15:
        return "mid"
    return "low"


def format_reply(per_image, n_archive, cfg):
    lines = []
    for k, (results, conf) in enumerate(per_image):
        if len(per_image) > 1:
            lines.append(f"**Image {k + 1}**")
        head = {"high": "🟢 **Match found** (very likely)",
                "mid": "🟡 **Likely match**",
                "low": "🔴 **Uncertain**: this base may not be in the archive, here are the closest ones",
                "none": "⚪ **No similar base** in the archive"}[conf]
        lines.append(head)
        if conf == "none" and not cfg.get("show_nearest_when_none", True):
            continue
        for i, (sc, how, e) in enumerate(results, 1):
            th = f"{e['th']} · " if e.get("th") else ""
            kind = {"A": "same orientation", "M": "mirrored"}.get(how, "rotated/mirrored")
            src = "▶️ video" if "youtube.com" in e["jump"] else "original post"
            lines.append(f"`{i}.` {th}score **{sc}** ({kind}) · [📋 copy in game]({e['link']}) · [{src}]({e['jump']})")
        lines.append("")
    lines.append(f"-# Archive: {n_archive} bases · tip: add \"th19\" to your message to search only TH19 bases")
    return "\n".join(lines)[:1990]


def answer_queries(idx: Index, token, search_ids, cfg, deadline):
    me = dapi("GET", "/users/@me", token)
    done_emoji = "✅"
    answered = 0
    for cid in search_ids:
        cid = str(cid)
        try:
            msgs = dapi("GET", f"/channels/{cid}/messages", token, query={"limit": 50}) or []
        except DiscordError as e:
            log(f"⚠️  Canale di ricerca {cid} non leggibile ({e})")
            continue
        for m in sorted(msgs, key=lambda m: int(m["id"])):
            if time.time() > deadline:
                return answered
            author = m.get("author") or {}
            if author.get("id") == me["id"] or author.get("bot"):
                continue
            if time.time() - snowflake_time(m["id"]) > 48 * 3600:
                continue
            if any(r.get("me") and (r.get("emoji") or {}).get("name") in (done_emoji, "⚠️") for r in m.get("reactions", [])):
                continue
            imgs = message_images(m)
            if not imgs:
                continue
            emoji_path = f"/channels/{cid}/messages/{m['id']}/reactions/{urllib.parse.quote('🔍')}/@me"
            try:
                dapi("PUT", emoji_path, token)
            except DiscordError:
                pass
            thm = re.search(r"\bth\s?(\d{1,2})\b", m.get("content", ""), re.I)
            th = f"TH{thm.group(1)}" if thm else None
            per_image = []
            n_arch = 0
            for key, url in imgs[:3]:
                try:
                    img = decode_image(download(url))
                except Exception:
                    img = None
                if img is None:
                    per_image.append(([], "none"))
                    continue
                qf = feats(img)
                results, _, n_arch = search(idx, qf, th, int(cfg.get("max_results", 3)))
                per_image.append((results, confidence(results, cfg)))
                log(f"🔎 richiesta {m['id']}: " + ", ".join(f"{r[0]}{r[1]}" for r in results))
            if not idx.meta["entries"]:
                content = "⚠️ The base archive is empty: check `archive_channels` in cerca_base.toml."
            else:
                content = format_reply(per_image, n_arch or len(idx.meta["entries"]), cfg)
            dapi("POST", f"/channels/{cid}/messages", token, body={
                "content": content,
                "message_reference": {"message_id": m["id"], "fail_if_not_exists": False},
                "allowed_mentions": {"parse": [], "replied_user": False},
                "flags": 4,          # niente anteprime dei link
            })
            try:
                dapi("PUT", f"/channels/{cid}/messages/{m['id']}/reactions/{urllib.parse.quote(done_emoji)}/@me", token)
                dapi("DELETE", emoji_path, token)
            except DiscordError:
                pass
            answered += 1
    return answered


def main():
    ap = argparse.ArgumentParser(description="Cerca il link di una base da uno screenshot")
    ap.add_argument("--config", default=str(BASE_DIR / "cerca_base.toml"))
    ap.add_argument("--index", default=str(BASE_DIR / "base_index"))
    ap.add_argument("--catalog", default=str(BASE_DIR / "yt_catalog"))
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--image", help="prova locale: confronta un'immagine con l'archivio e stampa i risultati")
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    with open(args.config, "rb") as f:
        full_cfg = tomllib.load(f)
    cfg = full_cfg.get("cerca_base", {})
    ycfg = full_cfg.get("youtube", {})
    idx = Index(Path(args.index))
    (idx.root / "CHANGED").unlink(missing_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        res, _, n = search(idx, feats(img), None, int(cfg.get("max_results", 3)))
        print(f"Archivio: {n} basi — affidabilità: {confidence(res, cfg)}")
        for sc, how, e in res:
            print(f"  {sc:4d} {how}  {e.get('th')}  {e['link']}")
        return 0

    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        log("❌ Manca il segreto DISCORD_BOT_TOKEN")
        return 2
    archive = [str(c).strip() for c in cfg.get("archive_channels", []) if str(c).strip().isdigit()]
    searchc = [str(c).strip() for c in cfg.get("search_channels", []) if str(c).strip().isdigit()]
    if not searchc:
        log("❌ Metti l'ID del canale di ricerca in search_channels (cerca_base.toml)")
        return 2
    idx_deadline = t0 + args.max_minutes * 60 * 0.8
    added = update_index(idx, token, archive, idx_deadline) if archive else 0
    cat = Catalog(Path(args.catalog))
    ykey = os.environ.get("YOUTUBE_API_KEY_BASI", "").strip() or os.environ.get("YOUTUBE_API_KEY", "").strip()
    youtube_collect(ycfg, cat, ykey)
    cat.save()
    added += index_youtube(idx, cat, idx_deadline)
    idx.save()
    n_yt = sum(1 for k in idx.meta["entries"] if k.startswith("yt_"))
    log(f"📚 Archivio: {len(idx.meta['entries'])} basi ({n_yt} da YouTube, +{added} nuove)")
    n = answer_queries(idx, token, searchc, cfg, deadline)
    idx.save()
    log(f"Fatto: {n} richieste risposte in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
