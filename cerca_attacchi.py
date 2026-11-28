#!/usr/bin/env python3
"""
Cerca Attacchi — trova nei video delle guerre (VOD esports) i momenti in cui
compare una base, partendo da uno screenshot.

Come funziona
  1. Raccolta (--index, avviata da GitHub Actions ogni ora):
     • prende i video pubblicati dal bot delle guerre (state.json) e quelli
       elencati in cerca_attacchi.toml (anche VOD di Twitch);
     • per ogni video guarda un fotogramma ogni N secondi (senza scaricare il
       video intero: yt-dlp trova l'indirizzo, ffmpeg salta al secondo giusto);
     • raggruppa i fotogrammi consecutivi che mostrano la stessa base in un
       "momento" (scouting o attacco) e salva l'immagine e le feature della base.
  2. Ricerca (/findattacks su Discord → Worker → GitHub Actions --interaction):
     confronta lo screenshot con tutte le basi viste nei video e risponde con i
     video e i minuti in cui compare quella base.

Usa lo stesso confronto di cerca_base.py (rotazioni, scenari, percentuale).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import cerca_base as cb

BASE_DIR = Path(__file__).resolve().parent
log = cb.log

SEG_FEATURES = 1500          # feature salvate per ogni base vista in un video (bastano: lo screenshot ne ha 3000)
SAME_BASE = 30               # punteggio oltre il quale due fotogrammi vicini mostrano la stessa base
MIN_AREA = 0.12              # l'area di gioco deve occupare almeno il 12% del fotogramma
FRAME_H = 720                # altezza dei fotogrammi analizzati
STEP_BACK = 10               # il link apre il video qualche secondo prima che la base compaia


# ═════════════════════════════ Archivio dei video ═════════════════════════════

class AttackIndex:
    """attack_index/meta.json  → video e momenti
       attack_index/feats/*.npz → feature delle basi viste
       attack_index/thumbs/*.jpg → immagine della base (per la risposta su Discord)"""

    def __init__(self, root: Path):
        self.root = root
        (root / "feats").mkdir(parents=True, exist_ok=True)
        (root / "thumbs").mkdir(parents=True, exist_ok=True)
        p = root / "meta.json"
        self.meta = json.loads(p.read_text()) if p.exists() else {"version": 1, "videos": {}, "segments": {}}
        self.changed = False
        (root / "CHANGED").unlink(missing_ok=True)

    def save(self):
        tmp = self.root / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False))
        tmp.replace(self.root / "meta.json")
        if self.changed:
            (self.root / "CHANGED").write_text("1")

    def add_segment(self, vkey, t0, t1, n_frames, f, thumb_jpg, area):
        skey = f"{vkey}__{int(t0)}"
        np.savez_compressed(self.root / "feats" / f"{skey}.npz", p=f[0], d=f[1], s=f[2].astype(np.float16))
        if thumb_jpg:
            (self.root / "thumbs" / f"{skey}.jpg").write_bytes(thumb_jpg)
        self.meta["segments"][skey] = {"video": vkey, "t0": int(t0), "t1": int(t1), "n": n_frames, "area": area}
        self.changed = True
        return skey

    def drop_video(self, vkey):
        for skey in [k for k, s in self.meta["segments"].items() if s["video"] == vkey]:
            (self.root / "feats" / f"{skey}.npz").unlink(missing_ok=True)
            (self.root / "thumbs" / f"{skey}.jpg").unlink(missing_ok=True)
            del self.meta["segments"][skey]
        self.meta["videos"].pop(vkey, None)
        self.changed = True

    def load_feats(self):
        out = {}
        for skey in self.meta["segments"]:
            f = self.root / "feats" / f"{skey}.npz"
            if f.exists():
                z = np.load(f)
                out[skey] = (z["p"], z["d"], z["s"].astype(np.float32))
        return out


# ═════════════════════════════ Video: indirizzi e fotogrammi ═════════════════════════════

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")
TW_ID_RE = re.compile(r"twitch\.tv/videos/(\d+)")


def video_key(s):
    """Chiave stabile di un video da un link o da un ID YouTube."""
    s = str(s).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return f"yt_{s}"
    m = YT_ID_RE.search(s)
    if m:
        return f"yt_{m.group(1)}"
    m = TW_ID_RE.search(s)
    if m:
        return f"tw_{m.group(1)}"
    if os.path.exists(s):                                 # prova locale con un file
        return "file_" + re.sub(r"[^A-Za-z0-9]", "_", Path(s).stem)[:40]
    return "url_" + re.sub(r"[^A-Za-z0-9]", "_", s)[-40:]


def video_url(vkey, v=None):
    if vkey.startswith("yt_"):
        return f"https://www.youtube.com/watch?v={vkey[3:]}"
    if vkey.startswith("tw_"):
        return f"https://www.twitch.tv/videos/{vkey[3:]}"
    return (v or {}).get("url", "")


def hms(t):
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def jump_url(vkey, v, t):
    t = max(0, int(t))
    if vkey.startswith("yt_"):
        return f"https://www.youtube.com/watch?v={vkey[3:]}&t={t}s"
    if vkey.startswith("tw_"):
        return f"https://www.twitch.tv/videos/{vkey[3:]}?t={t // 3600}h{t // 60 % 60}m{t % 60}s"
    return (v or {}).get("url", "")


class Blocked(Exception):
    """YouTube chiede di dimostrare di non essere un bot: inutile insistere in questo giro."""


def stream_info(url, cookies=None):
    """Indirizzo diretto del video (max 720p), durata, titolo. Con yt-dlp, senza scaricare nulla."""
    if os.path.exists(url):                               # file locale (prove)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "default=nw=1:nk=1", url], capture_output=True, text=True).stdout or 0)
        return {"src": url, "headers": {}, "duration": dur, "title": Path(url).stem, "channel": "",
                "live": False, "date": ""}
    import yt_dlp                                         # solo nella raccolta (nel workflow è installato)
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
            "format": "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/b[height<=720]/bv*/b"}
    if cookies:
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        msg = str(e)
        if ("confirm you" in msg and "bot" in msg) or "Sign in to confirm" in msg:
            raise Blocked(msg) from None
        raise
    live = info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live")
    src = info.get("url")
    if not src and info.get("requested_formats"):
        src = info["requested_formats"][0].get("url")
    return {"src": src, "headers": info.get("http_headers") or {}, "duration": float(info.get("duration") or 0),
            "title": info.get("title") or "", "channel": info.get("channel") or info.get("uploader") or "",
            "live": bool(live), "date": info.get("upload_date") or ""}


def grab(src, t, headers=None):
    """Un fotogramma al secondo t (ffmpeg salta lì: scarica solo un pezzetto del video)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-i", src, "-frames:v", "1", "-vf", f"scale=-2:'min({FRAME_H},ih)'",
            "-f", "image2pipe", "-vcodec", "bmp", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=90).stdout
    except subprocess.TimeoutExpired:
        return None
    if not out:
        return None
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


# ═════════════════════════════ Dai fotogrammi ai "momenti" ═════════════════════════════

def base_box(img):
    """Riquadro dell'area di gioco se il fotogramma mostra un villaggio (non telecronisti, punteggi...)."""
    box = cb.play_area_box(img)
    if not box:
        return None
    x0, y0, x1, y1 = box
    if (x1 - x0) * (y1 - y0) < MIN_AREA * img.shape[0] * img.shape[1]:
        return None
    return box


def same_base(a, b):
    if a is None or b is None or len(a[0]) < 30 or len(b[0]) < 30:
        return False
    s, _ = cb.score(a, b, fast=True)
    return s >= SAME_BASE


class Segmenter:
    """Riceve i fotogrammi in ordine e salva un "momento" per ogni tratto in cui si vede la stessa base."""

    def __init__(self, ai: AttackIndex, vkey, step):
        self.ai, self.vkey, self.step = ai, vkey, step
        self.cur = None
        self.saved = 0

    def feed(self, t, img):
        box = base_box(img) if img is not None else None
        if box is None:
            self.close()
            return
        f = cb.feats(img, n=SEG_FEATURES)
        if len(f[0]) < 80:
            self.close()
            return
        area = (box[2] - box[0]) * (box[3] - box[1])
        c = self.cur
        if c and t - c["t1"] <= self.step * 2.5 and same_base(c["last"], f):
            c["t1"], c["n"], c["last"] = t, c["n"] + 1, f
            # immagine migliore: la più "larga" (meno zoom) nei primi 45 s, prima che la base venga distrutta
            if t - c["t0"] <= 45 and area > c["area_px"] * 1.1:
                c.update(rep=f, img=img, box=box, area_px=area)
            return
        self.close()
        self.cur = {"t0": t, "t1": t, "n": 1, "last": f, "rep": f, "img": img, "box": box, "area_px": area}

    def close(self):
        c, self.cur = self.cur, None
        if not c:
            return
        thumb = None
        try:
            x0, y0, x1, y1 = c["box"]
            h, w = c["img"].shape[:2]
            mx, my = (x1 - x0) * 0.03, (y1 - y0) * 0.05
            crop = c["img"][int(max(0, y0 - my)):int(min(h, y1 + my)), int(max(0, x0 - mx)):int(min(w, x1 + mx))]
            if crop.shape[1] > 720:
                crop = cv2.resize(crop, (720, int(crop.shape[0] * 720 / crop.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumb = buf.tobytes() if ok else None
        except Exception:
            pass
        self.ai.add_segment(self.vkey, c["t0"], c["t1"], c["n"], c["rep"], thumb, c["box"])
        self.saved += 1


def index_video(ai: AttackIndex, vkey, v, info, step, deadline, workers=4):
    """Analizza un video dal punto in cui si era fermato. Ritorna True se l'ha finito."""
    dur = info["duration"]
    pos = float(v.get("pos", 0))
    seg = Segmenter(ai, vkey, step)
    times = list(np.arange(pos, max(dur - 1, 0), step))
    misses = 0
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(times), workers * 3):
            if time.time() > deadline:
                seg.close()
                v["pos"] = float(times[i])
                v["segments"] = v.get("segments", 0) + seg.saved
                ai.changed = True
                return False
            batch = times[i:i + workers * 3]
            frames = list(ex.map(lambda t: grab(info["src"], t, info["headers"]), batch))
            for t, img in zip(batch, frames):
                misses = misses + 1 if img is None else 0
                seg.feed(float(t), img)
            if misses >= 40:                              # l'indirizzo è scaduto o il video non si legge più
                seg.close()
                v["pos"] = float(batch[-1])
                v["segments"] = v.get("segments", 0) + seg.saved
                raise RuntimeError("troppi fotogrammi non letti")
    seg.close()
    v["pos"] = dur
    v["segments"] = v.get("segments", 0) + seg.saved
    ai.changed = True
    return True


# ═════════════════════════════ Raccolta dei video ═════════════════════════════

def war_feed_videos(state_path: Path):
    """Video pubblicati dal bot delle guerre (state.json, nello stesso repository)."""
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for vid, val in (st.get("seen") or {}).items():
        if isinstance(val, list) and val and val[0] == "posted":
            out.append((vid, val[1] if len(val) > 1 else ""))
    return [v for v, _ in sorted(out, key=lambda x: x[1], reverse=True)]


def collect(ai: AttackIndex, acfg, state_path: Path):
    """Aggiunge alla coda i video nuovi."""
    vids = ai.meta["videos"]
    added = 0
    sources = []
    if acfg.get("from_war_feed", True):
        sources += [(f"https://www.youtube.com/watch?v={v}", "war-feed") for v in war_feed_videos(state_path)]
    sources += [(u, "list") for u in acfg.get("extra_videos", [])]
    for url, src in sources:
        k = video_key(url)
        if k not in vids:
            vids[k] = {"url": url, "status": "todo", "from": src, "added": time.time(), "tries": 0}
            added += 1
            ai.changed = True
    return added


def run_index(acfg, ai: AttackIndex, state_path: Path, minutes):
    t_end = time.time() + minutes * 60
    n_new = collect(ai, acfg, state_path)
    step = float(acfg.get("step_seconds", 15))
    max_min = float(acfg.get("max_video_minutes", 480))
    cookies = None
    if os.environ.get("YT_COOKIES", "").strip():          # facoltativo: cookie di un account YouTube
        cookies = str(Path(tempfile.mkdtemp()) / "cookies.txt")
        Path(cookies).write_text(os.environ["YT_COOKIES"])
    vids = ai.meta["videos"]
    order = sorted([k for k, v in vids.items() if v["status"] in ("todo", "partial", "live", "retry")],
                   key=lambda k: (vids[k]["status"] != "partial", -vids[k].get("added", 0)))
    log(f"⚔️  Video delle guerre: {len(vids)} in archivio, {n_new} nuovi, {len(order)} da analizzare")
    done = 0
    for k in order:
        if time.time() > t_end - 60:
            break
        v = vids[k]
        try:
            info = stream_info(v["url"], cookies)
        except Blocked:
            log("⛔ YouTube chiede di confermare di non essere un bot: riprovo al prossimo giro. "
                "Se succede sempre, leggi la nota su YT_COOKIES in cerca_attacchi.toml.")
            break
        except Exception as e:
            v["tries"] = v.get("tries", 0) + 1
            v["status"] = "error" if v["tries"] >= 3 else "retry"
            v["error"] = str(e)[:200]
            ai.changed = True
            log(f"⚠️  {v['url']}: non leggibile ({str(e)[:120]})")
            continue
        v.update(title=info["title"] or v.get("title", ""), channel=info["channel"], duration=info["duration"],
                 date=info["date"])
        if info["live"]:
            v["status"] = "live"                          # diretta in corso: si analizza quando è finita
            ai.changed = True
            continue
        if info["duration"] <= 0 or info["duration"] > max_min * 60 or not info["src"]:
            v["status"] = "skipped"
            ai.changed = True
            continue
        t0 = time.time()
        try:
            finished = index_video(ai, k, v, info, step, t_end - 30)
        except Exception as e:
            v["tries"] = v.get("tries", 0) + 1
            v["status"] = "error" if v["tries"] >= 3 else "partial"
            v["error"] = str(e)[:200]
            ai.changed = True
            ai.save()
            log(f"⚠️  {info['title'][:60]}: interrotto ({e})")
            continue
        v["status"] = "done" if finished else "partial"
        ai.save()
        done += finished
        log(f"🎬 {info['title'][:70]} ({hms(info['duration'])}): {v.get('segments', 0)} basi viste"
            + ("" if finished else f", continuo da {hms(v['pos'])} al prossimo giro")
            + f" [{time.time() - t0:.0f}s]")
    # pulizia: i video più vecchi di keep_days escono dall'archivio
    keep = float(acfg.get("keep_days", 180)) * 86400
    for k in [k for k, v in vids.items() if time.time() - v.get("added", time.time()) > keep]:
        ai.drop_video(k)
    ai.save()
    n_seg = len(ai.meta["segments"])
    n_done = sum(1 for v in vids.values() if v["status"] == "done")
    log(f"📚 Archivio attacchi: {n_done} video analizzati, {n_seg} basi viste (+{done} video finiti in questo giro)")
    return 0


# ═════════════════════════════ Ricerca ═════════════════════════════

def find_attacks(ai: AttackIndex, qf, top=4):
    """Video in cui compare la base dello screenshot, con i minuti."""
    feats_all = ai.load_feats()
    keys = list(feats_all)
    if not keys:
        return [], 0
    scores = cb._score_all(qf, feats_all, keys)
    segs = ai.meta["segments"]
    by_video = {}
    for k in keys:
        s, how = scores[k]
        by_video.setdefault(segs[k]["video"], []).append((s, how, k))
    best = sorted(((max(x[0] for x in lst), v) for v, lst in by_video.items()), reverse=True)
    all_best = [s for s, _ in best]
    results = []
    for s, vkey in best[:top]:
        # percentuale: quanto spicca rispetto ai video che NON mostrano questa base
        bg = [x for x in all_best if x < 0.5 * s]
        pct = cb.match_percent(sorted([s] + bg, reverse=True), 0) if s > 0 else 0
        if pct < 25:
            continue
        lst = sorted(by_video[vkey], key=lambda x: -x[0])
        hits = sorted([x for x in lst if x[0] >= max(25, 0.5 * s)], key=lambda x: segs[x[2]]["t0"])
        times = []
        for sc_, how, k in hits:                         # momenti vicini (stesso attacco spezzato) = uno solo
            t = segs[k]["t0"]
            if not times or t - times[-1][0] > 90:
                times.append((t, k, sc_))
        v = ai.meta["videos"].get(vkey, {})
        results.append({"video": vkey, "title": v.get("title") or vkey, "channel": v.get("channel", ""),
                        "pct": pct, "score": s, "how": lst[0][1], "best_seg": lst[0][2],
                        "times": [(t, jump_url(vkey, v, t - STEP_BACK)) for t, _, _ in times[:6]]})
    return results, len(keys)


def build_reply(ai: AttackIndex, results, n_seg, n_videos):
    if not results:
        return ("⚪ **I couldn't find this base in the war videos yet.**\n"
                f"-# Archive: {n_videos} war videos, {n_seg} bases seen", [], [])
    p = results[0]["pct"]
    head = ("⚔️ **Found in " + (f"{len(results)} videos**" if len(results) > 1 else "1 video**")
            if p >= 50 else "🟠 **Possible matches**: compare the picture before trusting them")
    embeds, files = [], []
    for i, r in enumerate(results):
        when = " · ".join(f"[{hms(t)}]({u})" for t, u in r["times"])
        kind = {"A": "", "M": " · mirrored"}.get(r["how"], " · rotated/mirrored")
        emb = {"title": f"{r['pct']}% · {r['title']}"[:250], "url": r["times"][0][1] if r["times"] else "",
               "description": (f"▶️ On screen at: {when}{kind}" + (f"\n{r['channel']}" if r["channel"] else ""))[:4000],
               "color": cb.COLORS[cb.level(r["pct"])]}
        thumb = ai.root / "thumbs" / f"{r['best_seg']}.jpg"
        if thumb.exists():
            name = f"attack_{i + 1}.jpg"
            files.append((name, thumb.read_bytes()))
            emb["image"] = {"url": f"attachment://{name}"}
        embeds.append(emb)
    embeds[-1]["footer"] = {"text": f"Archive: {n_videos} war videos, {n_seg} bases seen"}
    return head, embeds, files



def run_interaction(ai: AttackIndex):
    app_id = os.environ.get("INTERACTION_APP_ID", "").strip()
    token = os.environ.get("INTERACTION_TOKEN", "").strip()
    try:
        urls = json.loads(os.environ.get("INTERACTION_IMAGES", "[]"))
    except json.JSONDecodeError:
        urls = []
    try:
        n_videos = sum(1 for v in ai.meta["videos"].values() if v.get("segments"))
        if not ai.meta["segments"]:
            cb.interaction_reply(app_id, token, "⏳ The war-video archive is still being built. "
                                                "Please try again later.", [])
            return 0
        img = None
        for u in urls[:1]:
            try:
                img = cb.decode_image(cb.download(u))
            except Exception:
                img = None
        if img is None:
            cb.interaction_reply(app_id, token, "⚠️ I couldn't open that picture.", [])
            return 0
        t = time.time()
        res, n_seg = find_attacks(ai, cb.feats(img))
        log(f"🔎 /findattacks: " + ", ".join(f"{r['pct']}% {r['video']}" for r in res) + f" in {time.time() - t:.0f}s")
        content, embeds, files = build_reply(ai, res, n_seg, n_videos)
        cb.interaction_reply(app_id, token, content, embeds, files)
    except Exception as e:
        log(f"❌ Errore: {e}")
        try:
            cb.interaction_reply(app_id, token, "⚠️ Something went wrong while searching. Please try again in a minute.", [])
        except Exception:
            pass
        return 1
    return 0


# ═════════════════════════════ Avvio ═════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Trova i momenti dei video delle guerre in cui compare una base")
    ap.add_argument("--config", default=str(BASE_DIR / "cerca_attacchi.toml"))
    ap.add_argument("--archive", default=str(BASE_DIR / "attack_index"))
    ap.add_argument("--state", default=str(BASE_DIR / "state.json"), help="stato del bot delle guerre")
    ap.add_argument("--index", action="store_true", help="raccolta e analisi dei video (GitHub Actions, ogni ora)")
    ap.add_argument("--max-minutes", type=float, default=40)
    ap.add_argument("--interaction", action="store_true", help="risponde a /findattacks (avviato dal Worker)")
    ap.add_argument("--video", help="prova: analizza un solo video (link o file)")
    ap.add_argument("--image", help="prova: cerca uno screenshot nell'archivio e stampa i risultati")
    args = ap.parse_args()
    acfg = {}
    if Path(args.config).exists():
        with open(args.config, "rb") as f:
            acfg = tomllib.load(f).get("attacchi", {})
    ai = AttackIndex(Path(args.archive))
    if args.interaction:
        return run_interaction(ai)
    if args.video:
        k = video_key(args.video)
        v = ai.meta["videos"].setdefault(k, {"url": args.video, "status": "todo", "added": time.time(), "tries": 0})
        info = stream_info(args.video)
        v.update(title=info["title"], duration=info["duration"])
        index_video(ai, k, v, info, float(acfg.get("step_seconds", 15)), time.time() + args.max_minutes * 60)
        v["status"] = "done"
        ai.save()
        log(f"🎬 {info['title']}: {v.get('segments', 0)} basi viste")
        return 0
    if args.image:
        res, n = find_attacks(ai, cb.feats(cv2.imread(args.image)))
        print(f"Basi viste nei video: {n}")
        for r in res:
            print(f"  {r['pct']:3d}%  {r['title']}  " + ", ".join(hms(t) for t, _ in r["times"]))
        return 0
    if args.index:
        if not acfg.get("enabled", True):
            log("ℹ️  Cerca Attacchi spento in cerca_attacchi.toml")
            return 0
        if not shutil.which("ffmpeg"):
            log("❌ Manca ffmpeg")
            return 2
        return run_index(acfg, ai, Path(args.state), args.max_minutes)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
