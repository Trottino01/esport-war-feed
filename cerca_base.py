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
import html
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
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


FEATURE_VERSION = 2          # cambia quando cambia il modo di estrarre le feature: l'archivio si ricostruisce


def play_area(img):
    """Contorno dell'area di gioco (il rombo d'erba della base), oppure None se non si trova.
    Serve a ignorare scenario, webcam, punteggi, barra delle truppe e scritte."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = ((h >= 18) & (h <= 50) & (sat >= 50) & (val >= 70)).astype(np.uint8)
    k = max(3, min(img.shape[:2]) // 80)
    m = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    c = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(cv2.findNonZero((lab == c).astype(np.uint8)))
    frac = cv2.contourArea(hull) / float(img.shape[0] * img.shape[1])
    if frac < 0.06 or frac > 0.95:                    # niente rombo riconoscibile: non filtrare
        return None
    return hull


def play_area_box(img):
    hull = play_area(img)
    if hull is None:
        return None
    x, y, w, h = cv2.boundingRect(hull)
    return [int(x), int(y), int(x + w), int(y + h)]


def feats(img, n=3000):
    if img.ndim == 3:
        sc = LONG / max(img.shape[:2])
        small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = None
        hull = play_area(small)
        if hull is not None:
            # solo DENTRO l'area di gioco, senza la fascia del bordo: il contorno del rombo e lo
            # scenario attorno sono uguali in tutte le basi e farebbero sembrare simili basi diverse
            mask = np.zeros(g.shape, np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            r = max(8, LONG // 50)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    else:
        g, mask = prep(img), None
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, mask)
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
    _score_affine.pts = B[sel][inl.ravel().astype(bool)]
    return _count_cells(_score_affine.pts)


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


def _score_symmetric(inst, s, max_hyp=400, rng=None, ratios=None):
    if len(inst) < 6:
        return 0
    cb = np.float32([i[0] for i in inst]); cq = np.float32([i[1] for i in inst])
    w = np.float32([i[2] for i in inst]); cen = np.float32([i[3] for i in inst])
    n = len(inst)
    hyp = np.arange(n) if n <= max_hyp else (rng or np.random.default_rng(0)).choice(n, max_hyp, replace=False)
    best = 0
    tol = 5.0
    for r in (ratios or RATIOS):
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
            if c > best:
                best = c
                _score_symmetric.pts = cen[ok[h]]
    return best


def score(q, b, fast=False):
    """Punteggio di somiglianza (edifici allineati) e metodo: "A" stesso orientamento, "B" ruotata/specchiata."""
    m = _matches(q, b)
    if m is None:
        return 0, "A"
    Q, B, bidx, ratio, rk = m
    s = _scale(ratio)
    if s is None:
        return 0, "A"
    if s > 1.8:                      # base molto piccola nell'immagine d'archivio: lavora alla scala dello screenshot
        B, s = B * s, 1.0
    _score_affine.mirrored = False
    a = _score_affine(Q, B, rk, s)
    ka = "M" if _score_affine.mirrored else "A"
    if a >= 60:                      # già evidente: inutile provare le rotazioni
        return a, ka
    bsc = _score_symmetric(_instances(Q, B, bidx, s), s, max_hyp=150 if fast else 400,
                           ratios=(0.75,) if fast else None)
    return (a, ka) if a >= bsc else (bsc, "B")


def match_region(q, b):
    """Zona (x0, y0, x1, y1) dell'immagine d'archivio che corrisponde allo screenshot,
    in coordinate normalizzate (lato lungo = LONG). None se non si riesce a stimarla."""
    m = _matches(q, b)
    if m is None:
        return None
    Q, B, bidx, ratio, rk = m
    s = _scale(ratio)
    if s is None:
        return None
    k = 1.0
    if s > 1.8:
        k, B, s = s, B * s, 1.0
    _score_affine.pts = None
    _score_symmetric.pts = None
    a = _score_affine(Q, B, rk, s)
    pts = _score_affine.pts
    if a < 60:
        bsc = _score_symmetric(_instances(Q, B, bidx, s), s)
        if bsc > a and _score_symmetric.pts is not None:
            pts = _score_symmetric.pts
    if pts is None or len(pts) < 4:
        return None
    pts = np.asarray(pts) / k
    x0, y0 = np.percentile(pts, 2, axis=0)
    x1, y1 = np.percentile(pts, 98, axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _run(frac, center, rel=0.2):
    """Tratto continuo di valori alti (>= rel * massimo) che contiene 'center'."""
    if frac.max() <= 0:
        return None
    on = frac >= rel * frac.max()
    c = int(min(max(center, 0), len(frac) - 1))
    if not on[c]:
        idxs = np.where(on)[0]
        c = int(idxs[np.argmin(np.abs(idxs - c))])
    a = c
    while a > 0 and on[a - 1]:
        a -= 1
    b = c
    while b < len(on) - 1 and on[b + 1]:
        b += 1
    return a, b + 1


def _sharp_box(img, near):
    """Riquadro della zona nitida (la base) attorno a 'near': esclude bande nere e copie
    sfocate ai lati, tipiche delle miniature degli Shorts."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    cell = max(8, min(img.shape[:2]) // 60)
    E = cv2.resize(cv2.blur(lap * lap, (cell, cell)), (img.shape[1] // cell, img.shape[0] // cell),
                   interpolation=cv2.INTER_AREA)
    sharp = (E > max(10.0, 0.04 * float(np.percentile(E, 95)))).astype(np.float32)
    cols = _run(sharp.mean(0), near[0] / cell)
    if not cols:
        return None
    rows = _run(sharp[:, cols[0]:cols[1]].mean(1), near[1] / cell)
    if not rows:
        return None
    # quanto è "piatto" (nero o sfocato) tutto ciò che sta fuori dal riquadro?
    flat = E < max(4.0, 0.004 * float(np.percentile(E, 95)))
    out = np.ones_like(flat, bool)
    out[rows[0]:rows[1], cols[0]:cols[1]] = False
    flat_out = float(flat[out].mean()) if out.any() else 0.0
    return cols[0] * cell, rows[0] * cell, cols[1] * cell, rows[1] * cell, flat_out


def base_crop(img, region, out_w=1000, area=None):
    """Ritaglia la base dall'immagine d'archivio e la rende più grande e nitida."""
    h, w = img.shape[:2]
    if area:                                          # rombo dell'area di gioco già noto: ritaglio esatto
        x0, y0, x1, y1 = area
        mx, my = (x1 - x0) * 0.02, (y1 - y0) * 0.04   # poco margine (gli edifici in alto sporgono un po')
        X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        X1, Y1 = int(min(w, x1 + mx)), int(min(h, y1 + my * 0.5))
        if X1 - X0 > 40 and Y1 - Y0 > 40:
            return _finish_crop(img[Y0:Y1, X0:X1], out_w)
    k = LONG / max(h, w)                              # le coordinate delle feature sono riscalate
    if region:
        x0, y0, x1, y1 = (v / k for v in region)
    else:
        x0, y0, x1, y1 = w * 0.25, h * 0.25, w * 0.75, h * 0.75
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sb = _sharp_box(img, (cx, cy))
    if sb and sb[4] >= 0.6:
        ex0, ey0, ex1, ey1 = sb[:4]                   # miniatura "a pannello" (bande nere / lati sfocati)
        mx, my = (ex1 - ex0) * 0.06, (ey1 - ey0) * 0.06  # piccolo margine: non tagliare le punte del rombo
        ex0, ey0, ex1, ey1 = ex0 - mx, ey0 - my, ex1 + mx, ey1 + my
    else:
        ex0, ey0, ex1, ey1 = 0, 0, w, h               # base a tutto schermo: meglio intera che tagliata
    X0, Y0 = int(max(0, ex0)), int(max(0, ey0))
    X1, Y1 = int(min(w, ex1)), int(min(h, ey1))
    crop = img[Y0:Y1, X0:X1] if (X1 - X0) > 20 and (Y1 - Y0) > 20 else img
    return _finish_crop(crop, out_w)


def _finish_crop(crop, out_w):
    ch, cw = crop.shape[:2]
    if cw < out_w:
        crop = cv2.resize(crop, (out_w, int(ch * out_w / cw)), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(crop, (0, 0), 1.2)
        crop = cv2.addWeighted(crop, 1.5, blur, -0.5, 0)   # leggera nitidezza
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


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
        self.outdated = bool(self.meta["entries"]) and self.meta.get("fv") != FEATURE_VERSION
        if not self.meta["entries"]:
            self.meta["fv"] = FEATURE_VERSION

    def reset(self):
        """Svuota l'archivio (le basi verranno rianalizzate con il metodo nuovo)."""
        for f in self.fdir.glob("*.npz"):
            f.unlink()
        self.meta = {"version": 1, "entries": {}, "channels": {}, "fv": FEATURE_VERSION}
        self.outdated = False
        self.changed = True
        self._cache = None

    def save(self):
        tmp = self.root / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False))
        tmp.replace(self.root / "meta.json")
        if self.changed:
            (self.root / "CHANGED").write_text("1")

    def add(self, key, img, link, jump, ts, img_url=None):
        if key in self.meta["entries"]:
            return False
        pts, desc, sizes = feats(img)
        if len(pts) < 50:
            return False
        np.savez_compressed(self.fdir / f"{key}.npz", p=pts, d=desc, s=sizes.astype(np.float16))
        self.meta["entries"][key] = {"link": link, "th": th_of(link), "jump": jump, "ts": ts}
        if img_url:
            self.meta["entries"][key]["img"] = img_url
        area = play_area_box(img)
        if area:
            self.meta["entries"][key]["area"] = area
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
    """Elenco persistente dei video YouTube esaminati e dei creator di basi (non si rifà ogni settimana)."""
    VERSION = 2

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        p = root / "catalog.json"
        self.data = json.loads(p.read_text()) if p.exists() else {"items": {}, "seen": {}, "last_run": None}
        self.changed = False
        (root / "CHANGED").unlink(missing_ok=True)
        self._migrate()

    def _migrate(self):
        d = self.data
        d.setdefault("creators", {})
        d.setdefault("recheck", [])
        d.setdefault("queries_done", [])
        d.setdefault("tg", {})
        d.setdefault("tg_items", {})
        if d.get("v", 1) < 2:
            if d.get("last_run"):                           # ricerche della prima versione: già fatte
                d["queries_done"] = ["th19 base link", "th19 war base link", "town hall 19 base link",
                                     "th19 anti 3 star base", "th19 base"]
            for it in d["items"].values():                  # vecchio formato: un solo link per video
                if "link" in it and "links" not in it:
                    it["links"] = [[it.pop("link"), None]]
            # i video con più basi prima venivano saltati: ora vanno riesaminati
            d["recheck"] = sorted({v for v, st in d["seen"].items() if st == "multi"} | set(d["recheck"]))
            d["v"] = 2
            self.changed = True

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


TS_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def _secs(m):
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s)


def links_with_times(text):
    """Link di basi nell'ordine in cui compaiono, con il minuto del video se scritto
    sulla stessa riga o nelle 2 righe precedenti (es. "1:20 Base 2 → link")."""
    out, seen = [], set()
    last_ts, last_line = None, -10
    for n, line in enumerate(text.splitlines()):
        tss = list(TS_RE.finditer(line))
        if tss:
            last_ts, last_line = _secs(tss[0]), n
        for l in LINK_RE.findall(line):
            l = l.rstrip(".,;:!*_~|")
            lid = layout_id(l)
            if not lid or lid in seen:
                continue
            seen.add(lid)
            out.append([l, last_ts if n - last_line <= 2 else None])
    return out


def _video_record(vid, sn, extra_links=()):
    text = sn.get("title", "") + "\n" + sn.get("description", "")
    links = links_with_times(text)
    have = {layout_id(l) for l, _ in links}
    for l in extra_links:                                   # link trovati nei commenti del creator
        l = l.rstrip(".,;:!*_~|")
        if layout_id(l) and layout_id(l) not in have:
            have.add(layout_id(l)); links.append([l, None])
    thumb = best_thumb(sn.get("thumbnails"))
    return {"links": links, "thumb": thumb, "title": sn.get("title", ""), "channel": sn.get("channelTitle", ""),
            "channel_id": sn.get("channelId") or sn.get("videoOwnerChannelId", ""), "ts": sn.get("publishedAt", "")}


TH19_TITLE_RE = re.compile(r"(?:\bth|\bcv|\bhdv|тх|town\s*hall|ayuntamiento|rathaus|h[oô]tel\s*de\s*ville)\s*-?\s*19(?!\d)"
                           r"|19\s*(?:本|홀|級|级)", re.I)


def _creator_comment_links(vid, channel_id, key):
    """Link di basi scritti dal creator nei commenti (spesso nel commento fissato). Costa 1 unità."""
    out = []
    cm = yt_get("commentThreads", {"part": "snippet", "videoId": vid, "maxResults": 20,
                                   "order": "relevance", "textFormat": "plainText"}, key)
    for c in cm.get("items", []):
        top = ((c.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {}
        if (top.get("authorChannelId") or {}).get("value") == channel_id:
            out += LINK_RE.findall(top.get("textDisplay", "") + " " + top.get("textOriginal", ""))
    return out


def _pacific_day():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


def _accept(cat, vid, rec, th_ok):
    """Salva il video se ha almeno un link del municipio giusto."""
    seen, items = cat.data["seen"], cat.data["items"]
    if not rec["links"]:
        seen[vid] = "none"
        return False
    if th_ok and not any(th_of(l) in th_ok for l, _ in rec["links"]):
        seen[vid] = "th"
        return False
    if not rec["thumb"]:
        seen[vid] = "skip"
        return False
    seen[vid] = "ok" if len(rec["links"]) == 1 else "multi_ok"
    items[vid] = rec
    return True


def youtube_collect(ycfg: dict, cat: Catalog, key: str):
    if not ycfg.get("enabled", True):
        return 0
    if not key:
        log("ℹ️  Raccolta da YouTube spenta: manca YOUTUBE_API_KEY_BASI (o YOUTUBE_API_KEY)")
        return 0
    now = time.time()
    last = cat.data.get("last_run")                     # ultimo giro di ricerche
    last_c = cat.data.get("last_creators")              # ultimo controllo dei creator
    pending = (set(ycfg.get("queries", [])) - set(cat.data.get("queries_done", []))) or cat.data.get("recheck")
    do_search = (not last) or now - last >= float(ycfg.get("every_hours", 24)) * 3600 - 300 or bool(pending)
    do_creators = (do_search or not last_c
                   or now - last_c >= float(ycfg.get("creators_every_hours", 3)) * 3600 - 300)
    if not (do_search or do_creators):
        return 0
    # tetto giornaliero (la quota di YouTube si azzera a mezzanotte, ora del Pacifico)
    day = _pacific_day()
    qd = cat.data.setdefault("quota_day", {"day": day, "used": 0})
    if qd.get("day") != day:
        qd.update(day=day, used=0)
    back_iso = datetime.fromtimestamp(now - float(ycfg.get("backfill_days", 365)) * 86400,
                                      timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_iso = datetime.fromtimestamp((last or now) - 2 * 86400, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    done_q = set(cat.data.get("queries_done", []))
    th_ok = {t.upper() for t in ycfg.get("town_halls", ["TH19"])}
    seen, items, creators = cat.data["seen"], cat.data["items"], cat.data["creators"]
    quota, added = 0, 0
    budget = min(int(ycfg.get("max_quota_per_run", 8000)), int(ycfg.get("daily_quota", 9500)) - int(qd.get("used", 0)))
    if budget < 20:
        log(f"ℹ️  YouTube: quota di oggi quasi finita ({qd.get('used')} unità usate), riprendo domani")
        return 0
    stop = False

    # 1) ricerche (tutte le lingue)
    new_ids = []
    for qtext in (ycfg.get("queries", []) if do_search else []):
        fresh_q = qtext not in done_q                       # ricerca nuova: guarda indietro di un anno
        orders = ["date", "relevance"] if fresh_q else ["date"]
        pages = int(ycfg.get("backfill_pages" if fresh_q else "pages", 1))
        ok_q = True
        for order in orders:
            params = {"part": "id", "q": qtext, "type": "video", "order": order, "maxResults": 50}
            if order == "date":
                params["publishedAfter"] = back_iso if fresh_q else recent_iso
            for _ in range(max(1, pages)):
                if quota + 100 > budget:
                    stop = True
                    break
                try:
                    data = yt_get("search", params, key)
                except Exception as e:
                    log(f"⚠️  Ricerca YouTube '{qtext}' non riuscita: {e}")
                    ok_q = False
                    break
                quota += 100
                for it in data.get("items", []):
                    vid = (it.get("id") or {}).get("videoId")
                    if vid and vid not in seen and vid not in new_ids:
                        new_ids.append(vid)
                if not data.get("nextPageToken"):
                    break
                params["pageToken"] = data["nextPageToken"]
            if stop:
                ok_q = False
                break
        if ok_q:
            done_q.add(qtext)
    cat.data["queries_done"] = sorted(done_q)
    if do_search:
        new_ids += [v for v in cat.data.get("recheck", []) if v not in new_ids]
        cat.data["recheck"] = []

    # 2) dettagli dei video trovati (1 unità ogni 50 video)
    for i in range(0, len(new_ids), 50):
        try:
            data = yt_get("videos", {"part": "snippet", "id": ",".join(new_ids[i:i + 50])}, key)
        except Exception as e:
            log(f"⚠️  Dettagli video non disponibili: {e}")
            break
        quota += 1
        for v in data.get("items", []):
            sn = v.get("snippet") or {}
            extra = []
            if not LINK_RE.search(sn.get("title", "") + sn.get("description", "")) and ycfg.get("check_creator_comments", True):
                try:
                    quota += 1
                    extra = _creator_comment_links(v["id"], sn.get("channelId"), key)
                except Exception:
                    pass                      # commenti disattivati
            rec = _video_record(v["id"], sn, extra)
            if rec["links"] and rec["channel_id"]:          # conta i video con link per canale
                c = creators.setdefault(rec["channel_id"], {"name": rec["channel"], "hits": 0})
                c["hits"] = c.get("hits", 0) + 1
            if _accept(cat, v["id"], rec, th_ok):
                added += 1
    for vid in new_ids:
        seen.setdefault(vid, "skip")

    # 3) creator di basi: leggi tutto il loro archivio di video (1 unità ogni 50 video)
    min_hits = int(ycfg.get("creator_min_videos", 2))
    crawl_pages = int(ycfg.get("creator_pages_per_run", 120))
    check_cm = ycfg.get("check_creator_comments", True)
    used_pages = 0

    def take(vid, sn, cid, c):
        """Un video dall'archivio di un canale: salvalo se ha link TH19 (anche nei commenti del creator)."""
        nonlocal quota, added
        sn = dict(sn); sn.setdefault("channelId", cid)
        extra = []
        text = sn.get("title", "") + " " + sn.get("description", "")
        if check_cm and not LINK_RE.search(text) and TH19_TITLE_RE.search(sn.get("title", "")) and quota + 1 <= budget:
            try:
                quota += 1
                extra = _creator_comment_links(vid, cid, key)
            except Exception:
                pass
        if _accept(cat, vid, _video_record(vid, sn, extra), th_ok):
            added += 1
            c["hits"] = c.get("hits", 0) + 1
            if not c.get("name"):
                c["name"] = sn.get("channelTitle") or None
        else:
            seen.setdefault(vid, "skip")

    # a rotazione: prima i canali controllati da più tempo (i nuovi per primi)
    for cid, c in sorted(creators.items(), key=lambda kv: (kv[1].get("checked", 0), -kv[1].get("hits", 0))):
        if c.get("hits", 0) < min_hits or not cid.startswith("UC") or not do_creators:
            continue
        if stop or used_pages >= crawl_pages or quota + 1 > budget:
            break
        token = c.get("next") if not c.get("done") else None
        pages_here = 0
        while True:
            params = {"part": "snippet", "playlistId": "UU" + cid[2:], "maxResults": 50}
            if token:
                params["pageToken"] = token
            try:
                data = yt_get("playlistItems", params, key)
            except Exception as e:
                log(f"⚠️  Archivio del canale {c.get('name', cid)} non leggibile: {e}")
                c["done"] = True
                break
            quota += 1; used_pages += 1; pages_here += 1
            fresh = 0
            for it in data.get("items", []):
                sn = it.get("snippet") or {}
                vid = (sn.get("resourceId") or {}).get("videoId")
                if not vid or vid in seen:
                    continue
                fresh += 1
                take(vid, sn, cid, c)
            token = data.get("nextPageToken")
            if c.get("done") and fresh == 0:                # già letto tutto: basta la prima pagina
                break
            if not token:
                c["done"], c["next"] = True, None
                break
            if used_pages >= crawl_pages or quota + 1 > budget:
                c["next"] = token                            # si riprende al giro successivo
                break
        c["name"] = c.get("name") or cid
        c["checked"] = now

    # 3b) nuovi creator: i canali "in evidenza" dei creator già seguiti (spesso altri base builder)
    found_c = 0
    if ycfg.get("discover_featured", True) and do_creators:
        for cid, c in list(creators.items()):
            if c.get("hits", 0) < min_hits or not cid.startswith("UC") or c.get("fc"):
                continue
            if quota + 1 > budget:
                break
            c["fc"] = True
            try:
                quota += 1
                data = yt_get("channelSections", {"part": "contentDetails", "channelId": cid}, key)
            except Exception:
                continue
            for sec in data.get("items", []):
                for ch in ((sec.get("contentDetails") or {}).get("channels") or []):
                    if isinstance(ch, str) and ch.startswith("UC") and ch not in creators:
                        creators[ch] = {"name": None, "hits": 0, "cand": True, "via": cid}
                        found_c += 1
        # prova i candidati: basta la prima pagina dei loro video per capire se pubblicano basi TH19
        probes = 0
        for cid, c in list(creators.items()):
            if not c.get("cand") or c.get("probed"):
                continue
            if probes >= int(ycfg.get("candidate_probes_per_run", 200)) or quota + 1 > budget:
                break
            c["probed"] = True
            try:
                quota += 1; probes += 1
                data = yt_get("playlistItems", {"part": "snippet", "playlistId": "UU" + cid[2:], "maxResults": 50}, key)
            except Exception:
                continue
            for it in data.get("items", []):
                sn = it.get("snippet") or {}
                vid = (sn.get("resourceId") or {}).get("videoId")
                if vid and vid not in seen:
                    take(vid, sn, cid, c)
            if data.get("nextPageToken"):
                c["next"] = data["nextPageToken"]            # il resto dell'archivio al giro dopo
            else:
                c["done"] = True

    # 4) durata dei video con più basi (serve per abbinare i fotogrammi ai link)
    need = [v for v, it in items.items() if len(it["links"]) > 1 and "dur" not in it]
    for i in range(0, len(need), 50):
        try:
            data = yt_get("videos", {"part": "contentDetails", "id": ",".join(need[i:i + 50])}, key)
        except Exception:
            break
        quota += 1
        for v in data.get("items", []):
            items[v["id"]]["dur"] = iso_duration_seconds((v.get("contentDetails") or {}).get("duration"))

    if do_search:
        cat.data["last_run"] = now
    if do_creators:
        cat.data["last_creators"] = now
    qd["used"] = int(qd.get("used", 0)) + quota
    cat.changed = True
    n_creators = sum(1 for c in creators.values() if c.get("hits", 0) >= min_hits)
    what = "ricerche + creator" if do_search else "solo creator"
    log(f"▶️  YouTube ({what}): {len(new_ids)} video dalle ricerche, {added} con link TH19 aggiunti, "
        f"{n_creators} creator seguiti" + (f", {found_c} canali nuovi da provare" if found_c else "") +
        f" (quota ≈ {quota} unità, oggi {qd['used']})")
    return added


def iso_duration_seconds(s):
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", s or "")
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def frame_urls(vid, n):
    """Fotogrammi automatici di YouTube (n = 1, 2, 3 → circa 25%, 50%, 75% del video)."""
    return [f"https://i.ytimg.com/vi/{vid}/{p}{n}.jpg" for p in ("maxres", "sd", "hq")]


def _pick_link(links, frac, dur):
    """Quale link corrisponde al fotogramma a 'frac' del video."""
    timed = [(t, l) for l, t in links if t is not None]
    if dur and len(timed) >= max(2, len(links) // 2):
        t_f = frac * dur
        best = None
        for t, l in sorted(timed):
            if t <= t_f + 3:
                best = l
        return (best or sorted(timed)[0][1]), int(t_f), False
    idx = min(len(links) - 1, int(frac * len(links)))  # niente minuti: si stima dall'ordine
    return links[idx][0], (int(frac * dur) if dur else None), True


def index_youtube(idx: Index, cat: Catalog, deadline, th_ok=("TH19",)):
    added = 0
    th_ok = {t.upper() for t in th_ok}
    failed = idx.meta.setdefault("failed", {})
    for vid, it in cat.data["items"].items():
        links = it.get("links") or ([[it["link"], None]] if it.get("link") else [])
        if not links:
            continue
        ok_links = [l for l, _ in links if not th_ok or th_of(l) in th_ok]
        if not ok_links:
            continue
        watch = f"https://www.youtube.com/watch?v={vid}"
        jobs = []
        if len(links) == 1:
            jobs.append((f"yt_{vid}", it["thumb"], links[0][0], watch, None, False, False))
        else:
            jobs.append((f"yt_{vid}_t", it["thumb"], None, watch, ok_links, True, False))
            for n, frac in ((1, 0.25), (2, 0.5), (3, 0.75)):
                link, t, guess = _pick_link(links, frac, it.get("dur"))
                if link not in ok_links:
                    continue
                jump = watch + (f"&t={t}s" if t else "")
                jobs.append((f"yt_{vid}_f{n}", frame_urls(vid, n), link, jump, ok_links, guess, True))
        for key, url, link, jump, all_links, guess, need_area in jobs:
            if key in idx.meta["entries"] or key in failed:
                continue
            if time.time() > deadline:
                log("⏱️  Tempo quasi finito: continuo con le immagini al prossimo giro")
                return added
            img, used = None, None
            for u in (url if isinstance(url, list) else [url]):
                try:
                    img = decode_image(download(u, retries=1))
                except Exception:
                    img = None
                if img is not None:
                    used = u
                    break
            # un fotogramma senza area di gioco (intro, facecam...) non è una base
            if img is None or (need_area and play_area_box(img) is None) or not idx.add(
                    key, img, link or all_links[0], jump, time.time(), img_url=used):
                failed[key] = 1
                idx.changed = True
                continue
            e = idx.meta["entries"][key]
            if all_links and len(all_links) > 1:
                e["links"] = all_links
                e["guess"] = bool(guess)
                if link is None:
                    e["video_only"] = True                 # miniatura di un video con più basi
            added += 1
            if added % 25 == 0:
                idx.save()
    return added

# ═════════════════════════════ Telegram ═════════════════════════════
# I canali Telegram PUBBLICI hanno una versione web (t.me/s/<canale>) leggibile
# senza login: il bot legge tutti i post (foto + link della base) e li aggiunge
# all'archivio. Nessuna chiave necessaria.

TG_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TG_POST_RE = re.compile(r'data-post="([A-Za-z0-9_]+)/(\d+)"')
TG_TAG_RE = re.compile(r"<(?:a|i|div)\s[^>]*>", re.I)
TG_BG_RE = re.compile(r"background-image:\s*url\(\s*['\"]?([^'\")]+)['\"]?\s*\)")
TG_HREF_RE = re.compile(r'href="([^"]+)"')
TG_TIME_RE = re.compile(r'<time[^>]*datetime="([^"]+)"')


def tg_get(url, retries=2):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": TG_UA, "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or attempt == retries - 1:
                raise
            time.sleep(3)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)


def tg_channel_name(s):
    s = str(s).strip()
    m = re.search(r"(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z0-9_]{4,})", s)
    if m:
        return m.group(1)
    s = s.lstrip("@")
    return s if re.fullmatch(r"[A-Za-z0-9_]{4,}", s) else None


def tg_parse(page):
    """Post di una pagina web di Telegram (t.me/s/... o un singolo post ?embed=1):
    numero del post, foto (in ordine), link delle basi, data."""
    if isinstance(page, bytes):
        page = page.decode("utf-8", "replace")
    posts = []
    starts = [m for m in TG_POST_RE.finditer(page)]
    for i, m in enumerate(starts):
        block = page[m.start(): starts[i + 1].start() if i + 1 < len(starts) else len(page)]
        head = page[page.rfind("<", 0, m.start()): m.start()]        # il tag del messaggio stesso
        if "service_message" in head:
            continue
        photos = []
        for tag in TG_TAG_RE.findall(block):
            video = "message_video_thumb" in tag
            if "message_photo_wrap" not in tag and not video:
                continue
            if "link_preview" in tag:
                continue
            bg = TG_BG_RE.search(tag)
            if not bg:
                continue
            href = TG_HREF_RE.search(tag)
            pid = re.search(r"/(\d+)(?:\?|$)", href.group(1)) if href else None
            photos.append([int(pid.group(1)) if pid else int(m.group(2)), html.unescape(bg.group(1)), video])
        text = html.unescape(block)
        links, have = [], set()
        for l in LINK_RE.findall(text):
            l = l.rstrip(".,;:!*_~|")
            lid = layout_id(l) or l
            if lid not in have:
                have.add(lid); links.append(l)
        t = TG_TIME_RE.search(block)
        try:
            ts = datetime.fromisoformat(t.group(1)).timestamp() if t else time.time()
        except ValueError:
            ts = time.time()
        posts.append({"chan": m.group(1), "id": int(m.group(2)), "photos": photos, "links": links, "ts": ts})
    return posts


def tg_post_photos(chan, post):
    """Indirizzi aggiornati delle foto di un post (quelli vecchi scadono dopo un po')."""
    try:
        for p in tg_parse(tg_get(f"https://t.me/{chan}/{post}?embed=1&mode=tme")):
            if p["id"] == int(post) and p["photos"]:
                return p["photos"]
    except Exception:
        pass
    return None


def telegram_collect(tcfg: dict, cat: Catalog, th_ok=("TH19",)):
    """Legge i canali Telegram pubblici: al primo avvio tutto lo storico, poi solo i post nuovi.
    Tiene solo i post con almeno un link dei municipi scelti (gli altri vengono ignorati)."""
    if not tcfg or not tcfg.get("enabled", True):
        return
    chans = [c for c in (tg_channel_name(x) for x in tcfg.get("channels", [])) if c]
    if not chans:
        return
    states, items = cat.data.setdefault("tg", {}), cat.data.setdefault("tg_items", {})
    every = float(tcfg.get("every_hours", 3)) * 3600
    max_pages = int(tcfg.get("max_pages_per_run", 150))
    t_end = time.time() + float(tcfg.get("max_minutes", 10)) * 60    # tempo massimo per tutti i canali
    th_ok = {t.upper() for t in th_ok}
    for chan in chans:
        cl = chan.lower()
        st = states.setdefault(cl, {"newest": 0, "oldest": None, "done": False, "last_run": None})
        if st["last_run"] and time.time() - st["last_run"] < every - 300 and st["done"]:
            continue
        got, pages, blocked = {}, 0, None

        def walk(before, stop_at):
            """Pagine all'indietro da 'before'. Ritorna (id più basso visto, storico finito?)."""
            nonlocal pages, blocked
            low = None
            while pages < max_pages and time.time() < t_end:
                url = f"https://t.me/s/{chan}" + (f"?before={before}" if before else "")
                try:
                    posts = [p for p in tg_parse(tg_get(url)) if p["chan"].lower() == cl]
                except Exception as e:
                    blocked = str(e)
                    return low, False
                pages += 1
                if not posts:
                    return low, True
                for p in posts:
                    got[p["id"]] = p
                lo = min(p["id"] for p in posts)
                if (before is not None and lo >= before) or lo <= 1:
                    return min(lo, low or lo), True
                low = lo if low is None else min(low, lo)
                if lo <= stop_at:
                    return low, False
                before = lo
                time.sleep(0.8)
            return low, False

        # 1) i post nuovi (fino a quelli già letti)
        low, end = walk(None, st["newest"])
        if st["newest"] == 0:                       # primo avvio: questo giro è anche lo storico
            st["oldest"], st["done"] = low, end
        elif end:
            st["done"] = True
        # 2) lo storico non ancora letto (se il primo avvio non è bastato)
        if not st["done"] and st["oldest"] and not blocked and pages < max_pages:
            low2, end2 = walk(st["oldest"], 0)
            if low2:
                st["oldest"] = min(st["oldest"], low2)
            st["done"] = end2
        if got:
            st["newest"] = max(st["newest"], max(got))
        if not blocked:
            st["last_run"] = time.time()
        # foto + link nello stesso post, oppure foto e subito dopo un post con il link
        n_new = n_link_only = n_other = 0
        pending = None
        for pid in sorted(got):
            p = got[pid]
            photos, links, src = p["photos"], p["links"], p
            if photos and not links:
                pending = p
                continue
            if links and not photos:
                last_photo = max([pending["id"]] + [ph[0] for ph in pending["photos"]]) if pending else 0
                if pending and p["id"] - last_photo <= 3 and abs(p["ts"] - pending["ts"]) < 3600:
                    photos, src = pending["photos"], pending
                else:
                    n_link_only += 1
            pending = None
            if not photos or not links:
                continue
            if th_ok and not any(th_of(l) in th_ok for l in links):
                n_other += 1                              # base di un altro municipio
                continue
            key = f"{cl}-{src['id']}"
            if key not in items:
                items[key] = {"chan": chan, "post": src["id"], "photos": photos, "links": links, "ts": src["ts"]}
                n_new += 1
        if got or n_new:
            cat.changed = True
        log(f"✈️  Telegram {chan}: {len(got)} post letti in {pages} pagine, +{n_new} con foto e link"
            + (f", {n_other} di altri municipi ignorati" if n_other else "")
            + (f", {n_link_only} con il link ma senza foto" if n_link_only else "")
            + ("" if st["done"] else " (storico non ancora finito: continuo al prossimo giro)")
            + (f" ⚠️ interrotto: {blocked}" if blocked else ""))


def _tg_refresh(cat, chan, post, done):
    """Rinnova in un colpo solo gli indirizzi delle foto di ~20 post (quelli vecchi scadono).
    Ritorna False se Telegram non risponde."""
    cl = chan.lower()
    if (cl, post) in done:
        return done[(cl, post)]
    try:
        posts = tg_parse(tg_get(f"https://t.me/s/{chan}?before={post + 1}"))
    except Exception as e:
        log(f"⚠️  Telegram non risponde ({e}): continuo con le sue immagini al prossimo giro")
        done[(cl, post)] = False
        return False
    items = cat.data.get("tg_items", {})
    for p in posts:
        done[(cl, p["id"])] = True
        it = items.get(f"{cl}-{p['id']}")
        if it and p["photos"] and len(p["photos"]) == len(it["photos"]) and p["photos"] != it["photos"]:
            it["photos"] = p["photos"]
            cat.changed = True
    done.setdefault((cl, post), True)
    time.sleep(0.5)
    return True


def index_telegram(idx: Index, cat: Catalog, deadline, th_ok=("TH19",)):
    added = 0
    th_ok = {t.upper() for t in th_ok}
    failed = idx.meta.setdefault("failed", {})
    refreshed = {}

    def fetch(url):
        try:
            return decode_image(tg_get(url, retries=1)), True
        except Exception:
            return None, False                       # non scaricata (indirizzo scaduto o rete)

    for key, it in cat.data.get("tg_items", {}).items():
        links = it["links"]
        ok_links = [l for l in links if not th_ok or th_of(l) in th_ok]
        if not ok_links:
            continue
        jump = f"https://t.me/{it['chan']}/{it['post']}"
        for n in range(len(it["photos"])):
            photos = it["photos"]
            if len(links) == 1:
                link, all_links, guess, only = links[0], None, False, False
            elif len(photos) == len(links):           # album: foto e link nello stesso ordine
                link, all_links, guess, only = links[n], ok_links, True, False
            else:                                     # una foto, più link: non si sa quale sia
                link, all_links, guess, only = None, ok_links, False, True
            if link is not None and link not in ok_links:
                continue
            ekey = f"tg_{key}-{n}"
            if ekey in idx.meta["entries"] or ekey in failed:
                continue
            if time.time() > deadline:
                log("⏱️  Tempo quasi finito: continuo con le immagini di Telegram al prossimo giro")
                return added
            used = photos[n][1]
            img, got = fetch(used)
            if not got:
                if not _tg_refresh(cat, it["chan"], it["post"], refreshed):
                    return added                       # Telegram non risponde: niente scarti, si riprova dopo
                if it["photos"][n][1] != used:
                    used = it["photos"][n][1]
                    img, got = fetch(used)
            # negli album e nei video conta solo ciò che mostra davvero un villaggio
            need_area = len(photos) > 1 or len(links) > 1 or photos[n][2]
            if img is None or (need_area and play_area_box(img) is None) or not idx.add(
                    ekey, img, link or all_links[0], jump, it["ts"], img_url=used):
                failed[ekey] = 1
                idx.changed = True
                continue
            e = idx.meta["entries"][ekey]
            e["tg"] = [it["chan"], it["post"], n]
            if all_links and len(all_links) > 1:
                e["links"] = all_links
                e["guess"] = bool(guess)
                if only:
                    e["video_only"] = True
            added += 1
            if added % 25 == 0:
                idx.save()
    return added

# ═════════════════════════════ Ricerca ═════════════════════════════

def source_of(key):
    """Video o post da cui viene un'immagine dell'archivio (più immagini possono venire dallo stesso)."""
    if key.startswith("yt_"):
        return "yt:" + key[3:14]                   # gli ID dei video YouTube sono lunghi 11 caratteri
    if key.startswith("tg_"):
        return "tg:" + key[3:].rsplit("-", 1)[0]
    return key


def entry_image(key, e):
    if e.get("img"):
        return e["img"]
    if key.startswith("yt_"):
        return f"https://i.ytimg.com/vi/{key[3:]}/hqdefault.jpg"
    return None


def match_percent(scores, k):
    """Quanto il risultato k spicca su tutte le altre basi dell'archivio (0-99%).

    Si confronta il punteggio con quello "di fondo" delle basi non correlate
    (mediana e dispersione) e con il risultato successivo: una base giusta
    stacca nettamente sia il resto dell'archivio sia la seconda classificata.
    """
    s = scores
    bg = s[3:] if len(s) >= 8 else s[1:]
    med = float(np.median(bg)) if len(bg) else 8.0
    mad = float(np.median(np.abs(np.asarray(bg) - med))) * 1.4826 if len(bg) else 1.5
    sd = max(mad, 1.0, 0.08 * med)
    z = (s[k] - med) / sd
    gap = (s[k] - (s[k + 1] if k + 1 < len(s) else med)) / sd
    e = 0.5 * z + 0.5 * gap
    pct = 100.0 / (1.0 + np.exp(-(e - 4.5) * 1.2))
    # tetto in base al numero di edifici allineati: pochi edifici = mai una percentuale alta
    cap = 100.0 / (1.0 + np.exp(-(s[k] - 22.0) / 5.0))
    return int(max(0, min(99, round(min(pct, cap)))))


_POOL_DATA = {}


def _score_worker(key):
    return key, score(_POOL_DATA["q"], _POOL_DATA["feats"][key], fast=_POOL_DATA.get("fast", False))


def _score_pass(qf, feats_all, keys, fast):
    workers = min(os.cpu_count() or 1, 8)
    if workers > 1 and len(keys) >= 24:
        import multiprocessing as mp
        try:
            _POOL_DATA.update(q=qf, feats=feats_all, fast=fast)
            with mp.get_context("fork").Pool(workers) as pool:
                return dict(pool.map(_score_worker, keys, chunksize=max(1, len(keys) // (workers * 4))))
        except Exception as e:
            log(f"⚠️  Confronto parallelo non riuscito ({e}), procedo in sequenza")
        finally:
            _POOL_DATA.clear()
    return {k: score(qf, feats_all[k], fast=fast) for k in keys}


FULL_CHECK = 80          # quante basi ricontrollare con il confronto completo


def _score_all(qf, feats_all, keys):
    """Confronta lo screenshot con tutte le basi: prima un passaggio veloce su tutto l'archivio,
    poi il confronto completo (tutte le rotazioni) sulle più promettenti."""
    if len(keys) <= FULL_CHECK * 2:
        return _score_pass(qf, feats_all, keys, fast=False)
    first = _score_pass(qf, feats_all, keys, fast=True)
    top = sorted(keys, key=lambda k: -first[k][0])[:FULL_CHECK]
    first.update(_score_pass(qf, feats_all, top, fast=False))
    return first


def search(idx: Index, qf, th=None, top=3):
    feats_all = idx.load_all()
    best_by_link = {}
    keys = [k for k in feats_all
            if not (th and idx.meta["entries"][k].get("th") and idx.meta["entries"][k]["th"] != th)]
    all_scores = _score_all(qf, feats_all, keys)
    for key in keys:
        e = idx.meta["entries"][key]
        sc, how = all_scores[key]
        # stessa base = stesso ID nel link (anche se cambia la lingua: /it?, /en?...)
        group = ("src:" + source_of(key)) if e.get("video_only") else (layout_id(e["link"]) or e["link"])
        cur = best_by_link.get(group)
        if cur is None or sc > cur[0]:
            best_by_link[group] = (sc, how, e, key)
    ranked = sorted(best_by_link.values(), key=lambda x: -x[0])

    results = []
    for k, (sc, how, e, key) in enumerate(ranked[:top]):
        # le altre immagini dello STESSO video (miniatura, altri fotogrammi) non sono concorrenti
        same = source_of(key)
        scores = [r[0] for i, r in enumerate(ranked) if i == k or source_of(r[3]) != same]
        kk = scores.index(sc) if sc in scores else 0
        results.append({"score": sc, "pct": match_percent(scores, kk), "how": how,
                        "link": e["link"], "th": e.get("th"), "jump": e["jump"], "img": entry_image(key, e),
                        "key": key, "img_exact": bool(e.get("img")), "area": e.get("area"),
                        "links": e.get("links"), "guess": e.get("guess", False), "video_only": e.get("video_only", False),
                        "tg": e.get("tg")})
    if results:                       # immagine pulita della base trovata (solo per la prima)
        results[0]["crop"] = card_image(qf, feats_all.get(results[0]["key"]), results[0])
    return results, len(feats_all)


def confidence(results, cfg):
    if not results:
        return "none"
    p = results[0]["pct"]
    if p >= 80:
        return "high"
    if p >= 50:
        return "mid"
    if p >= 25:
        return "low"
    return "none"


COLORS = {"high": 0x2ECC71, "mid": 0xF1C40F, "low": 0xE67E22, "none": 0x95A5A6}


def level(p):
    return "high" if p >= 80 else "mid" if p >= 50 else "low" if p >= 25 else "none"


def card_image(qf, bf, r):
    """Scarica l'immagine d'archivio della base trovata e ne ritaglia solo la base, ingrandita."""
    if not r.get("img") or bf is None:
        return None
    try:
        img = decode_image(download(r["img"]))
    except Exception:
        img = None
    if img is None and r.get("tg"):                  # indirizzo Telegram scaduto: ne chiedo uno nuovo
        fresh = tg_post_photos(r["tg"][0], r["tg"][1])
        if fresh and len(fresh) > r["tg"][2]:
            r["img"] = fresh[r["tg"][2]][1]
            try:
                img = decode_image(tg_get(r["img"], retries=1))
            except Exception:
                img = None
    if img is None:
        return None
    try:
        if r.get("img_exact") and r.get("area"):
            return base_crop(img, None, area=r["area"])
        region = match_region(qf, bf) if r.get("img_exact") else None
        return base_crop(img, region)
    except Exception as e:
        log(f"⚠️  Ritaglio non riuscito ({e})")
        return None


def build_reply(per_image, n_archive, cfg):
    """Testo in alto + UNA scheda con la base trovata: immagine pulita, percentuale, link cliccabile.
    Restituisce (testo, schede, file da allegare)."""
    heads, embeds, files = [], [], []
    alt = bool(cfg.get("show_alternatives", False))
    for k, (results, conf) in enumerate(per_image):
        pre = f"**Image {k + 1}:** " if len(per_image) > 1 else ""
        best = f" ({results[0]['pct']}%)" if results else ""
        heads.append(pre + {"high": "🟢 **Match found**" + best,
                            "mid": "🟡 **Likely match**" + best,
                            "low": "🟠 **Possible match**" + best + ": compare the picture before using it",
                            "none": "⚪ **No similar base** in the archive"}[conf])
        if not results or (conf == "none" and not cfg.get("show_nearest_when_none", True)):
            continue
        r = results[0]
        kind = {"A": "same orientation", "M": "mirrored"}.get(r["how"], "rotated/mirrored")
        yt = "youtube.com" in r["jump"]
        what = "video" if yt else "post"
        src = "▶️ Source video" if yt else "✈️ Telegram post" if "t.me/" in r["jump"] else "💬 Original post"
        desc = f"**[📋 Copy base in game]({r['link']})**\n[{src}]({r['jump']}) · {kind}"
        many = [l for l in (r.get("links") or []) if l]
        if r.get("video_only") and len(many) > 1:
            # miniatura di un video con più basi: non si sa quale sia, mostra tutti i link
            desc = (f"This {what} shows **{len(many)} bases** and the picture matches one of them:\n"
                    + " · ".join(f"[Base {i}]({l})" for i, l in enumerate(many[:12], 1))
                    + f"\n[{src}]({r['jump']}) · {kind}")
        elif len(many) > 1:
            others = [l for l in many if l != r["link"]][:10]
            note = f" (guessed from the order of the {what})" if r.get("guess") else ""
            desc += (f"\nThis {what} has {len(many)} bases{note}. Not this one? "
                     + " · ".join(f"[{i}]({l})" for i, l in enumerate(others, 2)))
        if alt and len(results) > 1:
            desc += "\n\nOther candidates: " + " · ".join(
                f"[{x['pct']}%]({x['link']})" for x in results[1:])
        title = f"{r['pct']}% match" + (f" · {r['th']}" if r.get("th") else "")
        title += " · pick the base below" if (r.get("video_only") and len(many) > 1) else " · tap to copy"
        emb = {"title": title, "url": r["jump"] if (r.get("video_only") and len(many) > 1) else r["link"],
               "description": desc[:4000], "color": COLORS[level(r["pct"])]}
        if r.get("crop"):
            name = f"base_{k + 1}.jpg"
            files.append((name, r["crop"]))
            emb["image"] = {"url": f"attachment://{name}"}
        elif r.get("img"):
            emb["image"] = {"url": r["img"]}
        embeds.append(emb)
    if embeds:
        embeds[-1]["footer"] = {"text": f"Archive: {n_archive} TH19 bases"}
    return "\n".join(heads)[:1990], embeds, files


def discord_multipart(method, url, payload, files, headers=None, retries=4):
    """Invia un messaggio con allegati (immagini) a Discord."""
    boundary = "----BaseFinder" + os.urandom(8).hex()
    if files:
        payload = dict(payload)
        payload["attachments"] = [{"id": i, "filename": n} for i, (n, _) in enumerate(files)]
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
             f'Content-Type: application/json\r\n\r\n'.encode() + json.dumps(payload).encode() + b"\r\n"]
    for i, (name, data) in enumerate(files):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files[{i}]"; filename="{name}"\r\n'
                     f'Content-Type: image/jpeg\r\n\r\n'.encode() + data + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": UA}
    h.update(headers or {})
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")
            if e.code == 429:
                time.sleep(2)
                continue
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(2)
                continue
            raise DiscordError(e.code, txt) from None


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
            th = None                       # l'archivio contiene solo basi TH19
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
                results, n_arch = search(idx, qf, th, int(cfg.get("max_results", 3)))
                per_image.append((results, confidence(results, cfg)))
                log(f"🔎 richiesta {m['id']}: " + ", ".join(f"{r['score']}{r['how']}={r['pct']}%" for r in results))
            if not idx.meta["entries"]:
                content, embeds, files = "⚠️ The base archive is still empty: wait for the first YouTube collection.", [], []
            else:
                content, embeds, files = build_reply(per_image, n_arch or len(idx.meta["entries"]), cfg)
            if not idx.meta["entries"]:
                files = []
            discord_multipart("POST", f"{DISCORD_API}/channels/{cid}/messages", {
                "content": content,
                "embeds": embeds,
                "message_reference": {"message_id": m["id"], "fail_if_not_exists": False},
                "allowed_mentions": {"parse": [], "replied_user": False},
            }, files, headers={"Authorization": f"Bot {token}"})
            try:
                dapi("PUT", f"/channels/{cid}/messages/{m['id']}/reactions/{urllib.parse.quote(done_emoji)}/@me", token)
                dapi("DELETE", emoji_path, token)
            except DiscordError:
                pass
            answered += 1
    return answered


def interaction_reply(app_id, token, content, embeds, files=()):
    url = f"https://discord.com/api/v10/webhooks/{app_id}/{token}/messages/@original"
    discord_multipart("PATCH", url, {"content": content, "embeds": embeds, "allowed_mentions": {"parse": []}},
                      list(files))


def run_interaction(idx: Index, cfg):
    app_id = os.environ.get("INTERACTION_APP_ID", "").strip()
    token = os.environ.get("INTERACTION_TOKEN", "").strip()
    try:
        urls = json.loads(os.environ.get("INTERACTION_IMAGES", "[]"))
    except json.JSONDecodeError:
        urls = []
    m = re.search(r"(\d{1,2})", os.environ.get("INTERACTION_TH", ""))
    th = f"TH{m.group(1)}" if m else None
    if not app_id or not token:
        log("❌ Mancano i dati del comando Discord")
        return 2
    try:
        if not idx.meta["entries"] or idx.outdated:
            interaction_reply(app_id, token, "⏳ The base archive is being updated. Please try again in a few minutes.", [])
            return 0
        per_image, n_arch = [], len(idx.meta["entries"])
        for url in urls[:3]:
            try:
                img = decode_image(download(url))
            except Exception:
                img = None
            if img is None:
                per_image.append(([], "none"))
                continue
            t = time.time()
            results, n_arch = search(idx, feats(img), th, int(cfg.get("max_results", 3)))
            per_image.append((results, confidence(results, cfg)))
            log(f"🔎 /findbase: " + ", ".join(f"{r['pct']}%" for r in results) + f" in {time.time() - t:.0f}s")
        content, embeds, files = build_reply(per_image, n_arch, cfg)
        interaction_reply(app_id, token, content, embeds, files)
    except Exception as e:
        log(f"❌ Errore: {e}")
        try:
            interaction_reply(app_id, token, "⚠️ Something went wrong while searching. Please try again in a minute.", [])
        except Exception:
            pass
        return 1
    return 0


def register_commands(token):
    app = dapi("GET", "/oauth2/applications/@me", token)
    cmds = [
        {"name": "findbase", "type": 1,
         "description": "Find the link of a Clash of Clans base from a screenshot",
         "options": [
             {"type": 11, "name": "screenshot", "description": "Screenshot of the base", "required": True},
         ],
         "integration_types": [0], "contexts": [0]},
        {"name": "Find base link", "type": 3, "integration_types": [0], "contexts": [0]},
    ]
    dapi("PUT", f"/applications/{app['id']}/commands", token, body=cmds)
    log(f"✅ Comandi registrati per {app.get('name', app['id'])}: /findbase e 'Find base link'")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Cerca il link di una base da uno screenshot")
    ap.add_argument("--config", default=str(BASE_DIR / "cerca_base.toml"))
    ap.add_argument("--index", default=str(BASE_DIR / "base_index"))
    ap.add_argument("--catalog", default=str(BASE_DIR / "yt_catalog"))
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--image", help="prova locale: confronta un'immagine con l'archivio e stampa i risultati")
    ap.add_argument("--interaction", action="store_true", help="risponde a un comando /findbase (avviato dal Worker)")
    ap.add_argument("--register-commands", action="store_true", help="registra /findbase su Discord (una volta)")
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    with open(args.config, "rb") as f:
        full_cfg = tomllib.load(f)
    cfg = full_cfg.get("cerca_base", {})
    ycfg = full_cfg.get("youtube", {})
    if args.register_commands:
        return register_commands(os.environ.get("DISCORD_BOT_TOKEN", "").strip())
    idx = Index(Path(args.index))
    (idx.root / "CHANGED").unlink(missing_ok=True)
    if args.interaction:
        return run_interaction(idx, cfg)
    if idx.outdated:
        log("♻️  Metodo di confronto aggiornato: ricostruisco l'archivio (nessuna quota YouTube usata)")
        idx.reset()

    if args.image:
        img = cv2.imread(args.image)
        res, n = search(idx, feats(img), None, int(cfg.get("max_results", 3)))
        print(f"Archivio: {n} basi — affidabilità: {confidence(res, cfg)}")
        for r in res:
            print(f"  {r['pct']:3d}%  ({r['score']} {r['how']})  {r.get('th')}  {r['link']}")
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
    tcfg = full_cfg.get("telegram", {})
    try:
        telegram_collect(tcfg, cat, tcfg.get("town_halls", ycfg.get("town_halls", ["TH19"])))
    except Exception as e:
        log(f"⚠️  Telegram non letto: {e}")
    cat.save()
    for key, e in idx.meta["entries"].items():
        if key.startswith("yt_") and not e.get("img") and key[3:] in cat.data["items"]:
            e["img"] = cat.data["items"][key[3:]]["thumb"]
            idx.changed = True
    th_ok = ycfg.get("town_halls", ["TH19"])
    added += index_telegram(idx, cat, idx_deadline, tcfg.get("town_halls", th_ok))
    added += index_youtube(idx, cat, idx_deadline, th_ok)
    cat.save()
    idx.save()
    n_yt = sum(1 for k in idx.meta["entries"] if k.startswith("yt_"))
    n_tg = sum(1 for k in idx.meta["entries"] if k.startswith("tg_"))
    log(f"📚 Archivio: {len(idx.meta['entries'])} basi ({n_yt} da YouTube, {n_tg} da Telegram, +{added} nuove)")
    n = answer_queries(idx, token, searchc, cfg, deadline)
    idx.save()
    log(f"Fatto: {n} richieste risposte in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
