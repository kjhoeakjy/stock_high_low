"""공통: HTTP, 경로, 숫자 파싱."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STOCK_DATA", ROOT / "data"))
CACHE = Path(os.environ.get("STOCK_CACHE", ROOT / "cache"))
OUT = Path(os.environ.get("STOCK_OUT", ROOT.parent / "docs"))
for _p in (DATA, CACHE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Referer": "https://m.stock.naver.com/",
    "Accept": "*/*",
}


def http(url: str, *, data: dict | None = None, headers: dict | None = None,
         timeout: int = 30, retries: int = 4, encoding: str | None = None) -> str | None:
    """GET/POST with retries. Returns text or None on hard failure (404 etc.)."""
    h = dict(UA)
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        h.setdefault("Content-Type", "application/x-www-form-urlencoded")
    last = None
    for i in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=h, data=body), timeout=timeout)
            raw = r.read()
            if encoding:
                return raw.decode(encoding, "replace")
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("euc-kr", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 404):
                return None
            time.sleep(1.5 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    print(f"[http] fail {url}: {last}")
    return None


def http_json(url: str, **kw):
    t = http(url, **kw)
    if t is None:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def pmap(fn, items, workers: int = 16, label: str = ""):
    """Threaded map that keeps order and reports progress."""
    out = [None] * len(items)
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in cf.as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[pmap] {label} item {i} error: {e}")
            done += 1
            if done % 500 == 0 or done == len(items):
                print(f"[pmap] {label} {done}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    return out


_num = re.compile(r"[^0-9.\-]")


def num(s, default=None):
    """'1,234' / '+75,714' / '45.95%' / '8.14배' -> float."""
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    s = _num.sub("", str(s).replace("−", "-"))
    if s in ("", "-", ".", "-."):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def today_kst() -> str:
    import datetime as dt
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y%m%d")
