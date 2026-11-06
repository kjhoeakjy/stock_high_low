"""장중 실시간 스냅샷(표준 라이브러리만). GitHub Actions에서 5분마다 실행.

입력: docs/latest.json(아침 후보·상한가 가격), docs/scores.json(전 종목 아침 확률)
출력: live/live.json(화면용), live/snap/YYYYMMDD.csv(장중 스냅샷 누적 → 추후 장중 모델 학습용)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

from stock.common import OUT, http_json, num, pmap

NAVER = "https://m.stock.naver.com/api"
LIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "live"
HOLIDAYS_PATH = None


def now_kst() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)


def tick(p: float) -> int:
    return 1 if p < 2000 else 5 if p < 5000 else 10 if p < 20000 else 50 if p < 50000 else 100 if p < 200000 else 500 if p < 500000 else 1000


def up_limit(prev: float) -> float:
    u = prev * 1.3
    return math.floor(u / tick(u)) * tick(u)


_UNIT = {"조": 1e12, "억": 1e8, "만": 1e4}


def kval(s) -> float | None:
    """'699억' '1.2조' '3,400만' → 원 단위."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace(",", "").strip()
    m = re.match(r"^(-?[\d.]+)\s*([조억만])?", s)
    if not m:
        return None
    return float(m.group(1)) * _UNIT.get(m.group(2), 1.0)


REALTIME = "https://polling.finance.naver.com/api/realtime/domestic/stock/"


def fetch_all(codes: list[str]) -> dict[str, dict]:
    """전 종목 현재가를 100개씩 묶어 조회."""
    chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]

    def one(chunk):
        j = http_json(REALTIME + ",".join(chunk))
        res = {}
        for s in (j or {}).get("datas") or []:
            r = _row(s)
            if r:
                res[r["code"]] = r
        return res

    out = {}
    for r in pmap(one, chunks, workers=6, label="realtime"):
        out.update(r)
    return out


def _row(s: dict) -> dict | None:
    """네이버 목록/기본 응답 한 건 → 공통 형식."""
    price = num(s.get("closePriceRaw") or s.get("closePrice"))
    chg = num(s.get("compareToPreviousClosePriceRaw") or s.get("compareToPreviousClosePrice"))
    if not price or chg is None:
        return None
    sign = s.get("compareToPreviousPrice", {}) or {}
    if isinstance(sign, dict) and sign.get("code") in ("4", "5"):  # 하락
        chg = -abs(chg)
    prev = price - chg
    if prev <= 0:
        return None
    return {
        "code": s.get("itemCode") or s.get("stockCode"), "name": s.get("stockName"),
        "price": price, "prev": prev, "ret": price / prev - 1,
        "high": num(s.get("highPriceRaw") or s.get("highPrice")), "open": num(s.get("openPriceRaw") or s.get("openPrice")),
        "vol": num(s.get("accumulatedTradingVolumeRaw") or s.get("accumulatedTradingVolume")),
        "val": kval(s.get("accumulatedTradingValue")),
        "limit_code": (s.get("compareToPreviousPrice") or {}).get("code"),
        "halt": (s.get("tradeStopType") or {}).get("code") not in (None, "1"),
        "market": "코스닥" if (s.get("stockExchangeType") or {}).get("name", "").upper().startswith("KOSDAQ") or s.get("sosok") == "1" else "코스피",
    }


def fetch_up_lists(min_ret: float = 0.08) -> dict[str, dict]:
    out = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        for page in (1, 2, 3, 4):
            j = http_json(f"{NAVER}/stocks/up/{mkt}?page={page}&pageSize=100")
            stocks = (j or {}).get("stocks") or []
            if not stocks:
                break
            stop = False
            for s in stocks:
                if s.get("stockEndType") not in (None, "stock"):
                    continue
                r = _row(s)
                if not r:
                    continue
                if r["ret"] < min_ret:
                    stop = True
                    break
                r["market"] = "코스피" if mkt == "KOSPI" else "코스닥"
                out[r["code"]] = r
            if stop or len(stocks) < 100:
                break
    return out


def fetch_one(code: str) -> dict | None:
    j = http_json(f"{NAVER}/stock/{code}/basic")
    if not j:
        return None
    r = _row(j)
    if r:
        r["code"] = code
        r["val"] = r["val"] or (num(j.get("accumulatedTradingValueKrw")) if isinstance(j, dict) else None)
    return r


def fetch_index() -> dict:
    out = {}
    for k in ("KOSPI", "KOSDAQ"):
        j = http_json(f"{NAVER}/index/{k}/basic")
        if j:
            out[k] = {"close": num(j.get("closePrice")), "r1": (num(j.get("fluctuationsRatio")) or 0) / 100}
    return out


def main():
    t = now_kst()
    day = t.strftime("%Y%m%d")
    hm = t.strftime("%H%M")
    force = "--force" in sys.argv
    latest = json.loads((OUT / "latest.json").read_text()) if (OUT / "latest.json").exists() else {}
    if not force:
        if t.weekday() >= 5 or hm < "0855" or hm > "1545":
            print(f"[live] 장외 시간({hm}) → 건너뜀")
            return
        if latest.get("target_date") and latest["target_date"] != day:
            print(f"[live] 오늘({day})은 예측 대상일({latest.get('target_date')})이 아님(휴일?) → 건너뜀")
            return
    scores = {}
    if (OUT / "scores.json").exists():
        for row in json.loads((OUT / "scores.json").read_text()).get("rows", []):
            scores[row[0]] = {"name": row[1], "p": row[4]}
    morning = latest.get("up", [])[:25]
    mcodes = [m["code"] for m in morning]

    codes = list(dict.fromkeys(list(scores) + mcodes))
    allq = fetch_all(codes) if codes else {}
    if not allq:  # 묶음 조회 실패 시 상승 목록으로 대체
        allq = fetch_up_lists()
        allq.update({r["code"]: r for r in pmap(fetch_one, mcodes, workers=8, label="morning") if r})
    ups = {c: r for c, r in allq.items() if r["ret"] >= 0.08 or r.get("limit_code") == "1"}
    singles = allq
    idx = fetch_index()
    n_total = len(allq)
    n_up = sum(1 for r in allq.values() if r["ret"] > 0)

    def enrich(r: dict, m: dict | None) -> dict:
        prev = r["prev"]
        ul = (m or {}).get("up_limit") or up_limit(prev)
        at = r.get("limit_code") == "1" or r["price"] >= ul - 1e-6
        touched = bool(r.get("high") and r["high"] >= ul - 1e-6)
        hi_ret = (r["high"] / prev - 1) if r.get("high") else None
        return {
            "code": r["code"], "name": r.get("name") or (m or {}).get("name") or scores.get(r["code"], {}).get("name"),
            "market": (m or {}).get("market") or r.get("market"),
            "price": r["price"], "ret": round(r["ret"], 4), "hi_ret": round(hi_ret, 4) if hi_ret is not None else None,
            "to_limit": round(ul / r["price"] - 1, 4), "up_limit": ul, "val": r.get("val"), "vol": r.get("vol"),
            "open": r.get("open"), "gap": round(r["open"] / prev - 1, 4) if r.get("open") else None,
            "o_ret": round(r["price"] / r["open"] - 1, 4) if r.get("open") else None,
            "status": "상한가" if at else "도달 후 이탈" if touched else "상승" if r["ret"] > 0 else "하락",
            "p": (m or {}).get("p") if m else scores.get(r["code"], {}).get("p"),
            "morning_rank": (mcodes.index(r["code"]) + 1) if r["code"] in mcodes else None,
            "pref": (m or {}).get("pref", False),
        }

    morning_rows = []
    for m in morning:
        r = singles.get(m["code"]) or ups.get(m["code"])
        if r:
            morning_rows.append(enrich(r, m))
    mset = set(mcodes)
    market_rows = [enrich(r, next((m for m in morning if m["code"] == c), None)) for c, r in ups.items()]
    for mr in morning_rows:  # 아침 후보가 목록에 없더라도 시장 표에 포함
        if mr["code"] not in ups and mr["ret"] >= 0.08:
            market_rows.append(mr)
    market_rows.sort(key=lambda x: (x["status"] != "상한가", -x["ret"], -(x["p"] or 0)))
    at_limit = [x for x in market_rows if x["status"] == "상한가"]
    near = [x for x in market_rows if x["status"] != "상한가" and x["ret"] >= 0.10][:60]

    doc = {
        "at": t.strftime("%Y-%m-%d %H:%M"), "date": day, "as_of": latest.get("as_of"), "target_date": latest.get("target_date"),
        "market": idx, "n_limit": len(at_limit), "n_up10": sum(1 for x in market_rows if x["ret"] >= 0.10),
        "n_total": n_total, "up_ratio": (n_up / n_total) if n_total else None,
        "morning": morning_rows, "limit": at_limit, "near": near,
        "lock_rate": {"all": 0.59, "morning": 0.68},  # 일봉 2024~: 장중 상한가 터치 종목 중 상한가 마감 비율(아침 상위25는 68%)
        "closed": hm >= "1530",
    }
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "live.json").write_text(json.dumps(doc, ensure_ascii=False))
    # 스냅샷 누적(장중 모델 학습용): 시각, 종목, 등락, 고가 등락, 거래대금, 아침 확률, 상태
    snap_dir = LIVE / "snap"
    snap_dir.mkdir(exist_ok=True)
    f = snap_dir / f"{day}.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "code", "ret", "hi_ret", "val", "p", "status", "morning_rank"])
        for x in market_rows + [m for m in morning_rows if m["code"] not in {y["code"] for y in market_rows}]:
            w.writerow([hm, x["code"], x["ret"], x["hi_ret"], x["val"], x["p"], x["status"], x["morning_rank"]])
    print(f"[live] {doc['at']} 상한가 {len(at_limit)} 근접 {len(near)} 아침후보 {len(morning_rows)}/{len(morning)} ups={len(ups)}")


if __name__ == "__main__":
    main()
