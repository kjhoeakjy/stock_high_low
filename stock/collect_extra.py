"""키 없이 되는 추가 수집: 시장경보·거래정지·관리종목 목록, 밤사이(애프터마켓·프리마켓) 가격, 신규상장 일정.

출력(DATA):
  status_today.json   code → {alert, halt, managed, up_limit, dn_limit, shares}
  overnight.json      code → {over_price, over_ret, over_vol, over_status, up_limit, dn_limit}
  ipo.json            최근 상장 목록
캐시(CACHE): status_history.parquet, overnight_history.parquet (날짜별 누적 → 추후 학습 변수)
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

from stock.common import CACHE, DATA, http_json, num, pmap, today_kst

PC = {"Referer": "https://stock.naver.com/"}
LIST = "https://stock.naver.com/api/domestic/market/stock/default?tradeType=KRX&marketType=ALL&orderType={order}&startIdx={idx}&pageSize=100"


def _list(order: str, extra: str = "") -> list[dict]:
    rows = []
    for idx in range(0, 4000, 100):
        j = http_json(LIST.format(order=order, idx=idx) + extra, headers=PC)
        if not isinstance(j, list) or not j:
            break
        rows += j
        if len(j) < 100:
            break
    return rows


def fetch_status() -> dict:
    out = {}
    for at in ("01", "02", "03"):
        for r in _list("marketAlertType", f"&alertType={at}"):
            if r.get("marketAlertType") == at:
                out.setdefault(r["itemcode"], {})["alert"] = int(at)
    for r in _list("tradeStopYn"):
        if r.get("tradeStopYn") == "Y":
            out.setdefault(r["itemcode"], {})["halt"] = 1
    for r in _list("statusTag"):
        if r.get("manageStatusGb") not in (None, "0", ""):
            out.setdefault(r["itemcode"], {})["managed"] = int(r.get("manageStatusGb") or 0)
    # 공통 필드(제한가·상장주식수)
    for order in ("marketAlertType&alertType=01", "tradeStopYn", "statusTag"):
        pass
    print(f"[status] alert={sum(1 for v in out.values() if v.get('alert'))} halt={sum(1 for v in out.values() if v.get('halt'))} managed={sum(1 for v in out.values() if v.get('managed'))}")
    return out


def _kval(s):
    """'699억' '1.2조' → 원."""
    import re
    if s is None:
        return None
    m = re.match(r"^(-?[\d.,]+)\s*([조억만])?", str(s).strip())
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * {"조": 1e12, "억": 1e8, "만": 1e4}.get(m.group(2), 1.0)


def fetch_overnight(codes: list[str]) -> dict:
    """polling/domestic/stock: 100개씩 묶음 조회(JSON). 정규장 종가·시가·고저·거래대금, 애프터마켓(overMarketPriceInfo)."""
    out = {}
    chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]

    def one(chunk):
        j = http_json("https://polling.finance.naver.com/api/realtime/domestic/stock/" + ",".join(chunk))
        res = {}
        for d in (j or {}).get("datas") or []:
            ov = d.get("overMarketPriceInfo") or {}
            close = num(d.get("closePrice"))
            op = num(ov.get("overPrice"))
            res[d["itemCode"]] = {
                "close": close, "open": num(d.get("openPrice")), "high": num(d.get("highPrice")), "low": num(d.get("lowPrice")),
                "vol": num(d.get("accumulatedTradingVolume")), "val": _kval(d.get("accumulatedTradingValue")),
                "chg_code": (d.get("compareToPreviousPrice") or {}).get("code"),
                "halt": (d.get("tradeStopType") or {}).get("code") not in (None, "1"),
                "over_price": op, "over_ret": (op / close - 1) if (op and close) else None,
                "over_status": ov.get("overMarketStatus"), "over_session": ov.get("tradingSessionType"),
                "over_open": num(ov.get("openPrice")), "over_high": num(ov.get("highPrice")), "over_low": num(ov.get("lowPrice")),
                "over_vol": num(ov.get("accumulatedTradingVolume")), "over_val": _kval(ov.get("accumulatedTradingValue")),
                "over_at": ov.get("localTradedAt"),
            }
        return res

    for r in pmap(one, chunks, workers=6, label="overnight"):
        if r:
            out.update(r)
    n_over = sum(1 for v in out.values() if v.get("over_ret") is not None)
    print(f"[overnight] {len(out)} stocks, {n_over} with after/pre-market price")
    return out


def fetch_ipo() -> list[dict]:
    out = []
    for tp in ("LISTING", "SUBSCRIPTION"):
        j = http_json(f"https://stock.naver.com/api/domestic/market/ipo/progress?IpoProgressType={tp}&startIdx=0&pageSize=100", headers=PC)
        if not j:
            continue
        for k in ("listingList", "subscriptionList", "subscriptionCompleteList"):
            for r in j.get(k) or []:
                out.append({"name": r.get("compName"), "market": r.get("marketType"), "listing": r.get("lcalDate"),
                            "price": num(r.get("fixPubPrice")), "ratio": num(r.get("fnlCmptRatio")),
                            "sub_start": r.get("poStartDate"), "status": r.get("ipoStatus")})
    print(f"[ipo] {len(out)}")
    return out


def _append_history(name: str, day: str, rows: list[dict]):
    cache = CACHE / name
    new = pd.DataFrame(rows)
    if new.empty:
        return
    new.insert(0, "date", day)
    old = pd.read_parquet(cache) if cache.exists() else None
    df = pd.concat([old[old["date"] != day], new], ignore_index=True) if old is not None else new
    df.to_parquet(cache, index=False)


def main(argv):
    t0 = time.time()
    day = today_kst()
    uni = json.loads((DATA / "universe.json").read_text())
    codes = [u["code"] for u in uni]
    status = fetch_status()
    (DATA / "status_today.json").write_text(json.dumps(status, ensure_ascii=False))
    _append_history("status_history.parquet", day, [{"code": c, **v} for c, v in status.items()])
    over = fetch_overnight(codes)
    (DATA / "overnight.json").write_text(json.dumps(over, ensure_ascii=False))
    _append_history("overnight_history.parquet", day,
                    [{"code": c, **r} for c, r in over.items()])
    (DATA / "ipo.json").write_text(json.dumps(fetch_ipo(), ensure_ascii=False))
    print(f"[collect_extra] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
