"""데이터 수집: 네이버 금융(전 종목 시세·수급·업종·테마·공시), 야후(해외지수·환율), DART(당일 공시).

출력(DATA 디렉터리):
  universe.json   종목 목록 + 현재 스냅샷(시총, 주식수 추정 등)
  prices.parquet  종목별 일봉(시가 고가 저가 종가 거래량 외국인소진율), 2019~
  flows.parquet   종목별 투자자 순매수(외국인·기관·개인), 캐시 기반 증분
  groups.json     업종·테마 구성 종목
  market.parquet  KOSPI·KOSDAQ·해외지수·환율 일봉
  dart_today.json 최근 공시 목록(전일 저녁~당일 아침)
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import re
import sys
import time

import pandas as pd

from stock.common import CACHE, DATA, http, http_json, num, pmap, today_kst

START = "20190101"
FLOW_START = "20220101"
NAVER = "https://m.stock.naver.com/api"


# ---------------------------------------------------------------- universe
def fetch_universe() -> list[dict]:
    rows = []
    for mkt in ("KOSPI", "KOSDAQ"):
        page = 1
        while True:
            j = http_json(f"{NAVER}/stocks/marketValue/{mkt}?page={page}&pageSize=100")
            if not j or not j.get("stocks"):
                break
            for s in j["stocks"]:
                if s.get("stockEndType") != "stock":
                    continue
                close = num(s.get("closePriceRaw"))
                mcap = num(s.get("marketValueRaw"))
                rows.append({
                    "code": s["itemCode"], "name": s["stockName"], "market": mkt,
                    "close": close, "mcap": mcap,
                    "shares": (mcap / close) if (mcap and close) else None,
                    "status": s.get("tradableStatusCode"),
                    "halt": (s.get("tradeStopType") or {}).get("name"),
                })
            if page * 100 >= int(j.get("totalCount", 0)):
                break
            page += 1
    seen=set(); rows=[r for r in rows if not (r["code"] in seen or seen.add(r["code"]))]
    print(f"[universe] {len(rows)} stocks")
    return rows


# ---------------------------------------------------------------- prices
def _parse_sise(text: str) -> list[list]:
    # pseudo-JSON: header uses single quotes; rows are JSON arrays
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        rows = re.findall(r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+),\s*([\d.]+)\]', text)
        return [["h"]] + [[d, float(o), float(h), float(l), float(c), int(v), float(f)] for d, o, h, l, c, v, f in rows]


def fetch_prices_one(code: str, start: str = START) -> pd.DataFrame | None:
    url = (f"https://api.finance.naver.com/siseJson.naver?symbol={code}&requestType=1"
           f"&startTime={start}&endTime=20991231&timeframe=day")
    t = http(url)
    if not t:
        return None
    rows = _parse_sise(t)
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows[1:], columns=["date", "open", "high", "low", "close", "volume", "frate"])
    df.insert(0, "code", code)
    return df


def fetch_prices(codes: list[str]) -> pd.DataFrame:
    parts = pmap(fetch_prices_one, codes, workers=16, label="prices")
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    df["date"] = df["date"].astype(str)
    for c in ("open", "high", "low", "close", "frate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df = df[(df["close"] > 0)].drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- flows (investor net buy)
def _trend_page(code: str, bizdate: str | None) -> list[dict]:
    url = f"{NAVER}/stock/{code}/trend?pageSize=60"
    if bizdate:
        url += f"&bizdate={bizdate}"
    j = http_json(url)
    if not isinstance(j, list):
        return []
    out = []
    for r in j:
        out.append({
            "code": code, "date": str(r.get("bizdate")),
            "frgn": num(r.get("foreignerPureBuyQuant"), 0.0),
            "inst": num(r.get("organPureBuyQuant"), 0.0),
            "indiv": num(r.get("individualPureBuyQuant"), 0.0),
            "fhold": num(r.get("foreignerHoldRatio")),
        })
    return out


def fetch_flows_one(args) -> list[dict]:
    code, since = args  # since: fetch rows with date > since
    rows, cursor, seen = [], None, set()
    for _ in range(40):
        page = _trend_page(code, cursor)
        if not page:
            break
        new = [r for r in page if r["date"] not in seen]
        if not new:
            break
        rows += new
        seen.update(r["date"] for r in new)
        oldest = min(r["date"] for r in new)
        if oldest <= since or len(page) < 60:
            break
        d = dt.datetime.strptime(oldest, "%Y%m%d") - dt.timedelta(days=1)
        cursor = d.strftime("%Y%m%d")
    return [r for r in rows if r["date"] > since]


def fetch_flows(codes: list[str]) -> pd.DataFrame:
    cache = CACHE / "flows.parquet"
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["code", "date", "frgn", "inst", "indiv", "fhold"])
    last = old.groupby("code")["date"].max().to_dict() if len(old) else {}
    since_default = (dt.datetime.strptime(FLOW_START, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")
    jobs = [(c, last.get(c, since_default)) for c in codes]
    parts = pmap(fetch_flows_one, jobs, workers=24, label="flows")
    new = pd.DataFrame([r for p in parts if p for r in p])
    df = pd.concat([old, new], ignore_index=True) if len(new) else old
    df = df.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"[flows] cache {len(old)} -> {len(df)} rows (+{len(new)})")
    return df


# ---------------------------------------------------------------- groups: industry / theme
def _group_members(kind: str, no: int) -> list[str]:
    codes = []
    for page in range(1, 6):
        j = http_json(f"{NAVER}/stocks/{kind}/{no}?page={page}&pageSize=100")
        if not j:
            break
        stocks = j.get("stocks") or j.get("groupStocks") or []
        codes += [s["itemCode"] for s in stocks if s.get("stockEndType", "stock") == "stock"]
        if len(stocks) < 100:
            break
    return codes


def fetch_groups() -> dict:
    out = {}
    for kind in ("industry", "theme"):
        groups = []
        page = 1
        while True:
            j = http_json(f"{NAVER}/stocks/{kind}?page={page}&pageSize=100")
            if not j or not j.get("groups"):
                break
            groups += j["groups"]
            if page * 100 >= int(j.get("totalCount", 0)):
                break
            page += 1
        members = pmap(lambda g: _group_members(kind, g["no"]), groups, workers=8, label=kind)
        out[kind] = [{"no": g["no"], "name": g["name"], "codes": m or []} for g, m in zip(groups, members)]
        print(f"[groups] {kind}: {len(groups)} groups, {sum(len(x['codes']) for x in out[kind])} memberships")
    return out


# ---------------------------------------------------------------- market: indices / fx
def _fchart(symbol: str, count: int = 3000) -> pd.DataFrame | None:
    t = http(f"https://fchart.stock.naver.com/sise.nhn?symbol={symbol}&timeframe=day&count={count}&requestType=0",
             encoding="euc-kr")
    if not t:
        return None
    rows = re.findall(r'data="(\d{8})\|([\d.]+)\|([\d.]+)\|([\d.]+)\|([\d.]+)\|(\d+)"', t)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df.insert(0, "symbol", symbol)
    return df


def _yahoo(symbol: str, name: str) -> pd.DataFrame | None:
    j = http_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d",
                  headers={"Referer": "https://finance.yahoo.com/"})
    try:
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        ts = r["timestamp"]
    except Exception:  # noqa: BLE001
        print(f"[yahoo] fail {symbol}")
        return None
    df = pd.DataFrame({"ts": ts, "open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]})
    off = r["meta"].get("gmtoffset", 0)
    df["date"] = pd.to_datetime(df["ts"] + off, unit="s").dt.strftime("%Y%m%d")
    df = df.dropna(subset=["close"]).drop(columns="ts")
    df.insert(0, "symbol", name)
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


YAHOO = {
    "%5EIXIC": "NASDAQ", "%5EGSPC": "SP500", "%5ESOX": "SOX", "%5EVIX": "VIX",
    "KRW%3DX": "USDKRW", "%5ERUT": "RUSSELL", "CL%3DF": "WTI", "GC%3DF": "GOLD",
    "%5ETNX": "US10Y", "BTC-USD": "BTC", "%5EN225": "NIKKEI", "000001.SS": "SHANGHAI",
    "%5EHSI": "HANGSENG", "EWY": "EWY", "NVDA": "NVDA",
}


def fetch_market() -> pd.DataFrame:
    parts = [_fchart("KOSPI"), _fchart("KOSDAQ")]
    parts += [_yahoo(s, n) for s, n in YAHOO.items()]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- DART recent disclosures
def fetch_dart(days: int = 2) -> list[dict]:
    out = []
    today = dt.datetime.strptime(today_kst(), "%Y%m%d")
    for d in range(days):
        day = (today - dt.timedelta(days=d)).strftime("%Y%m%d")
        for page in range(1, 15):
            t = http("https://dart.fss.or.kr/dsac001/search.ax",
                     data={"currentPage": page, "mdayCnt": 0, "selectDate": day, "textCrpCik": "",
                           "textCrpNm": "", "pageGrouping": "", "maxResults": 100},
                     headers={"Referer": "https://dart.fss.or.kr/dsac001/mainAll.do"})
            if not t:
                break
            rows = re.findall(
                r'<tr>\s*<td>\s*([\d:]+)\s*</td>.*?title="([^"]+?) 기업개황 새창"[^>]*>.*?'
                r'rcpNo=(\d+)"[^>]*title="([^"]+?) 공시뷰어 새창"[^>]*>(.*?)</a>', t, flags=re.S)
            if not rows:
                break
            for tm, corp, rcp, title, inner in rows:
                tag = re.findall(r'<span[^>]*class="txtCB"[^>]*>(.*?)</span>', inner)
                out.append({"date": day, "time": tm, "corp": corp.strip(), "rcp": rcp,
                            "title": title.strip(), "tag": tag[0] if tag else ""})
            if len(rows) < 100:
                break
    print(f"[dart] {len(out)} disclosures")
    return out


# ---------------------------------------------------------------- main
def main(argv):
    t0 = time.time()
    uni = fetch_universe()
    codes = [u["code"] for u in uni]
    if "--quick" in argv:
        codes = codes[:120]
    (DATA / "universe.json").write_text(json.dumps(uni, ensure_ascii=False))

    market = fetch_market()
    market.to_parquet(DATA / "market.parquet", index=False)
    print(f"[market] {market.groupby('symbol')['date'].agg(['min', 'max', 'count']).to_string()}")

    prices = fetch_prices(codes)
    prices.to_parquet(DATA / "prices.parquet", index=False)
    print(f"[prices] {len(prices)} rows, {prices['code'].nunique()} codes, {prices['date'].min()}~{prices['date'].max()}")

    if "--no-flows" not in argv:
        flows = fetch_flows(codes)
        flows.to_parquet(DATA / "flows.parquet", index=False)
        print(f"[flows] {len(flows)} rows, {flows['code'].nunique()} codes")

    groups = fetch_groups()
    (DATA / "groups.json").write_text(json.dumps(groups, ensure_ascii=False))

    dart = fetch_dart()
    (DATA / "dart_today.json").write_text(json.dumps(dart, ensure_ascii=False))
    print(f"[collect] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
