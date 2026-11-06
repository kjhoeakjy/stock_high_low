"""비정형 데이터 수집: 종목별 공시 제목(전 기간)·뉴스 제목(증분 캐시). 네이버 금융 API.

출력(DATA):
  disclosures.parquet  code, dt(YYYYMMDDHHMM), title
  news.parquet         code, dt(YYYYMMDDHHMM), title, office
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import pandas as pd

from stock.common import CACHE, DATA, http_json, pmap

NAVER = "https://m.stock.naver.com/api"
NEWS_PAGES_FULL = 60   # 첫 수집: 종목당 최대 60쪽(1,200건)
NEWS_PAGES_INC = 6     # 매일: 최신 6쪽까지만, 이미 있는 기사 만나면 중단


def fetch_disclosures_one(code: str) -> list[dict]:
    j = http_json(f"{NAVER}/stock/{code}/disclosure?pageSize=500&page=1")
    if not isinstance(j, list):
        return []
    out = []
    for r in j:
        t = (r.get("datetime") or "").replace("-", "").replace("T", "").replace(":", "")[:12]
        if len(t) == 12:
            out.append({"code": code, "dt": t, "title": (r.get("title") or "").strip()})
    return out


def fetch_disclosures(codes: list[str]) -> pd.DataFrame:
    parts = pmap(fetch_disclosures_one, codes, workers=16, label="disclosures")
    df = pd.DataFrame([r for p in parts if p for r in p], columns=["code", "dt", "title"])
    df = df.drop_duplicates().sort_values(["code", "dt"]).reset_index(drop=True)
    print(f"[disclosures] {len(df)} rows, {df['code'].nunique()} codes, {df['dt'].min() if len(df) else ''}~")
    return df


def fetch_news_one(args) -> list[dict]:
    code, known_ids, max_pages = args
    out = []
    for page in range(1, max_pages + 1):
        j = http_json(f"{NAVER}/news/stock/{code}?pageSize=20&page={page}")
        if not isinstance(j, list) or not j:
            break
        hit_known = False
        n_items = 0
        for grp in j:
            for it in grp.get("items", []):
                n_items += 1
                nid = it.get("id") or f'{it.get("officeId")}{it.get("articleId")}'
                if nid in known_ids:
                    hit_known = True
                    continue
                out.append({"code": code, "id": nid, "dt": str(it.get("datetime") or "")[:12],
                            "title": (it.get("titleFull") or it.get("title") or "").strip(),
                            "office": it.get("officeName") or ""})
        if hit_known or n_items < 20:
            break
    return out


def fetch_news(codes: list[str]) -> pd.DataFrame:
    cache = CACHE / "news.parquet"
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["code", "id", "dt", "title", "office"])
    known = old.groupby("code")["id"].agg(set).to_dict() if len(old) else {}
    jobs = [(c, known.get(c, set()), NEWS_PAGES_INC if c in known else NEWS_PAGES_FULL) for c in codes]
    parts = pmap(fetch_news_one, jobs, workers=16, label="news")
    new = pd.DataFrame([r for p in parts if p for r in p], columns=old.columns)
    df = pd.concat([old, new], ignore_index=True) if len(new) else old
    df = df.drop_duplicates(["code", "id"], keep="last")
    # 2년 넘은 기사는 캐시에서 제거(크기 관리)
    cutoff = (dt.datetime.utcnow() - dt.timedelta(days=760)).strftime("%Y%m%d%H%M")
    df = df[df["dt"] >= cutoff].sort_values(["code", "dt"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"[news] cache {len(old)} -> {len(df)} rows (+{len(new)})")
    return df


def main(argv):
    t0 = time.time()
    uni = json.loads((DATA / "universe.json").read_text())
    codes = [u["code"] for u in uni]
    if "--quick" in argv:
        codes = codes[:100]
    if "--no-disclosures" not in argv:
        fetch_disclosures(codes).to_parquet(DATA / "disclosures.parquet", index=False)
    if "--no-news" not in argv:
        fetch_news(codes).to_parquet(DATA / "news.parquet", index=False)
    print(f"[collect_text] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
