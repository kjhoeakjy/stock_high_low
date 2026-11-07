"""뉴스 본문 수집(LLM 판독용): 1단계 검증 캐시의 일별 상위 후보에 대해 예측일 직전 3일~익일 07:00 기사 본문을 네이버 뉴스에서 받아 저장.

  python -m stock.collect_body [--top=100] [--days=400] [--per=4] [--max=20000] [--workers=8]
출력: stock/data/news_body.parquet (id, body)  ※ id = officeId(3)+articleId(10), 본문은 앞 1,500자
"""
from __future__ import annotations

import datetime as dt
import html
import re
import sys
import time

import pandas as pd

from stock.common import DATA, http, pmap
from stock.llm_score import targets

OUT = DATA / "news_body.parquet"
MAXLEN = 1500


def article_ids(t: pd.DataFrame, per: int) -> list[str]:
    news = pd.read_parquet(DATA / "news.parquet", columns=["code", "id", "dt", "title"])
    news = news[news["code"].isin(set(t["code"]))]
    by_code = {c: g.sort_values("dt") for c, g in news.groupby("code")}
    ids: list[str] = []
    for r in t.itertuples(index=False):
        g = by_code.get(r.code)
        if g is None:
            continue
        lo = (dt.datetime.strptime(r.date, "%Y%m%d") - dt.timedelta(days=3)).strftime("%Y%m%d") + "0000"
        hi = r.next_date + "0700"
        s = g[(g["dt"] >= lo) & (g["dt"] < hi)].drop_duplicates("title")
        ids.extend(s["id"].tail(per).tolist())
    return list(dict.fromkeys(ids))


def fetch_one(nid: str) -> dict | None:
    url = f"https://n.news.naver.com/mnews/article/{nid[:3]}/{nid[3:]}"
    txt = http(url, headers={"Referer": "https://n.news.naver.com/"}, retries=2, timeout=15)
    if not txt:
        return {"id": nid, "body": ""}
    m = re.search(r'<article[^>]*id="dic_area"[^>]*>(.*?)</article>', txt, re.S)
    if not m:
        return {"id": nid, "body": ""}
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", m.group(1), flags=re.S)
    body = re.sub(r"<br\s*/?>|</p>|</div>", "\n", body)
    body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    body = re.sub(r"[ \t　]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    return {"id": nid, "body": body[:MAXLEN]}


def main(argv):
    arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    top, days, per, mx, workers = int(arg("top", 100)), int(arg("days", 400)), int(arg("per", 4)), int(arg("max", 20000)), int(arg("workers", 8))
    t = targets(top, days)
    ids = article_ids(t, per)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["id", "body"])
    done = set(old["id"].astype(str))
    todo = [i for i in ids if i not in done][:mx]
    print(f"[body] 대상 {len(t)} · 기사 {len(ids)} · 기존 {len(done)} · 이번 {len(todo)}", flush=True)
    if not todo:
        return
    t0 = time.time()
    rows = [r for r in pmap(fetch_one, todo, workers=workers, label="body") if r]
    new = pd.DataFrame(rows, columns=["id", "body"])
    out = pd.concat([old, new], ignore_index=True).drop_duplicates("id", keep="last")
    out.to_parquet(OUT, index=False)
    ok = (new["body"].str.len() > 50).mean() if len(new) else 0
    print(f"[body] 완료 {len(new)}건 · 본문 있음 {ok:.2f} · 누적 {len(out)} · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
