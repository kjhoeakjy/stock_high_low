"""뉴스 본문 수집(LLM 판독용): 1단계 검증 캐시의 일별 상위 후보에 대해 예측일 직전 3일~익일 07:00 기사 본문을 네이버 뉴스에서 받아 저장.

  python -m stock.collect_body [--top=100] [--days=400] [--per=4] [--max=12000] [--workers=6] [--budget=300]
출력: stock/data/news_body.parquet (id, body)  ※ id = officeId(3)+articleId(10), 본문은 앞 1,500자
요청 속도: 동시 6건 이하로 제한(24건에서는 61%가 시간 초과). 1회 실행 약 0.7건/초 → 12,000건에 5시간.
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
    top, days, per, mx, workers = int(arg("top", 100)), int(arg("days", 400)), int(arg("per", 4)), int(arg("max", 20000)), int(arg("workers", 6))
    t = targets(top, days)
    ids = article_ids(t, per)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["id", "body"])
    done = set(old.loc[old["body"].str.len() > 50, "id"].astype(str))   # 본문 못 받은 것(시간 초과)은 다음 실행에서 다시 시도
    todo = [i for i in ids if i not in done][:mx]
    print(f"[body] 대상 {len(t)} · 기사 {len(ids)} · 기존 {len(done)} · 이번 {len(todo)}", flush=True)
    if not todo:
        return
    budget = float(arg("budget", 300)) * 60   # 실행 시간 상한(초). 워크플로 제한 전에 저장하고 끝냄
    t0 = time.time()
    out, n_new, n_ok = old, 0, 0
    for k in range(0, len(todo), 500):   # 500건마다 저장: 중단·시간 초과에도 수집분 보존
        if time.time() - t0 > budget:
            print(f"[body] 시간 상한 도달 · 남은 {len(todo)-k}건은 다음 실행", flush=True)
            break
        rows = [r for r in pmap(fetch_one, todo[k:k + 500], workers=workers, label="body") if r]
        new = pd.DataFrame(rows, columns=["id", "body"])
        out = pd.concat([out, new], ignore_index=True).drop_duplicates("id", keep="last")
        tmp = OUT.with_suffix(".tmp"); out.to_parquet(tmp, index=False); tmp.replace(OUT)
        n_new += len(new); n_ok += int((new["body"].str.len() > 50).sum())
        print(f"[body] 저장 {n_new}/{len(todo)} · 본문 있음 {n_ok/max(n_new,1):.2f} · 누적 {len(out)} · {time.time()-t0:.0f}s", flush=True)
    print(f"[body] 완료 {n_new}건 · 본문 있음 {n_ok/max(n_new,1):.2f} · 누적 {len(out)} · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
