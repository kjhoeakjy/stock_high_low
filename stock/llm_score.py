"""LLM 뉴스·공시 읽기 점수(소급 채점): 1단계 모형 검증 캐시의 일별 상위 후보에 대해, 예측일 t 기준 직전 3일~익일 07:00의
뉴스·공시 제목을 Claude가 읽고 방향(-3~+3)·사건 분류·신선도·위험 신호를 구조화 점수로 산출한다. 2단계 재순위·메타 라벨링 입력용.

  python -m stock.llm_score [--top=100] [--days=400] [--batch=20] [--workers=6] [--max_req=3000] [--body=1] [--out=파일명]
출력: stock/data/llm_scores.parquet (date, code, has, dir, strength, event, novelty, risk, note, model)
  --body=1: collect_body가 받은 기사 본문 앞부분(최근 4건×500자)을 제목 뒤에 붙여 판독, 출력은 llm_scores_body.parquet
누출 방지: 제목 시각 < next_date 07:00 만 사용(아침 07:20 수집과 동일 기준).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from stock.common import DATA

OUT = DATA / "llm_scores.parquet"
MODEL = os.environ.get("STOCK_LLM_SCORE_MODEL", "claude-haiku-5-5")
TAG = MODEL   # 기록용 모델 표시(본문 모드는 +body)
MIN_VAL = 1e9
SYSTEM = """당신은 한국 주식시장 뉴스·공시 판독기다. 각 종목의 최근 뉴스·공시 제목만 보고, 다음 거래일 장 시작 직후 매수해 당일 매도할 때의 기대 방향을 판단한다.
규칙: 제목에 없는 사실을 지어내지 않는다. 출력은 JSON 배열만.
각 원소: {"code": 종목코드, "dir": -3~3 정수(강한 악재 -3, 중립 0, 강한 호재 3), "event": 사건 분류(공급계약|무상증자|유상증자|CB|최대주주변경|합병인수|임상승인|실적|테마수혜|정책|거래정지|관리경고|횡령소송|기타|없음), "novelty": 0~2(0 새 사건, 1 후속 보도, 2 이미 알려진 재탕), "risk": 0|1(경고·정지·소송·감자·대규모 증자 등 위험 신호), "note": 20자 이내 명사형 근거}"""


def _client():
    import anthropic
    return anthropic.Anthropic()


def targets(top: int, days: int) -> pd.DataFrame:
    oos = pd.read_parquet(DATA / "oos_y_oc10.parquet")
    oos = oos[(oos["pref"] == 0) & (oos["val"] >= MIN_VAL)].copy()
    oos["date"], oos["next_date"] = oos["date"].astype(str), oos["next_date"].astype(str)
    cutoff = (dt.datetime.utcnow() + dt.timedelta(hours=9) - dt.timedelta(days=days)).strftime("%Y%m%d")
    oos = oos[oos["date"] >= cutoff]
    t = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(top)
    return t[["date", "next_date", "code", "p", "ret1"]].reset_index(drop=True)


BODY_CHARS = 500   # 본문 모드: 기사당 앞부분 글자 수
BODY_PER = 4       # 본문 모드: 대상당 최근 기사 수


def gather(t: pd.DataFrame, body: bool = False) -> dict[tuple[str, str], list[str]]:
    """(date, code) → 제목 목록. 공시는 전 기간, 뉴스는 수집된 구간만. body=True면 news_body.parquet의 본문 앞부분을 최근 기사 BODY_PER건에 붙인다."""
    news = pd.read_parquet(DATA / "news.parquet", columns=["code", "id", "dt", "title"])
    disc = pd.read_parquet(DATA / "disclosures.parquet", columns=["code", "dt", "title"])
    bodies = {}
    if body and (DATA / "news_body.parquet").exists():
        b = pd.read_parquet(DATA / "news_body.parquet")
        bodies = dict(zip(b["id"].astype(str), b["body"].fillna("")))
        print(f"[llm_score] 본문 {len(bodies)}건 사용", flush=True)
    news["body"] = news["id"].astype(str).map(bodies).fillna("") if bodies else ""
    disc["body"] = ""
    news["kind"], disc["kind"] = "뉴스", "공시"
    allt = pd.concat([disc, news], ignore_index=True)
    allt = allt[allt["code"].isin(set(t["code"]))]
    allt["title"] = allt["title"].str.replace("&quot;", '"').str.replace("&amp;", "&").str.strip()
    by_code = {c: g.sort_values("dt") for c, g in allt.groupby("code")}
    out = {}
    for r in t.itertuples(index=False):
        g = by_code.get(r.code)
        if g is None:
            out[(r.date, r.code)] = []
            continue
        lo = (dt.datetime.strptime(r.date, "%Y%m%d") - dt.timedelta(days=3)).strftime("%Y%m%d") + "0000"
        hi = r.next_date + "0700"
        s = g[(g["dt"] >= lo) & (g["dt"] < hi)]
        seen, lines = set(), []
        for x in s.itertuples(index=False):
            key = x.title[:30]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {x.kind} {x.dt[4:8]} {x.dt[8:12]} {x.title[:80]}")
        lines = lines[-14:]
        if body:
            # 최근 뉴스 BODY_PER건의 본문 앞부분(제목 뒤에 들여쓰기로)
            nb = s[(s["kind"] == "뉴스") & (s["body"].str.len() > 50)].tail(BODY_PER)
            for x in nb.itertuples(index=False):
                lines.append(f"  · 본문({x.dt[4:8]}): {x.body[:BODY_CHARS].replace(chr(10), ' ')}")
        out[(r.date, r.code)] = lines
    return out


def ask(cli, batch: list[tuple[str, str, float, float, list[str]]]) -> list[dict]:
    blocks = [f"[{code}] (전일 {ret1*100:+.1f}%, 모델 확률 {p*100:.0f}%)\n" + "\n".join(lines) for _, code, p, ret1, lines in batch]
    prompt = f"다음 {len(batch)}개 종목을 판독하라.\n\n" + "\n\n".join(blocks)
    for i in range(3):
        try:
            r = cli.messages.create(model=MODEL, max_tokens=3000, system=SYSTEM, messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") or "" for b in r.content)
            m = re.search(r"\[.*\]", text, re.S)
            arr = json.loads(m.group(0)) if m else []
            by = {str(a.get("code")).zfill(6): a for a in arr if isinstance(a, dict)}
            out = []
            for date, code, p, ret1, lines in batch:
                a = by.get(code)
                if a is None:
                    continue
                out.append({"date": date, "code": code, "has": 1, "dir": max(-3, min(3, int(a.get("dir") or 0))), "event": str(a.get("event") or "기타")[:12],
                            "novelty": max(0, min(2, int(a.get("novelty") or 0))), "risk": int(bool(a.get("risk"))), "note": str(a.get("note") or "")[:30], "model": TAG})
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            print(f"[llm_score] 재시도 {i+1} · {str(e)[:120]}", flush=True)
            time.sleep(3 * (i + 1))
    return []


def main(argv):
    arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    top, days, bs, workers, max_req = int(arg("top", 100)), int(arg("days", 400)), int(arg("batch", 20)), int(arg("workers", 6)), int(arg("max_req", 3000))
    body = arg("body", "0") == "1"
    global OUT, TAG
    OUT = DATA / arg("out", "llm_scores_body.parquet" if body else "llm_scores.parquet")
    if body:
        bs = min(bs, 8)   # 본문 포함 시 요청당 종목 수 축소
        TAG = MODEL + "+body"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[llm_score] ANTHROPIC_API_KEY 없음")
        return
    t = targets(top, days)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["date", "code"])
    done = set(zip(old["date"].astype(str), old["code"].astype(str)))
    titles = gather(t, body)
    rows, todo = [], []
    for r in t.itertuples(index=False):
        if (r.date, r.code) in done:
            continue
        lines = titles[(r.date, r.code)]
        if not lines:
            rows.append({"date": r.date, "code": r.code, "has": 0, "dir": 0, "event": "없음", "novelty": 2, "risk": 0, "note": "", "model": TAG})
        else:
            todo.append((r.date, r.code, float(r.p), float(r.ret1 or 0), lines))
    print(f"[llm_score] 대상 {len(t)} · 완료 {len(done)} · 제목 없음 {len(rows)} · 판독 대상 {len(todo)} → 요청 {min(max_req, -(-len(todo)//bs))}건")
    todo.sort(reverse=True)  # 최근부터
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)][:max_req]
    cli = _client()
    t0, n_ok = time.time(), 0

    def flush():
        nonlocal old
        if rows:
            new = pd.DataFrame(rows)
            old = pd.concat([old, new], ignore_index=True) if len(old) else new
            old = old.drop_duplicates(["date", "code"], keep="last")
            old.to_parquet(OUT, index=False)

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(ask, cli, b): b for b in batches}
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            rows.extend(res)
            n_ok += len(res)
            if k % 50 == 0:
                flush()
                print(f"[llm_score] {k}/{len(batches)} 요청 · 채점 {n_ok} · {time.time()-t0:.0f}s", flush=True)
    flush()
    print(f"[llm_score] 완료 · 요청 {len(batches)} · 채점 {n_ok} · 누적 {len(old)} · {time.time()-t0:.0f}s")
    if len(old):
        s = old[old["has"] == 1]
        print(f"[llm_score] 분포 dir: {s['dir'].value_counts().sort_index().to_dict()} risk: {s['risk'].mean():.3f} event: {s['event'].value_counts().head(8).to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1:])
