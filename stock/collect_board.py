"""네이버 종목토론방 수집(자체 토론 API) → 종목·날짜별 집계(board.parquet) + 종목별 마지막 글 번호(board_state.json).

  python -m stock.collect_board                       # 증분: 마지막 글 이후 새 글만(일과용)
  python -m stock.collect_board --backfill --shard 0/6 --days 400 --max-pages 80 --out /tmp/b0   # 과거 채우기(분할 실행)
  python -m stock.collect_board --merge /tmp/b0 /tmp/b1 ...                                     # 분할 결과 합치기

글 하나는 writtenAt(KST) 기준 07:00 이전이면 전날 묶음에 넣는다(아침 07:20 학습 때까지 쌓인 밤사이 글이 당일 신호).
개인정보(작성자)는 저장하지 않고 집계만 남긴다.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from stock.common import DATA, http_json, pmap

API = "https://m.stock.naver.com/front-api/discussion/list"
HDR = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
       "Referer": "https://m.stock.naver.com/domestic/stock/005930/discussion"}
PAGE = 100
KW = {  # 커뮤니티 어휘(제목+본문 앞부분)
    "b_kw_limit": r"상한가|상따|점상", "b_kw_surge": r"급등|폭등|떡상|불기둥|가즈아|가자", "b_kw_crash": r"급락|폭락|떡락|하한가|물렸|손절",
    "b_kw_force": r"세력|작전|매집|털기|개미털|설거지", "b_kw_news": r"공시|뉴스|기사|호재|악재|발표|계약|수주",
    "b_kw_issue": r"유상증자|유증|전환사채|CB|감자|거래정지|상장폐지|관리종목", "b_kw_theme": r"테마|대장|관련주|수혜",
    "b_kw_buy": r"매수|줍줍|풀매수|추매|들어간다|탑승", "b_kw_sell": r"매도|익절|던져|탈출|정리",
}
COLS = ["b_n", "b_rec", "b_notrec", "b_view", "b_holder", "b_reply"] + list(KW)


def bucket(written: str) -> str:
    t = dt.datetime.fromisoformat(written[:19]) - dt.timedelta(hours=7)
    return t.strftime("%Y%m%d")


def fetch(code: str, since_id: int = 0, max_pages: int = 30, until: str | None = None) -> list[dict]:
    """최신부터 거슬러 올라가며 since_id보다 큰 글, until(YYYYMMDD 묶음) 이후 글만 모은다."""
    out, offset = [], None
    for _ in range(max_pages):
        url = f"{API}?itemCode={code}&discussionType=domesticStock&pageSize={PAGE}" + (f"&offset={offset}" if offset else "")
        j = http_json(url, headers=HDR, retries=3)
        posts = ((j or {}).get("result") or {}).get("posts") or []
        if not posts:
            break
        stop = False
        for p in posts:
            pid = int(p.get("id") or 0)
            if pid <= since_id or (until and bucket(p["writtenAt"]) < until):
                stop = True
                break
            out.append(p)
        if stop or len(posts) < PAGE:
            break
        offset = posts[-1].get("orderNo")
    return out


def aggregate(code: str, posts: list[dict]) -> pd.DataFrame:
    if not posts:
        return pd.DataFrame(columns=["code", "date"] + COLS)
    rows = []
    for p in posts:
        txt = f"{p.get('title') or ''} {(p.get('contentSwReplacedButImg') or '')[:300]}"
        r = {"code": code, "date": bucket(p["writtenAt"]), "b_n": 1,
             "b_rec": int(p.get("recommendCount") or 0), "b_notrec": int(p.get("notRecommendCount") or 0),
             "b_view": int(p.get("viewCount") or 0), "b_holder": int(bool(p.get("isHolderVerified"))),
             "b_reply": int((p.get("replyDepth") or 0) > 0)}
        for k, rx in KW.items():
            r[k] = int(bool(re.search(rx, txt)))
        rows.append(r)
    return pd.DataFrame(rows).groupby(["code", "date"], as_index=False)[COLS].sum()


def _one(args):
    code, since, max_pages, until = args
    posts = fetch(code, since, max_pages, until)
    last = max((int(p["id"]) for p in posts), default=since)
    return code, aggregate(code, posts), last, len(posts)


def _codes() -> list[str]:
    uni = json.loads((DATA / "universe.json").read_text())
    return [u["code"] for u in uni]


def _merge_counts(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([old, new], ignore_index=True) if old is not None and len(old) else new
    df = df.groupby(["code", "date"], as_index=False)[COLS].sum()
    for c in COLS:
        df[c] = df[c].astype("int32")
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def run_incremental(max_pages: int = 20, workers: int = 8) -> None:
    codes = _codes()
    path, spath = DATA / "board.parquet", DATA / "board_state.json"
    old = pd.read_parquet(path) if path.exists() else None
    state = json.loads(spath.read_text()) if spath.exists() else {}
    # 상태가 없는 종목(신규)은 최근 3일만
    since_default = (dt.datetime.now() + dt.timedelta(hours=9) - dt.timedelta(days=3)).strftime("%Y%m%d")
    jobs = [(c, int(state.get(c, 0)), max_pages, None if c in state else since_default) for c in codes]
    t0 = time.time()
    res = pmap(_one, jobs, workers=workers, label="board")
    parts, n_posts = [], 0
    for r in res:
        if not r:
            continue
        code, agg, last, n = r
        parts.append(agg)
        n_posts += n
        state[code] = max(int(state.get(code, 0)), last)
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["code", "date"] + COLS)
    # 같은 묶음 날짜의 기존 행과 합산(어제 저녁 이후 글이 오늘 아침 추가됨)
    df = _merge_counts(old, new)
    df.to_parquet(path, index=False)
    spath.write_text(json.dumps(state))
    print(f"[board] +{n_posts} posts, {len(df)} rows, {df['code'].nunique()} codes, {df['date'].min() if len(df) else ''}~{df['date'].max() if len(df) else ''} in {time.time()-t0:.0f}s")


def run_backfill(shard: str, days: int, max_pages: int, out: Path, workers: int = 6) -> None:
    k, n = (int(x) for x in shard.split("/"))
    codes = [c for i, c in enumerate(_codes()) if i % n == k]
    until = (dt.datetime.now() + dt.timedelta(hours=9) - dt.timedelta(days=days)).strftime("%Y%m%d")
    t0 = time.time()
    res = pmap(_one, [(c, 0, max_pages, until) for c in codes], workers=workers, label=f"board-{k}")
    parts, state, n_posts, capped = [], {}, 0, 0
    for r in res:
        if not r:
            continue
        code, agg, last, n = r
        parts.append(agg)
        state[code] = last
        n_posts += n
        capped += int(n >= max_pages * PAGE)
    df = _merge_counts(None, pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=["code", "date"] + COLS)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "board.parquet", index=False)
    (out / "board_state.json").write_text(json.dumps(state))
    print(f"[board-backfill {shard}] {len(codes)} codes, {n_posts} posts, {len(df)} rows, 페이지 한도 도달 {capped}종목, {time.time()-t0:.0f}s")


def run_merge(dirs: list[str]) -> None:
    parts, state = [], {}
    for d in dirs:
        p = Path(d)
        if (p / "board.parquet").exists():
            parts.append(pd.read_parquet(p / "board.parquet"))
        if (p / "board_state.json").exists():
            state.update(json.loads((p / "board_state.json").read_text()))
    path, spath = DATA / "board.parquet", DATA / "board_state.json"
    old = pd.read_parquet(path) if path.exists() else None
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["code", "date"] + COLS)
    if old is not None and len(old):
        # 과거 채우기 결과가 있는 (종목, 날짜)는 새 값으로 교체, 나머지는 기존 유지
        key = set(zip(new["code"], new["date"]))
        old = old[[(c, d) not in key for c, d in zip(old["code"], old["date"])]]
    df = _merge_counts(old, new)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    ost = json.loads(spath.read_text()) if spath.exists() else {}
    for c, v in state.items():
        ost[c] = max(int(ost.get(c, 0)), int(v))
    spath.write_text(json.dumps(ost))
    print(f"[board-merge] {len(df)} rows, {df['code'].nunique()} codes, {df['date'].min() if len(df) else ''}~{df['date'].max() if len(df) else ''}")


def main(argv: list[str]) -> None:
    def opt(name, default):
        for a in argv:
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
        return default
    if "--merge" in argv:
        run_merge([a for a in argv if not a.startswith("--")])
    elif "--backfill" in argv:
        run_backfill(opt("shard", "0/1"), int(opt("days", 400)), int(opt("max-pages", 80)), Path(opt("out", "/tmp/board")))
    else:
        run_incremental(int(opt("max-pages", 20)))


if __name__ == "__main__":
    main(sys.argv[1:])
