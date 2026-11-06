"""한투 일별 분봉 수집: 1단계 모형 검증 캐시(oos_y_oc10)의 일별 상위 후보에 대해 거래일 09:00~09:30 1분봉을 모은다.
용도: 09:00:30 실제 체결 가능가 기준 라벨·검증 재계산, 장초 변수(첫 1·5분 거래대금·변동폭).

  python -m stock.minute [--top=100] [--max_calls=20000] [--days=365] [--sleep=0.12]
출력: stock/data/minute_open.parquet(date,code,time,open,high,low,close,vol,amt_cum), stock/data/minute_done.json(완료 목록·실패)
장중(09:00~15:45 KST)에는 실행하지 않는다(실시간 러너와 토큰 공유 충돌 방지).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import pandas as pd

from stock.common import DATA

MIN_VAL = 1e9

OUT = DATA / "minute_open.parquet"
DONE = DATA / "minute_done.json"


def targets(top: int, days: int) -> list[tuple[str, str]]:
    oos = pd.read_parquet(DATA / "oos_y_oc10.parquet")
    oos = oos[(oos["pref"] == 0) & (oos["val"] >= MIN_VAL)]
    oos["date"] = oos["date"].astype(str)
    oos["next_date"] = oos["next_date"].astype(str)
    cutoff = (dt.datetime.utcnow() + dt.timedelta(hours=9) - dt.timedelta(days=days)).strftime("%Y%m%d")
    oos = oos[oos["next_date"] >= cutoff]
    t = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(top)
    pairs = list(dict.fromkeys(zip(t["next_date"], t["code"])))
    pairs.sort(reverse=True)  # 최근 날부터(보관 기간 만료 전에 과거 쪽은 다음 실행이 이어받음)
    return pairs


def main(argv):
    arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    top, max_calls, days, pause = int(arg("top", 100)), int(arg("max_calls", 20000)), int(arg("days", 365)), float(arg("sleep", 0.12))
    now = dt.datetime.utcnow() + dt.timedelta(hours=9)
    if now.weekday() < 5 and "0855" <= now.strftime("%H%M") <= "1550" and "--force" not in argv:
        print("[minute] 장중 실행 금지")
        return
    from stock.kis import KIS
    k = KIS()
    done = json.loads(DONE.read_text()) if DONE.exists() else {"done": [], "fail": {}}
    done_set = {tuple(x) for x in done["done"]}
    pairs = [p for p in targets(top, days) if p not in done_set]
    print(f"[minute] 대상 {len(pairs)} (완료 {len(done_set)}) 이번 실행 최대 {max_calls}회")
    rows, calls, t0 = [], 0, time.time()
    old = pd.read_parquet(OUT) if OUT.exists() else None

    def flush():
        nonlocal rows, old
        if rows:
            new = pd.DataFrame(rows)
            old = pd.concat([old, new], ignore_index=True) if old is not None else new
            old = old.drop_duplicates(["date", "code", "time"], keep="last")
            old.to_parquet(OUT, index=False)
            rows = []
        done["done"] = sorted(done_set)
        DONE.write_text(json.dumps(done))

    for date, code in pairs:
        if calls >= max_calls:
            break
        try:
            bars = k.minutes(code, date, "093000")
            calls += 1
            bars = [b for b in bars if "090000" <= (b["time"] or "") <= "093000"]
            if not bars:
                done["fail"][f"{date}:{code}"] = "no bars"
            rows.extend(bars)
            done_set.add((date, code))
            done["fail"].pop(f"{date}:{code}", None)
        except Exception as e:  # noqa: BLE001
            calls += 1
            msg = str(e)[:100]
            done["fail"][f"{date}:{code}"] = msg
            if "EGW00201" in msg or "초당" in msg:
                time.sleep(1.0)
            elif "http 4" in msg or "토큰" in msg or "token" in msg.lower():
                print(f"[minute] 중단 · {msg}")
                break
        if calls % 500 == 0:
            flush()
            print(f"[minute] {calls}회 · 남은 {len(pairs) - calls} · {time.time() - t0:.0f}s", flush=True)
        time.sleep(pause)
    flush()
    n = len(old) if old is not None else 0
    print(f"[minute] 완료 호출 {calls} · 누적 봉 {n} · 완료 {len(done_set)} · 실패 {len(done['fail'])} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
