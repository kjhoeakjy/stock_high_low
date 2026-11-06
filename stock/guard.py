"""리스크 가드: 자동 매매의 손실 방어 장치. 주문 실행(live.py)과 비상 작업(guard.yml)이 공용.

원칙
  1. 원금 보호선: 계좌 평가액이 시작 자금의 50% 이하가 되면 전량 청산하고 자동 매매를 영구 정지(사용자가 resume 전까지 다음 날부터 거래 없음).
  2. 일일 한도: 당일 계좌 손실이 DAY_LOSS(8%)에 이르면 전량 청산, 당일 신규 매수 금지.
  3. 매도 불능 대비: 거래정지·하한가 잠김·미체결로 못 판 수량은 pending에 올려 두고 매 주기(30초) 호가가 살아나면 즉시 시장가 매도 재시도. 날이 바뀌어도 유지.
  4. 이월 보유 금지: 계획에 없는 보유 종목(전날 못 판 것 포함)은 장 시작 직후 전부 매도.
  5. 매수 보호: 매도1호가가 전일 종가 대비 +20% 이상이면 매수 생략(상한가 잠김 위험), 호가 잔량·전일 거래대금 1% 한도로 수량 제한, 미체결 매수는 60초 뒤 취소.
  6. 시스템 이상: 시세·계좌 조회 연속 실패, 러너 지각(09:20 이후 시작)이면 당일 매수 금지. 조회 불능 상태에서 보유가 있으면 청산 시도.
  7. 비상 정지: TRADE_MODE 변수(off/observe/live) 또는 `python -m stock.guard halt`로 즉시 정지·청산.

상태 파일: live/guard.json (live 브랜치, 화면에도 표시).
  python -m stock.guard status | halt | resume | liquidate | eod
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from stock import crypt
from stock.common import OUT

START_CAPITAL = 1_000_000
HALT_EQUITY = 0.5      # 평가액이 시작 자금의 50% 이하 → 영구 정지
DAY_LOSS = 0.08        # 당일 계좌 손실 한도
MAX_GAP_BUY = 0.20     # 매도1호가가 전일 종가 대비 이 이상 높으면 매수 생략
ADV_CAP = 0.01         # 종목당 전일 거래대금 1%
BUY_WINDOW = ("090030", "092000")
EXIT_AT = "1515"
CANCEL_AFTER = 60      # 미체결 매수 취소(초)
LIVE = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ("status", "halt", "resume", "liquidate", "eod") else OUT.parent / "live"
STATE = LIVE / "guard.json"
_K = {}


def now_kst() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)


def mode() -> str:
    """off | observe | live. 정지 상태면 off."""
    m = (os.environ.get("TRADE_MODE") or "observe").lower()
    st = load()
    if st.get("halted"):
        return "off"
    return m if m in ("off", "observe", "live") else "observe"


def load() -> dict:
    return crypt.read_json(STATE, None) or {"halted": False, "pending": {}, "orders": [], "day": None, "log": []}


def save(st: dict) -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    st["log"] = st.get("log", [])[-200:]
    st["orders"] = st.get("orders", [])[-300:]
    crypt.write_json(STATE, st)


def note(st: dict, msg: str) -> None:
    line = f"{now_kst().strftime('%m-%d %H:%M:%S')} {msg}"
    print(f"[guard] {msg}", flush=True)
    st.setdefault("log", []).append(line)


def kis():
    if "c" not in _K:
        from stock.kis import KIS
        _K["c"] = KIS()
    return _K["c"]


def tick(price: float) -> int:
    for lim, t in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if price < lim:
            return t
    return 1000


def _sell_market(st: dict, code: str, qty: int, reason: str, live: bool) -> bool:
    """시장가 매도. 실패하면 pending에 올려 다음 주기에 재시도."""
    if qty <= 0:
        return True
    if not live:
        note(st, f"(관찰) 매도 예정 {code} {qty}주 · {reason}")
        return True
    try:
        book = kis().orderbook(code)
        if not book["bids"] or book["bids"][0][0] <= 0:
            raise RuntimeError("매수 호가 없음(하한가 잠김·거래정지)")
        r = kis().order("sell", code, qty, 0)
        st.setdefault("orders", []).append({"t": now_kst().strftime("%Y-%m-%d %H:%M:%S"), "side": "sell", "code": code, "qty": qty, "why": reason, "no": r.get("order_no")})
        note(st, f"매도 주문 {code} {qty}주 · {reason} · {r.get('msg')}")
        st["pending"].pop(code, None)
        return True
    except Exception as e:  # noqa: BLE001
        p = st["pending"].setdefault(code, {"qty": qty, "since": now_kst().strftime("%Y-%m-%d %H:%M"), "tries": 0, "why": reason})
        p["qty"], p["tries"], p["last_err"] = qty, p.get("tries", 0) + 1, str(e)[:120]
        note(st, f"매도 실패 → 대기 {code} {qty}주 · {e}")
        return False


def liquidate(st: dict, reason: str, keep: set | None = None, live: bool | None = None) -> None:
    """보유 전량(keep 제외) 매도. 미체결 매수 주문도 취소."""
    live = mode() == "live" if live is None else live
    keep = keep or set()
    try:
        if live:
            for o in kis().pending():
                if o["side"] == "buy":
                    kis().cancel(o["order_no"], o["org_no"])
                    note(st, f"미체결 매수 취소 {o['code']} {o['qty']}주")
        holds = kis().balance()["holdings"]
    except Exception as e:  # noqa: BLE001
        note(st, f"계좌 조회 실패(청산 보류) · {e}")
        return
    for h in holds:
        if h["code"] in keep:
            continue
        _sell_market(st, h["code"], h["qty"], reason, live)


def sweep_pending(st: dict) -> None:
    """못 판 수량 재시도(거래 재개·호가 복귀 감지)."""
    live = mode() == "live"
    for code, p in list(st.get("pending", {}).items()):
        _sell_market(st, code, p["qty"], f"대기 매도 재시도({p.get('why')})", live)


def equity() -> dict:
    b = kis().balance()
    return {"total": b["total"] or (b["cash"] + b["stock_eval"]), "cash": b["cash"], "holdings": b["holdings"]}


def halt(st: dict, reason: str) -> None:
    st["halted"], st["halted_at"], st["halt_reason"] = True, now_kst().strftime("%Y-%m-%d %H:%M"), reason
    note(st, f"자동 매매 정지 · {reason}")
    liquidate(st, f"정지 청산({reason})", live=(os.environ.get("TRADE_MODE") or "").lower() == "live")


def check_limits(st: dict, eq: dict) -> None:
    """원금 보호선·일일 손실 한도."""
    if eq["total"] <= START_CAPITAL * HALT_EQUITY and not st.get("halted"):
        halt(st, f"평가액 {eq['total']:,}원 ≤ 시작 자금의 {int(HALT_EQUITY*100)}%")
        return
    d = st.get("day") or {}
    if d.get("start_equity") and eq["total"] / d["start_equity"] - 1 <= -DAY_LOSS and not d.get("no_buy"):
        d["no_buy"], d["day_loss_hit"] = True, True
        note(st, f"일일 손실 한도 도달({eq['total']/d['start_equity']-1:+.1%}) → 전량 청산·당일 매수 금지")
        liquidate(st, "일일 손실 한도")


def can_buy(st: dict, hms: str) -> tuple[bool, str]:
    if mode() != "live":
        return False, mode()
    d = st.get("day") or {}
    if d.get("no_buy"):
        return False, d.get("no_buy_reason") or "당일 매수 금지"
    if not (BUY_WINDOW[0] <= hms <= BUY_WINDOW[1]):
        return False, "매수 시간 아님"
    if d.get("bought"):
        return False, "매수 완료"
    return True, "가능"


def buy(st: dict, plan: list[dict]) -> None:
    """plan: [{code,name,n,close,val}] 화면 매수 계획과 동일 규칙. 호가 기준 재계산 후 지정가(매도1호가) 주문."""
    d = st.setdefault("day", {})
    placed = []
    for it in plan:
        try:
            book = kis().orderbook(it["code"])
            ask = book["asks"][0] if book["asks"] else (0, 0)
            if ask[0] <= 0:
                note(st, f"매수 생략 {it['name']} · 매도 호가 없음(상한가 잠김)")
                continue
            if it.get("close") and ask[0] / it["close"] - 1 >= MAX_GAP_BUY:
                note(st, f"매수 생략 {it['name']} · 시가 갭 {ask[0]/it['close']-1:+.0%} ≥ {MAX_GAP_BUY:.0%}")
                continue
            cap = int(ADV_CAP * it["val"]) if it.get("val") else None
            qty = int(it["n"])
            if cap:
                qty = min(qty, cap // ask[0])
            qty = min(qty, max(1, ask[1]))  # 매도1호가 잔량 이내
            if qty <= 0:
                note(st, f"매수 생략 {it['name']} · 수량 0")
                continue
            r = kis().order("buy", it["code"], qty, ask[0])
            placed.append({"code": it["code"], "no": r.get("order_no"), "org": r.get("org_no"), "t": time.time()})
            st.setdefault("orders", []).append({"t": now_kst().strftime("%Y-%m-%d %H:%M:%S"), "side": "buy", "code": it["code"], "qty": qty, "price": ask[0], "no": r.get("order_no")})
            note(st, f"매수 주문 {it['name']} {qty}주 @ {ask[0]:,} · {r.get('msg')}")
        except Exception as e:  # noqa: BLE001
            note(st, f"매수 실패 {it.get('name')} · {e}")
    d["bought"], d["buy_orders"] = True, placed


def cancel_stale_buys(st: dict) -> None:
    """60초 지난 미체결 매수 취소(추격 매수 방지)."""
    d = st.get("day") or {}
    if not d.get("buy_orders") or d.get("buys_cancelled"):
        return
    if time.time() - min(o["t"] for o in d["buy_orders"]) < CANCEL_AFTER:
        return
    try:
        for o in kis().pending():
            if o["side"] == "buy":
                kis().cancel(o["order_no"], o["org_no"])
                note(st, f"미체결 매수 취소 {o['code']} {o['qty']}주")
    except Exception as e:  # noqa: BLE001
        note(st, f"미체결 조회 실패 · {e}")
    d["buys_cancelled"] = True


def make_plan(items: list[dict], capital: float) -> list[dict]:
    """화면 매수 계획과 같은 규칙으로 주수 산출. items: latest.up 상위(보통주·거래정지·관리종목 제외 전)."""
    cand = [it for it in items if not it.get("pref") and not it.get("halted") and not it.get("managed")][:5]
    if not cand or capital <= 0:
        return []
    sc = [max(0.0, (it.get("p_oc") if it.get("p_oc") is not None else it.get("p", 0)) * (1 - (it.get("p_lock") or 0))) for it in cand]
    tot = sum(sc) or 1.0
    w = [min(0.4, max(0.1, x / tot)) for x in sc]
    wt = sum(w)
    w = [x / wt for x in w]
    out = []
    for it, wi in zip(cand, w):
        cap = ADV_CAP * it["val"] if it.get("val") else float("inf")
        target = min(wi * capital, cap)
        n = int(target // it["close"]) if it.get("close") else 0
        out.append({"code": it["code"], "name": it["name"], "n": n, "close": it.get("close"), "val": it.get("val"), "w": round(wi, 3)})
    return [x for x in out if x["n"] > 0]


def cycle(day: str, hms: str, plan: list[dict], exits: dict[str, str]) -> dict:
    """live.py가 30초마다 호출. plan: 매수 계획, exits: {code: 사유} 이번 주기에 팔아야 할 종목.
    반환: 화면용 상태."""
    st = load()
    m = mode()
    live = m == "live"
    if m == "off" and not st.get("halted"):
        save(st)
        return {"mode": m, "halted": False}
    # 날짜 전환: 전날 상태 정리, 시작 평가액 기록, 이월 보유 청산
    if (st.get("day") or {}).get("date") != day:
        st["day"] = {"date": day, "start_equity": None, "no_buy": False, "bought": False, "late": hms > BUY_WINDOW[1]}
        if st["day"]["late"]:
            st["day"]["no_buy"], st["day"]["no_buy_reason"] = True, "장 시작 후 지각 실행"
    d = st["day"]
    if m in ("observe", "live") or st.get("halted"):
        try:
            eq = equity()
            d["errors"] = 0
            if d["start_equity"] is None:
                d["start_equity"] = eq["total"]
            d["equity"] = eq["total"]
            st["equity"], st["equity_at"] = eq["total"], now_kst().strftime("%H:%M:%S")
            if live:
                check_limits(st, eq)
                if hms >= "090000" and not d.get("carry_cleared"):
                    liquidate(st, "이월 보유 청산", keep={p["code"] for p in plan} if hms <= BUY_WINDOW[1] else set())
                    d["carry_cleared"] = True
        except Exception as e:  # noqa: BLE001
            d["errors"] = d.get("errors", 0) + 1
            note(st, f"계좌 조회 실패 {d['errors']}회 · {e}")
            if d["errors"] >= 5 and not d.get("no_buy"):
                d["no_buy"], d["no_buy_reason"] = True, "계좌 조회 연속 실패"
    if live:
        sweep_pending(st)
        sold = set(d.get("sold") or [])
        for code, why in exits.items():
            if code in sold:
                continue
            try:
                h = next((x for x in equity()["holdings"] if x["code"] == code), None)
            except Exception:  # noqa: BLE001
                h = None
            if h and _sell_market(st, code, h["qty"], why, True):
                sold.add(code)
        d["sold"] = sorted(sold)
        ok, why = can_buy(st, hms)
        if ok and plan:
            buy(st, plan)
        cancel_stale_buys(st)
        if hms >= EXIT_AT and not d.get("eod_done"):
            liquidate(st, "15:15 전량 매도")
            d["eod_done"] = True
    elif m == "observe":
        for code, why in exits.items():
            note(st, f"(관찰) 매도 신호 {code} · {why}")
    save(st)
    return {"mode": m, "halted": bool(st.get("halted")), "halt_reason": st.get("halt_reason"), "equity": st.get("equity"),
            "day_start": d.get("start_equity"), "no_buy": d.get("no_buy"), "no_buy_reason": d.get("no_buy_reason"),
            "pending": st.get("pending", {}), "log": st.get("log", [])[-8:]}


def main(argv):
    cmd = next((a for a in argv if a in ("status", "halt", "resume", "liquidate", "eod")), "status")
    st = load()
    if cmd == "status":
        try:
            eq = equity()
            print(json.dumps({"mode": mode(), "halted": st.get("halted"), "equity": eq["total"], "holdings": eq["holdings"], "pending": st.get("pending")}, ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001
            print(f"[guard] 계좌 조회 실패 · {e}")
    elif cmd == "halt":
        halt(st, "수동 정지")
    elif cmd == "resume":
        st["halted"], st["halt_reason"] = False, None
        note(st, "자동 매매 재개(사용자)")
    elif cmd == "liquidate":
        liquidate(st, "수동 전량 매도", live=True)
    elif cmd == "eod":  # 장 마감 후 점검: 남은 보유는 다음 거래일 매도 대기
        try:
            for h in equity()["holdings"]:
                st["pending"].setdefault(h["code"], {"qty": h["qty"], "since": now_kst().strftime("%Y-%m-%d %H:%M"), "tries": 0, "why": "마감 후 잔여 보유"})
                note(st, f"잔여 보유 {h['name']} {h['qty']}주 → 다음 거래일 매도 대기")
        except Exception as e:  # noqa: BLE001
            note(st, f"마감 점검 실패 · {e}")
    save(st)


if __name__ == "__main__":
    main(sys.argv[1:])
