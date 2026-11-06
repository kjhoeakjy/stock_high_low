"""리스크 가드: 자동 매매의 손실 방어 장치. 주문 실행(live.py)과 비상 작업(guard.yml)이 공용.

원칙
  1. 원금 보호선: 계좌 평가액이 시작 자금의 50% 이하가 되면 전량 청산하고 자동 매매를 영구 정지(사용자가 resume 전까지 다음 날부터 거래 없음).
  2. 일일 한도: 당일 계좌 손실이 DAY_LOSS(8%)에 이르면 전량 청산, 당일 신규 매수 금지.
  3. 매도 불능 대비: 팔아야 할 종목은 pending(매도 대기)에 올리고, 매 주기(30초) 계좌 보유 수량으로 체결을 확인. 보유가 0이 될 때까지
     완료로 보지 않으며, 미체결 주문은 60초 뒤 취소 후 시장가 재주문. 거래정지·하한가 잠김이면 호가가 살아날 때까지 반복. 날이 바뀌어도 유지.
  4. 이월 보유 금지: 계획에 없는 보유 종목(전날 못 판 것 포함)은 장 시작 직후 전부 매도.
  5. 매수 보호: 매도1호가가 전일 종가 대비 +20% 이상이면 매수 생략(상한가 잠김 위험), 호가 잔량·전일 거래대금 1% 한도로 수량 제한, 미체결 매수는 60초 뒤 취소.
  6. 시스템 이상: 시세·계좌 조회 연속 실패, 러너 지각(09:20 이후 시작)이면 당일 매수 금지. 조회 불능 상태에서 보유가 있으면 청산 시도.
  7. 비상 정지: TRADE_MODE 변수(off/observe/live) 또는 `python -m stock.guard halt`로 즉시 정지·청산.
  9. 기준 자금 추적: 날이 바뀔 때 D+2 예수금 변화(밤사이에는 거래가 없으므로 입출금만 반영)를 기준 자금(capital)에 더해
     원금 보호선(50%)을 입금·출금에 맞춰 갱신. 수동: python -m stock.guard capital 1500000
  8. 체결 확인: 주문 접수가 아니라 실제 체결로 판단. 매수는 당일 주문·체결 조회로 종목별 체결 수량·평단을 기록하고(미체결은 취소),
     매도는 보유 수량 0 확인 시에만 완료 처리. 정지 상태에서도 매도 대기 종목의 청산 확인은 계속한다.

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
LIVE = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in ("status", "halt", "resume", "liquidate", "eod", "capital") else OUT.parent / "live"
STATE = LIVE / "guard.json"
_K = {}


def now_kst() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)


def mode(st: dict | None = None) -> str:
    """off | observe | live. 정지 상태면 off. st를 주면 저장 전 메모리 상태 기준."""
    m = (os.environ.get("TRADE_MODE") or "observe").lower()
    st = load() if st is None else st
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


def want_sell(st: dict, code: str, qty: int, reason: str, name: str | None = None) -> None:
    """매도 대기 등록. 실제 매도 주문·체결 확인은 sweep_pending이 맡는다."""
    p = st.setdefault("pending", {}).setdefault(code, {"qty": qty, "since": now_kst().strftime("%Y-%m-%d %H:%M"), "tries": 0, "why": reason, "name": name})
    p["qty"] = max(int(p.get("qty") or 0), int(qty))
    p["why"], p["name"] = reason, name or p.get("name")


def _sell_market(st: dict, code: str, qty: int, reason: str, live: bool) -> bool:
    """시장가 매도 주문. 접수 성공은 체결이 아니므로 pending을 지우지 않는다(sweep_pending이 보유 0 확인 후 삭제)."""
    if qty <= 0:
        return True
    if not live:
        note(st, f"(관찰) 매도 예정 {code} {qty}주 · {reason}")
        return True
    p = st.setdefault("pending", {}).setdefault(code, {"qty": qty, "since": now_kst().strftime("%Y-%m-%d %H:%M"), "tries": 0, "why": reason})
    try:
        book = kis().orderbook(code)
        if not book["bids"] or book["bids"][0][0] <= 0:
            raise RuntimeError("매수 호가 없음(하한가 잠김·거래정지)")
        r = kis().order("sell", code, qty, 0)
        st.setdefault("orders", []).append({"t": now_kst().strftime("%Y-%m-%d %H:%M:%S"), "side": "sell", "code": code, "qty": qty, "why": reason,
                                            "no": r.get("order_no"), "org": r.get("org_no"), "filled": None})
        p["qty"], p["tries"], p["order_no"], p["ordered_at"] = qty, p.get("tries", 0) + 1, r.get("order_no"), time.time()
        p.pop("last_err", None)
        note(st, f"매도 주문 접수 {code} {qty}주 · {reason} · {r.get('msg')} → 체결 확인 대기")
        return True
    except Exception as e:  # noqa: BLE001
        p["qty"], p["tries"], p["last_err"] = qty, p.get("tries", 0) + 1, str(e)[:120]
        note(st, f"매도 주문 실패 → 재시도 대기 {code} {qty}주 · {e}")
        return False


def liquidate(st: dict, reason: str, keep: set | None = None, live: bool | None = None) -> bool:
    """보유 전량(keep 제외) 매도 대기 등록 후 즉시 주문. 미체결 매수 주문도 취소. 계좌 조회 실패면 False(다음 주기 재시도)."""
    live = mode(st) == "live" if live is None else live
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
        return False
    for h in holds:
        if h["code"] in keep:
            continue
        if live:
            want_sell(st, h["code"], h["qty"], reason, h.get("name"))
        else:
            note(st, f"(관찰) 매도 예정 {h.get('name') or h['code']} {h['qty']}주 · {reason}")
    if live:
        sweep_pending(st, live=True)
    return True


def sweep_pending(st: dict, live: bool | None = None) -> None:
    """매도 대기 종목의 체결 확인·재주문. 계좌 보유 수량이 0이어야 완료.
    보유가 남아 있고 미체결 주문이 없으면 시장가 주문, 미체결이 CANCEL_AFTER를 넘기면 취소 후 다음 주기에 재주문."""
    live = (os.environ.get("TRADE_MODE") or "").lower() == "live" if live is None else live
    if not st.get("pending") or not live:
        return
    try:
        holds = {h["code"]: h for h in kis().balance()["holdings"]}
        opens = kis().pending()
    except Exception as e:  # noqa: BLE001
        note(st, f"체결 확인 실패(계좌 조회) · {e}")
        return
    d = st.setdefault("day", {}) or {}
    for code, p in list(st["pending"].items()):
        h = holds.get(code)
        if not h or h["qty"] <= 0:
            st["pending"].pop(code, None)
            d.setdefault("sold", [])
            if code not in d["sold"]:
                d["sold"].append(code)
            note(st, f"매도 체결 확인 {p.get('name') or code} · {p.get('why')}" if p.get("order_no") else f"매도 대상 보유 없음 {p.get('name') or code}(미체결 매수)")
            continue
        p["qty"], p["name"] = h["qty"], h.get("name") or p.get("name")
        open_sells = [o for o in opens if o["code"] == code and o["side"] == "sell"]
        if open_sells:
            if time.time() - float(p.get("ordered_at") or 0) < CANCEL_AFTER:
                continue  # 주문 체결 대기 중
            for o in open_sells:
                try:
                    kis().cancel(o["order_no"], o["org_no"])
                    note(st, f"매도 미체결 취소 {p.get('name') or code} {o['qty']}주 → 재주문")
                except Exception as e:  # noqa: BLE001
                    note(st, f"매도 취소 실패 {code} · {e}")
            p["ordered_at"] = None
            continue
        if p.get("order_no") and p.get("ordered_at") and time.time() - p["ordered_at"] < 5:
            continue  # 접수 직후 잔고 반영 지연
        sellable = h.get("sellable") if h.get("sellable") is not None else h["qty"]
        _sell_market(st, code, sellable or h["qty"], p.get("why") or "매도 대기", True)


def equity() -> dict:
    b = kis().balance()
    return {"total": b["total"] or (b["cash"] + b["stock_eval"]), "cash": b["cash"], "holdings": b["holdings"], "raw": b}


def halt(st: dict, reason: str) -> None:
    st["halted"], st["halted_at"], st["halt_reason"] = True, now_kst().strftime("%Y-%m-%d %H:%M"), reason
    note(st, f"자동 매매 정지 · {reason}")
    liquidate(st, f"정지 청산({reason})", live=(os.environ.get("TRADE_MODE") or "").lower() == "live")


def capital(st: dict) -> int:
    return int(st.get("capital") or START_CAPITAL)


def track_deposits(st: dict, eq: dict) -> None:
    """날이 바뀐 첫 조회에서 전날 마지막 D+2 예수금과 비교해 입출금을 감지하고 기준 자금에 반영."""
    raw = eq.get("raw") or {}
    d2 = raw.get("d2_cash")
    if d2 is None:
        return
    prev = st.get("eod") or {}
    today = (st.get("day") or {}).get("date")
    if prev.get("date") and prev.get("date") != today and prev.get("d2_cash") is not None and not st.get("day", {}).get("deposit_checked"):
        delta = int(d2) - int(prev["d2_cash"])
        st["day"]["deposit_checked"] = True
        if abs(delta) >= 10_000:
            st["capital"] = capital(st) + delta
            note(st, f"입출금 감지 {delta:+,}원 → 기준 자금 {capital(st):,}원(보호선 {int(capital(st) * HALT_EQUITY):,}원)")
    st["eod"] = {"date": today, "d2_cash": int(d2), "total": eq["total"]}


def check_limits(st: dict, eq: dict) -> None:
    """원금 보호선·일일 손실 한도."""
    if eq["total"] <= capital(st) * HALT_EQUITY and not st.get("halted"):
        halt(st, f"평가액 {eq['total']:,}원 ≤ 기준 자금 {capital(st):,}원의 {int(HALT_EQUITY*100)}%")
        return
    d = st.get("day") or {}
    if d.get("start_equity") and eq["total"] / d["start_equity"] - 1 <= -DAY_LOSS and not d.get("no_buy"):
        d["no_buy"], d["day_loss_hit"] = True, True
        note(st, f"일일 손실 한도 도달({eq['total']/d['start_equity']-1:+.1%}) → 전량 청산·당일 매수 금지")
        liquidate(st, "일일 손실 한도")


def can_buy(st: dict, hms: str) -> tuple[bool, str]:
    if mode(st) != "live":
        return False, mode(st)
    d = st.get("day") or {}
    if d.get("no_buy"):
        return False, d.get("no_buy_reason") or "당일 매수 금지"
    if not (BUY_WINDOW[0] <= hms <= BUY_WINDOW[1]):
        return False, "매수 시간 아님"
    if d.get("bought"):
        return False, "매수 완료"
    return True, "가능"


def buy(st: dict, plan: list[dict], day: str) -> None:
    """plan: [{code,name,n,close,val}] 화면 매수 계획과 동일 규칙. 호가 기준 재계산 후 지정가(매도1호가) 주문.
    중복 방지: 당일 이미 매수 주문이 있는 종목은 건너뛴다(프로세스 재시작 대비). 총액은 실제 주문가능금액 이내."""
    d = st.setdefault("day", {})
    placed = list(d.get("buy_orders") or [])
    try:
        already = {o["code"] for o in kis().orders(day)["orders"] if o["side"] == "buy"}
        power = kis().buying_power()
    except Exception as e:  # noqa: BLE001
        note(st, f"매수 보류(주문 내역·주문가능금액 조회 실패) · {e}")
        return
    if already:
        note(st, f"당일 기존 매수 주문 감지 {sorted(already)} → 해당 종목 건너뜀")
    spent = 0
    for it in plan:
        if it["code"] in already:
            continue
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
            qty = min(qty, int(max(0, power - spent) * 0.995) // ask[0])  # 주문가능금액(수수료 여유) 이내
            if qty <= 0:
                note(st, f"매수 생략 {it['name']} · 수량 0(한도·잔량·주문가능금액)")
                continue
            r = kis().order("buy", it["code"], qty, ask[0])
            spent += qty * ask[0]
            placed.append({"code": it["code"], "name": it["name"], "qty": qty, "price": ask[0], "no": r.get("order_no"), "org": r.get("org_no"), "t": time.time()})
            st.setdefault("orders", []).append({"t": now_kst().strftime("%Y-%m-%d %H:%M:%S"), "side": "buy", "code": it["code"], "qty": qty, "price": ask[0],
                                                "no": r.get("order_no"), "org": r.get("org_no"), "filled": None})
            note(st, f"매수 주문 접수 {it['name']} {qty}주 @ {ask[0]:,} · {r.get('msg')} → 체결 확인 대기")
            d["bought"], d["buy_orders"] = True, placed
            save(st)  # 주문 직후 저장(중단 시 중복 주문 방지)
        except Exception as e:  # noqa: BLE001
            note(st, f"매수 주문 실패 {it.get('name')} · {e}")
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
        return
    d["buys_cancelled"] = True


def confirm_fills(st: dict, day: str) -> None:
    """당일 주문·체결 조회로 종목별 실제 체결 수량·평단 기록. 주문 접수 수량과 체결 수량을 구분해 화면·로그에 남긴다."""
    d = st.setdefault("day", {}) or {}
    try:
        res = kis().orders(day)
    except Exception as e:  # noqa: BLE001
        note(st, f"체결 조회 실패 · {e}")
        return
    agg: dict[str, dict] = {}
    by_no = {o.get("order_no"): o for o in res["orders"] if o.get("order_no")}
    for o in res["orders"]:
        a = agg.setdefault(f"{o['side']}:{o['code']}", {"side": o["side"], "code": o["code"], "name": o["name"], "ordered": 0, "filled": 0, "amount": 0, "cancelled": 0, "open": 0})
        a["ordered"] += o["ord_qty"]; a["filled"] += o["qty"]; a["amount"] += o["amount"]
        if o["cancelled"]:
            a["cancelled"] += o["rmn_qty"] or o["ord_qty"]
        elif o["rmn_qty"] > 0:
            a["open"] += o["rmn_qty"]
    for a in agg.values():
        a["avg"] = round(a["amount"] / a["filled"]) if a["filled"] else None
    d["fills"] = list(agg.values())
    d["fill_total"] = res["total"]
    for rec in st.get("orders", []):
        o = by_no.get(rec.get("no"))
        if o:
            rec["filled"], rec["avg"], rec["cancelled"] = o["qty"], o["price"], o["cancelled"]
    if d.get("buys_cancelled") and not d.get("buy_confirmed"):
        d["buy_confirmed"] = True
        buys = [a for a in agg.values() if a["side"] == "buy"]
        txt = ", ".join(f"{a['name']} {a['filled']}/{a['ordered']}주" + (f" @ {a['avg']:,}" if a["avg"] else "") for a in buys) or "없음"
        note(st, f"매수 체결 결과 · {txt}")
        d["bought_codes"] = [a["code"] for a in buys if a["filled"] > 0]


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
            st["holdings"] = [{k: h.get(k) for k in ("code", "name", "qty", "sellable", "avg", "price", "eval", "pnl", "pnl_rt")} for h in eq["holdings"]]
            st["account"] = {k: eq["raw"].get(k) for k in ("cash", "d1_cash", "d2_cash", "total", "stock_eval", "today_buy_amt", "today_sell_amt", "today_fee", "pnl_sum")}
            track_deposits(st, eq)
            if live:
                check_limits(st, eq)
                if hms >= "090000" and not d.get("carry_cleared"):
                    d["carry_cleared"] = liquidate(st, "이월 보유 청산", keep={p["code"] for p in plan} if hms <= BUY_WINDOW[1] else set(), live=True)
        except Exception as e:  # noqa: BLE001
            d["errors"] = d.get("errors", 0) + 1
            note(st, f"계좌 조회 실패 {d['errors']}회 · {e}")
            if d["errors"] >= 5 and not d.get("no_buy"):
                d["no_buy"], d["no_buy_reason"] = True, "계좌 조회 연속 실패"
    can_order = (os.environ.get("TRADE_MODE") or "").lower() == "live"  # 정지 상태여도 잔여 보유 청산·체결 확인은 계속
    live = live and not st.get("halted")  # 이번 주기에 정지됐으면 신규 매수 경로 차단
    if live:
        sold = set(d.get("sold") or [])
        for code, why in exits.items():
            if code in sold or code in st.get("pending", {}):
                continue
            want_sell(st, code, 0, why)  # 수량은 sweep_pending이 계좌 보유로 확정
        d["sold"] = sorted(sold)
        ok, why = can_buy(st, hms)
        if ok and plan:
            buy(st, plan, day)
        cancel_stale_buys(st)
        if hms >= EXIT_AT and not d.get("eod_done"):
            d["eod_done"] = liquidate(st, "15:15 전량 매도")
    if can_order:
        sweep_pending(st, live=True)
        if d.get("buy_orders") and (not d.get("buy_confirmed") or st.get("pending") or hms >= EXIT_AT):
            confirm_fills(st, day)
    elif m == "observe":
        for code, why in exits.items():
            note(st, f"(관찰) 매도 신호 {code} · {why}")
    save(st)
    return state_view(st)


def state_view(st: dict) -> dict:
    d = st.get("day") or {}
    return {"mode": mode(st), "halted": bool(st.get("halted")), "halt_reason": st.get("halt_reason"), "equity": st.get("equity"), "equity_at": st.get("equity_at"),
            "day_start": d.get("start_equity"), "no_buy": d.get("no_buy"), "no_buy_reason": d.get("no_buy_reason"),
            "pending": st.get("pending", {}), "fills": d.get("fills") or [], "fill_total": d.get("fill_total"), "sold": d.get("sold") or [],
            "buy_placed": bool(d.get("bought")), "buy_confirmed": bool(d.get("buy_confirmed")), "eod_done": bool(d.get("eod_done")),
            "holdings": st.get("holdings") or [], "account": st.get("account"), "capital": capital(st), "log": st.get("log", [])[-10:]}


def main(argv):
    cmd = next((a for a in argv if a in ("status", "halt", "resume", "liquidate", "eod", "capital")), "status")
    st = load()
    if cmd == "capital":
        val = next((int(a) for a in argv if a.isdigit()), None)
        if val:
            st["capital"] = val
            note(st, f"기준 자금 수동 설정 {val:,}원")
        print(f"[guard] 기준 자금 {capital(st):,}원 · 보호선 {int(capital(st) * HALT_EQUITY):,}원")
        save(st)
        return
    if cmd == "status":
        try:
            eq = equity()
            day = now_kst().strftime("%Y%m%d")
            print(json.dumps({"mode": mode(), "halted": st.get("halted"), "equity": eq["total"], "holdings": eq["holdings"], "pending": st.get("pending"),
                              "today_orders": kis().orders(day), "open_orders": kis().pending(), "day": st.get("day")}, ensure_ascii=False, indent=1))
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
            confirm_fills(st, now_kst().strftime("%Y%m%d"))
            for h in equity()["holdings"]:
                want_sell(st, h["code"], h["qty"], "마감 후 잔여 보유", h.get("name"))
                note(st, f"잔여 보유 {h['name']} {h['qty']}주 → 다음 거래일 매도 대기")
            for a in (st.get("day") or {}).get("fills") or []:
                print(f"[guard] 체결 {a['side']} {a['name']} {a['filled']}/{a['ordered']}주 평단 {a['avg']}")
        except Exception as e:  # noqa: BLE001
            note(st, f"마감 점검 실패 · {e}")
    save(st)


if __name__ == "__main__":
    main(sys.argv[1:])
