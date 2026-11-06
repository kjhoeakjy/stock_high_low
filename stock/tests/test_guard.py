"""리스크 가드 오프라인 검증: 가짜 증권사 계좌로 매수·체결 확인·손절·거래정지 재시도·마감 청산·원금 보호선 정지를 전 경로 실행.
  python -m stock.tests.test_guard
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["TRADE_MODE"] = "live"
os.environ.pop("PAGE_PASSWORD", None)
from stock import guard  # noqa: E402


class FakeKIS:
    """체결 규칙: 시장가 매도는 즉시 전량 체결(halted면 호가 없음 예외, stuck이면 접수만 되고 미체결).
    지정가 매수는 fill_rate 비율만 체결(나머지 미체결 → 취소 대상)."""

    def __init__(self):
        self.cash = 1_000_000
        self.hold: dict[str, dict] = {}
        self.orders: list[dict] = []
        self.open: list[dict] = []
        self.halted: set[str] = set()
        self.stuck: set[str] = set()
        self.fill_rate: dict[str, float] = {}
        self.px: dict[str, int] = {}
        self.n = 0
        self.day = ""

    def _no(self):
        self.n += 1
        return f"{self.n:06d}"

    def price_of(self, c):
        return self.px.get(c, 10_000)

    def orderbook(self, c):
        if c in self.halted:
            return {"asks": [], "bids": []}
        p = self.price_of(c)
        return {"asks": [(p + 10, 1000)], "bids": [(p, 1000)]}

    def balance(self):
        holds = [{"code": c, "name": h["name"], "qty": h["qty"], "sellable": h["qty"] - sum(o["qty"] for o in self.open if o["code"] == c and o["side"] == "sell"),
                  "avg": h["avg"], "price": self.price_of(c), "eval": h["qty"] * self.price_of(c), "pnl": 0, "pnl_rt": 0.0} for c, h in self.hold.items() if h["qty"] > 0]
        se = sum(h["eval"] for h in holds)
        return {"cash": self.cash, "d2_cash": self.cash, "d1_cash": self.cash, "total": self.cash + se, "stock_eval": se, "holdings": holds,
                "today_buy_amt": 0, "today_sell_amt": 0, "today_fee": 0, "pnl_sum": 0}

    def buying_power(self, *a, **k):
        return self.cash

    def order(self, side, code, qty, price=0):
        no = self._no()
        rec = {"day": self.day, "code": code, "name": code, "side": side, "ord_qty": qty, "qty": 0, "rmn_qty": qty, "ord_price": price, "price": 0.0, "amount": 0,
               "cancelled": False, "order_no": no, "org_no": "0", "time": "090030"}
        self.orders.append(rec)
        if side == "sell":
            if code in self.halted:
                raise RuntimeError("KIS 하한가 잠김")
            if code in self.stuck:
                self.open.append({"code": code, "side": side, "qty": qty, "order_no": no, "org_no": "0"})
                return {"ok": True, "order_no": no, "org_no": "0", "msg": "접수"}
            p = self.price_of(code)
            rec.update(qty=qty, rmn_qty=0, price=float(p), amount=qty * p)
            self.hold[code]["qty"] -= qty
            self.cash += qty * p
            return {"ok": True, "order_no": no, "org_no": "0", "msg": "접수"}
        cost = qty * price
        if cost > self.cash:
            raise RuntimeError("KIS 주문가능금액 초과")
        f = int(qty * self.fill_rate.get(code, 1.0))
        if f:
            rec.update(qty=f, rmn_qty=qty - f, price=float(price), amount=f * price)
            h = self.hold.setdefault(code, {"name": code, "qty": 0, "avg": 0.0})
            h["avg"] = (h["avg"] * h["qty"] + f * price) / (h["qty"] + f)
            h["qty"] += f
            self.cash -= f * price
        if qty - f:
            self.open.append({"code": code, "side": side, "qty": qty - f, "order_no": no, "org_no": "0"})
        return {"ok": True, "order_no": no, "org_no": "0", "msg": "접수"}

    def pending(self):
        return [dict(o, price=0, name=o["code"], time="") for o in self.open]

    def cancel(self, order_no, org_no, qty=0):
        self.open = [o for o in self.open if o["order_no"] != order_no]
        for r in self.orders:
            if r["order_no"] == order_no:
                r["cancelled"] = True
        return {"ok": True, "msg": "취소"}

    def orders_q(self, day, filled_only=False):
        out = [dict(r) for r in self.orders if r["day"] == day and (r["qty"] > 0 or not filled_only)]
        return {"orders": out, "total": {"buy_amt": sum(o["amount"] for o in out if o["side"] == "buy"), "sell_amt": sum(o["amount"] for o in out if o["side"] == "sell"),
                                         "buy_qty": 0, "sell_qty": 0}}



def main():
    tmp = Path(tempfile.mkdtemp())
    guard.LIVE, guard.STATE = tmp, tmp / "guard.json"
    fk = FakeKIS()

    class K:  # orders 속성명 충돌 회피 래퍼
        def __getattr__(self, n):
            return getattr(fk, "orders_q" if n == "orders" else n)
    guard._K["c"] = K()
    day = "20261110"
    fk.day = day
    plan = [{"code": "A", "name": "A", "n": 30, "close": 10_000, "val": 5e9}, {"code": "B", "name": "B", "n": 30, "close": 10_000, "val": 5e9},
            {"code": "C", "name": "C", "n": 30, "close": 10_000, "val": 5e9}]
    fk.fill_rate = {"A": 1.0, "B": 0.5, "C": 0.0}
    fk.hold["X"] = {"name": "X", "qty": 7, "avg": 10_000.0}  # 전날 못 판 이월 보유

    # 1) 09:00:30 — 이월 보유 청산 + 매수 주문
    s = guard.cycle(day, "090030", plan, {})
    assert "X" not in fk.hold or fk.hold["X"]["qty"] == 0, "이월 보유 미청산"
    assert "X" not in s["pending"], "이월 매도 체결 확인 실패"
    assert s["buy_placed"] and fk.hold["A"]["qty"] == 30 and fk.hold["B"]["qty"] == 15 and "C" not in fk.hold
    assert len(fk.open) == 2, "미체결 매수 2건(B 잔량, C 전량) 기대"
    # 2) 60초 뒤 — 미체결 취소, 체결 결과 확정
    st = guard.load(); [o.__setitem__("t", o["t"] - 61) for o in st["day"]["buy_orders"]]; guard.save(st)
    s = guard.cycle(day, "090200", plan, {})
    assert not fk.open and s["buy_confirmed"], "미체결 매수 취소·체결 확정 실패"
    fills = {f["code"]: f for f in s["fills"] if f["side"] == "buy"}
    assert fills["A"]["filled"] == 30 and fills["B"]["filled"] == 15 and fills["B"]["ordered"] == 30 and fills["C"]["filled"] == 0, fills
    # 2b) 재시작 중복 방지 — bought 플래그가 사라져도 당일 매수 주문이 있으면 재주문 금지
    st = guard.load(); st["day"]["bought"] = False; guard.save(st)
    n_before = len(fk.orders)
    guard.cycle(day, "090300", plan, {})
    assert len(fk.orders) == n_before, "중복 매수 발생"
    # 3) 손절 신호: C(미체결)는 보유 없음 처리, A는 거래정지 → 재시도 대기
    fk.halted.add("A")
    s = guard.cycle(day, "094000", plan, {"C": "손절", "A": "손절"})
    assert "C" in s["sold"] and "C" not in s["pending"]
    assert "A" in s["pending"] and s["pending"]["A"]["tries"] >= 1 and fk.hold["A"]["qty"] == 30
    s = guard.cycle(day, "094030", plan, {})
    assert s["pending"]["A"]["tries"] >= 2, "거래정지 중 재시도 없음"
    fk.halted.discard("A")  # 거래 재개
    s = guard.cycle(day, "094100", plan, {})
    assert fk.hold["A"]["qty"] == 0 and "A" in s["pending"], "접수 직후에는 체결 확인 전이어야 함"
    s = guard.cycle(day, "094130", plan, {})
    assert "A" not in s["pending"] and fk.hold["A"]["qty"] == 0 and "A" in s["sold"], "거래 재개 후 매도 체결 확인 실패"
    # 4) 15:15 청산: B 주문은 접수되지만 미체결(stuck) → 60초 뒤 취소·재주문 → 체결 확인
    fk.stuck.add("B")
    s = guard.cycle(day, "151500", plan, {})
    assert s["eod_done"] and "B" in s["pending"] and fk.hold["B"]["qty"] == 15 and len(fk.open) == 1, "마감 매도 접수 상태 오류"
    s = guard.cycle(day, "151530", plan, {})
    assert "B" in s["pending"] and len(fk.open) == 1, "체결 대기 중 중복 주문"
    st = guard.load(); st["pending"]["B"]["ordered_at"] -= 61; guard.save(st)
    fk.stuck.discard("B")
    s = guard.cycle(day, "151630", plan, {})
    assert not fk.open, "미체결 매도 취소 실패"
    guard.cycle(day, "151700", plan, {})
    s = guard.cycle(day, "151730", plan, {})
    assert "B" not in s["pending"] and fk.hold["B"]["qty"] == 0 and "B" in s["sold"], "재주문 체결 확인 실패"
    assert not any(h["qty"] for h in fk.hold.values()), "마감 후 보유 잔량"
    # 5) 다음 날 — 평가액 50% 이하 → 영구 정지, 이후 매수 금지, 잔여 청산은 계속
    day2 = "20261111"
    fk.day = day2
    fk.cash = 450_000
    fk.hold["D"] = {"name": "D", "qty": 4, "avg": 10_000.0}
    fk.halted.add("D")
    s = guard.cycle(day2, "090030", plan, {})
    assert s["halted"] and s["mode"] == "off" and "D" in s["pending"], "원금 보호선 정지 실패"
    assert not any(o["side"] == "buy" and o["day"] == day2 for o in fk.orders), "정지 후 매수 발생"
    fk.halted.discard("D")
    guard.cycle(day2, "090100", plan, {})
    s = guard.cycle(day2, "090130", plan, {})
    assert "D" not in s["pending"] and fk.hold["D"]["qty"] == 0, "정지 상태 잔여 청산 실패"
    assert s["halted"] and not s["buy_placed"]
    print("[test_guard] 통과 · 주문", len(fk.orders), "건 · 로그:")
    for line in guard.load()["log"]:
        print("  ", line)


if __name__ == "__main__":
    main()
