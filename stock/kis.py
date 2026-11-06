"""한국투자증권 Open API(KIS Developers) 최소 연동: 토큰, 현재가·호가, 잔고·주문가능금액, 현금 주문, 체결 조회.

환경변수
  KIS_APP_KEY / KIS_APP_SECRET            실전투자 키
  KIS_PAPER_APP_KEY / KIS_PAPER_APP_SECRET 모의투자 키
  KIS_ACCOUNT                              계좌번호 "12345678-01"
  KIS_MODE                                 paper(기본) | real
  KIS_TOKEN_DIR                            토큰 캐시 폴더(토큰은 하루 유효, 분당 1회 발급 제한)

  python -m stock.kis check          # 연결 점검: 잔고·주문가능금액·삼성전자 현재가(읽기만)
  python -m stock.kis price 005930
  python -m stock.kis account        # 계좌 종합(잔고·보유·주문가능·오늘 주문/체결·미체결·기간 손익·실현손익)
  python -m stock.kis orders 20261106 | fills 20261106 | pnl 20261001 20261106
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = {"real": "https://openapi.koreainvestment.com:9443", "paper": "https://openapivts.koreainvestment.com:29443"}
# 거래 ID(실전/모의). 현금 매수·매도는 2024년 개정 코드(TTTC0012U/0011U), 조회류는 기존 코드.
TR = {
    "balance": ("TTTC8434R", "VTTC8434R"), "psbl": ("TTTC8908R", "VTTC8908R"),
    "buy": ("TTTC0012U", "VTTC0012U"), "sell": ("TTTC0011U", "VTTC0011U"),
    "ccld": ("TTTC8001R", "VTTC8001R"), "cancel": ("TTTC0013U", "VTTC0013U"), "pending": ("TTTC8036R", "TTTC8036R"),
    # 실전 전용(모의 미지원): 실현손익 포함 잔고, 기간별 손익(일별 합산), 기간별 종목 매매손익
    "balance_rlz": ("TTTC8494R", "TTTC8494R"), "pnl_daily": ("TTTC8708R", "TTTC8708R"), "pnl_stock": ("TTTC8715R", "TTTC8715R"),
}


def _i(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


class KIS:
    def __init__(self, mode: str | None = None):
        self.mode = mode or os.environ.get("KIS_MODE", "paper")
        pre = "KIS_PAPER_" if self.mode == "paper" else "KIS_"
        self.key = os.environ.get(f"{pre}APP_KEY") or os.environ.get("KIS_APP_KEY")
        self.secret = os.environ.get(f"{pre}APP_SECRET") or os.environ.get("KIS_APP_SECRET")
        acct = (os.environ.get("KIS_ACCOUNT") or "").replace(" ", "")
        self.cano, self.prdt = (acct.split("-") + ["01"])[:2] if acct else ("", "01")
        self.host = HOST[self.mode]
        self.cache = Path(os.environ.get("KIS_TOKEN_DIR", "/tmp")) / f"kis_token_{self.mode}.json"
        self._token = None
        if not (self.key and self.secret):
            raise RuntimeError(f"KIS 키 없음(mode={self.mode})")

    # ---- 인증
    def token(self) -> str:
        if self._token:
            return self._token
        if self.cache.exists():
            try:
                c = json.loads(self.cache.read_text())
                if c.get("exp", 0) - time.time() > 600 and c.get("key") == self.key[:6]:
                    self._token = c["token"]
                    return self._token
            except Exception:  # noqa: BLE001
                pass
        body = {"grant_type": "client_credentials", "appkey": self.key, "appsecret": self.secret}
        j = self._post("/oauth2/tokenP", body, auth=False)
        self._token = j["access_token"]
        self.cache.write_text(json.dumps({"token": self._token, "exp": time.time() + int(j.get("expires_in", 86400)), "key": self.key[:6]}))
        return self._token

    def _headers(self, tr_id: str | None = None, auth: bool = True) -> dict:
        h = {"content-type": "application/json; charset=utf-8", "appkey": self.key, "appsecret": self.secret, "custtype": "P"}
        if auth:
            h["authorization"] = f"Bearer {self.token()}"
        if tr_id:
            h["tr_id"] = tr_id
        return h

    def _post(self, path: str, body: dict, tr_id: str | None = None, auth: bool = True) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode(), headers=self._headers(tr_id, auth), method="POST")
        return self._send(req)

    def _get(self, path: str, params: dict, tr_id: str) -> dict:
        q = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
        req = urllib.request.Request(f"{self.host}{path}?{q}", headers=self._headers(tr_id), method="GET")
        return self._send(req)

    @staticmethod
    def _send(req) -> dict:
        for i in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    j = json.loads(r.read().decode())
                if str(j.get("rt_cd", "0")) != "0":
                    raise RuntimeError(f"KIS {j.get('msg_cd')} {j.get('msg1')}")
                return j
            except urllib.error.HTTPError as e:
                txt = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403) or i == 2:
                    raise RuntimeError(f"KIS http {e.code}: {txt}") from e
                time.sleep(1.5)
        raise RuntimeError("KIS 요청 실패")

    def _tr(self, name: str) -> str:
        return TR[name][0 if self.mode == "real" else 1]

    # ---- 시세(실전·모의 공통 tr_id)
    def price(self, code: str) -> dict:
        o = self._get("/uapi/domestic-stock/v1/quotations/inquire-price",
                      {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}, "FHKST01010100")["output"]
        return {"code": code, "price": int(o["stck_prpr"]), "open": int(o["stck_oprc"]), "high": int(o["stck_hgpr"]),
                "low": int(o["stck_lwpr"]), "prev": int(o["stck_sdpr"]), "upper": int(o["stck_mxpr"]), "lower": int(o["stck_llam"]),
                "volume": int(o["acml_vol"]), "value": int(o["acml_tr_pbmn"]), "rate": float(o["prdy_ctrt"]) / 100,
                "halt": o.get("temp_stop_yn") == "Y", "vi": o.get("vi_cls_code")}

    def orderbook(self, code: str) -> dict:
        o = self._get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
                      {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}, "FHKST01010200")["output1"]
        asks = [(int(o[f"askp{i}"]), int(o[f"askp_rsqn{i}"])) for i in range(1, 11)]
        bids = [(int(o[f"bidp{i}"]), int(o[f"bidp_rsqn{i}"])) for i in range(1, 11)]
        return {"code": code, "asks": asks, "bids": bids, "ask_total": int(o["total_askp_rsqn"]), "bid_total": int(o["total_bidp_rsqn"]),
                "spread": (asks[0][0] - bids[0][0]) / bids[0][0] if bids[0][0] else None}

    # ---- 계좌
    def balance(self) -> dict:
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
             "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-balance", p, self._tr("balance"))
        o2 = (j.get("output2") or [{}])[0]
        holds = [{"code": h["pdno"], "name": h["prdt_name"], "qty": _i(h["hldg_qty"]), "sellable": _i(h.get("ord_psbl_qty")),
                  "avg": _f(h["pchs_avg_pric"]), "cost": _i(h.get("pchs_amt")), "price": _i(h["prpr"]), "eval": _i(h["evlu_amt"]),
                  "pnl": _i(h["evlu_pfls_amt"]), "pnl_rt": _f(h.get("evlu_pfls_rt")),
                  "today_buy": _i(h.get("thdt_buyqty")), "today_sell": _i(h.get("thdt_sll_qty"))}
                 for h in j.get("output1") or [] if _i(h.get("hldg_qty")) > 0]
        return {"cash": _i(o2.get("dnca_tot_amt")), "d2_cash": _i(o2.get("prvs_rcdl_excc_amt")), "d1_cash": _i(o2.get("nxdy_excc_amt")),
                "total": _i(o2.get("tot_evlu_amt")), "stock_eval": _i(o2.get("scts_evlu_amt")), "net_asset": _i(o2.get("nass_amt")),
                "cost_sum": _i(o2.get("pchs_amt_smtl_amt")), "pnl_sum": _i(o2.get("evlu_pfls_smtl_amt")), "asset_chg": _i(o2.get("asst_icdc_amt")),
                "today_buy_amt": _i(o2.get("thdt_buy_amt")), "today_sell_amt": _i(o2.get("thdt_sll_amt")), "today_fee": _i(o2.get("thdt_tlex_amt")),
                "holdings": holds}

    def balance_rlz(self) -> dict:
        """실현손익 포함 잔고(실전 전용): 보유 종목별 실현손익과 계좌 누적 실현손익."""
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "00", "UNPR_DVSN": "01",
             "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00", "COST_ICLD_YN": "Y", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl", p, self._tr("balance_rlz"))
        o2 = (j.get("output2") or [{}])[0]
        return {"realized": _i(o2.get("rlzt_pfls")), "realized_rt": _f(o2.get("rlzt_erng_rt")), "total_pnl": _i(o2.get("evlu_pfls_smtl_amt")),
                "total_rt": _f(o2.get("tot_pftrt")), "cost_sum": _i(o2.get("pchs_amt_smtl_amt")), "eval_sum": _i(o2.get("evlu_amt_smtl_amt")),
                "holdings": [{"code": h["pdno"], "name": h["prdt_name"], "qty": _i(h["hldg_qty"]), "avg": _f(h["pchs_avg_pric"]), "price": _i(h["prpr"]),
                              "pnl": _i(h["evlu_pfls_amt"]), "pnl_rt": _f(h.get("evlu_pfls_rt")), "realized": _i(h.get("rlzt_pfls"))}
                             for h in j.get("output1") or [] if _i(h.get("hldg_qty")) > 0]}

    def pnl_daily(self, start: str, end: str) -> dict:
        """기간별 손익 일별 합산(실전 전용). start/end YYYYMMDD. days: [{date, buy_amt, sell_amt, realized, fee, tax, rt}], total."""
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": start, "INQR_END_DT": end, "SORT_DVSN": "00", "PDNO": "",
             "INQR_DVSN": "00", "CBLC_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-period-profit", p, self._tr("pnl_daily"))
        o2 = (j.get("output2") or [{}])
        o2 = o2[0] if isinstance(o2, list) else o2
        return {"days": [{"date": r.get("trad_dt"), "buy_amt": _i(r.get("buy_amt")), "sell_amt": _i(r.get("sll_amt")), "realized": _i(r.get("rlzt_pfls")),
                          "fee": _i(r.get("fee")), "tax": _i(r.get("tl_tax")), "rt": _f(r.get("pfls_rt"))} for r in j.get("output1") or []],
                "total": {"buy_amt": _i(o2.get("buy_amt_smtl")), "sell_amt": _i(o2.get("sll_amt_smtl")), "realized": _i(o2.get("rlzt_pfls_smtl")),
                          "fee": _i(o2.get("fee_smtl")), "tax": _i(o2.get("tl_tax_smtl")), "rt": _f(o2.get("pfls_rt"))}}

    def pnl_stock(self, start: str, end: str) -> list[dict]:
        """기간별 종목 매매 손익(실전 전용): 날짜·종목별 매수·매도 금액과 실현손익."""
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": start, "INQR_END_DT": end, "SORT_DVSN": "00", "PDNO": "",
             "INQR_DVSN": "00", "CBLC_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-period-trade-profit", p, self._tr("pnl_stock"))
        return [{"date": r.get("trad_dt"), "code": r.get("pdno"), "name": r.get("prdt_name"), "buy_qty": _i(r.get("buy_qty")), "sell_qty": _i(r.get("sll_qty")),
                 "buy_amt": _i(r.get("buy_amt")), "sell_amt": _i(r.get("sll_amt")), "realized": _i(r.get("rlzt_pfls")), "rt": _f(r.get("pfls_rt")),
                 "fee": _i(r.get("fee")), "tax": _i(r.get("tl_tax"))} for r in j.get("output1") or []]

    def buying_power(self, code: str = "005930", price: int = 0) -> int:
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": code, "ORD_UNPR": str(price), "ORD_DVSN": "01" if not price else "00",
             "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N"}
        o = self._get("/uapi/domestic-stock/v1/trading/inquire-psbl-order", p, self._tr("psbl"))["output"]
        return int(o.get("ord_psbl_cash") or o.get("nrcvb_buy_amt") or 0)

    # ---- 주문
    def order(self, side: str, code: str, qty: int, price: int = 0) -> dict:
        """side: buy|sell. price=0이면 시장가(ORD_DVSN 01), 아니면 지정가(00)."""
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": code, "ORD_DVSN": "01" if not price else "00",
                "ORD_QTY": str(int(qty)), "ORD_UNPR": str(int(price))}
        if side == "sell":
            body["SLL_TYPE"] = "01"
        j = self._post("/uapi/domestic-stock/v1/trading/order-cash", body, self._tr(side))
        o = j.get("output") or {}
        return {"ok": True, "order_no": o.get("ODNO"), "org_no": o.get("KRX_FWDG_ORD_ORGNO"), "time": o.get("ORD_TMD"), "msg": j.get("msg1")}

    def pending(self) -> list[dict]:
        """미체결 주문(정정·취소 가능 수량 기준). 모의투자는 미지원."""
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "CTX_AREA_FK100": "", "CTX_AREA_NK100": "", "INQR_DVSN_1": "0", "INQR_DVSN_2": "0"}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl", p, self._tr("pending"))
        return [{"code": r["pdno"], "name": r.get("prdt_name"), "side": "buy" if r["sll_buy_dvsn_cd"] == "02" else "sell",
                 "qty": int(r.get("psbl_qty") or 0), "price": int(float(r.get("ord_unpr") or 0)), "order_no": r["odno"],
                 "org_no": r.get("ord_gno_brno"), "time": r.get("ord_tmd")} for r in j.get("output") or [] if int(r.get("psbl_qty") or 0) > 0]

    def cancel(self, order_no: str, org_no: str, qty: int = 0) -> dict:
        """미체결 취소(qty=0이면 잔량 전부)."""
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "KRX_FWDG_ORD_ORGNO": org_no or "", "ORGN_ODNO": order_no,
                "ORD_DVSN": "00", "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(int(qty)), "ORD_UNPR": "0", "QTY_ALL_ORD_YN": "Y" if not qty else "N"}
        j = self._post("/uapi/domestic-stock/v1/trading/order-rvsecncl", body, self._tr("cancel"))
        return {"ok": True, "msg": j.get("msg1")}

    def orders(self, day: str, filled_only: bool = False) -> dict:
        """당일 주문·체결 내역(YYYYMMDD). 주문별 주문수량·체결수량·잔량·취소 여부. 연속 조회 포함.
        반환 {"orders": [...], "total": {buy_amt, sell_amt, buy_qty, sell_qty}}"""
        out, fk, nk, total = [], "", "", {}
        for _ in range(10):
            p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": day, "INQR_END_DT": day, "SLL_BUY_DVSN_CD": "00",
                 "INQR_DVSN": "00", "PDNO": "", "CCLD_DVSN": "01" if filled_only else "00", "ORD_GNO_BRNO": "", "ODNO": "", "INQR_DVSN_3": "00", "INQR_DVSN_1": "",
                 "CTX_AREA_FK100": fk, "CTX_AREA_NK100": nk}
            j = self._get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", p, self._tr("ccld"))
            for r in j.get("output1") or []:
                out.append({"code": r["pdno"], "name": r["prdt_name"], "side": "buy" if r["sll_buy_dvsn_cd"] == "02" else "sell",
                            "ord_qty": _i(r.get("ord_qty")), "qty": _i(r.get("tot_ccld_qty")), "rmn_qty": _i(r.get("rmn_qty")),
                            "ord_price": _i(r.get("ord_unpr")), "price": _f(r.get("avg_prvs")), "amount": _i(r.get("tot_ccld_amt")),
                            "kind": r.get("ord_dvsn_name"), "cancelled": (r.get("cncl_yn") or "") == "Y", "time": r.get("ord_tmd"),
                            "order_no": r.get("odno"), "org_no": r.get("ord_gno_brno"), "orig_no": r.get("orgn_odno")})
            o2 = j.get("output2") or {}
            if isinstance(o2, list):
                o2 = o2[0] if o2 else {}
            total = {"buy_amt": _i(o2.get("tot_ccld_amt")) if not total else total.get("buy_amt"), "buy_qty": _i(o2.get("tot_ccld_qty"))}
            fk, nk = j.get("ctx_area_fk100", "").strip(), j.get("ctx_area_nk100", "").strip()
            if not nk or (j.get("tr_cont") or "").strip() not in ("F", "M"):
                break
        tb = sum(o["amount"] for o in out if o["side"] == "buy"); ts = sum(o["amount"] for o in out if o["side"] == "sell")
        return {"orders": out, "total": {"buy_amt": tb, "sell_amt": ts, "buy_qty": sum(o["qty"] for o in out if o["side"] == "buy"),
                                         "sell_qty": sum(o["qty"] for o in out if o["side"] == "sell")}}

    def fills(self, day: str) -> list[dict]:
        """당일 체결 내역(체결 수량 > 0인 주문만)."""
        return [o for o in self.orders(day)["orders"] if o["qty"] > 0]

    def order_status(self, day: str, order_no: str) -> dict | None:
        """주문번호 하나의 체결 상태(ord_qty·qty·rmn_qty·cancelled)."""
        return next((o for o in self.orders(day)["orders"] if o.get("order_no") == order_no), None)

    def account_full(self, days: int = 30) -> dict:
        """계좌 종합: 잔고·보유·주문가능금액·오늘 주문/체결·최근 기간 일별 손익·종목별 매매손익·실현손익 잔고.
        실전 전용 조회는 실패해도 나머지는 반환(모의투자 대비)."""
        kst = time.time() + 9 * 3600
        today = time.strftime("%Y%m%d", time.gmtime(kst))
        start = time.strftime("%Y%m%d", time.gmtime(kst - 86400 * days))
        out = self.balance()
        out["at"], out["mode"], out["errors"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(kst)), self.mode, {}
        for name, fn in (("buying_power", lambda: self.buying_power()), ("today", lambda: self.orders(today)), ("pending", self.pending),
                         ("pnl_daily", lambda: self.pnl_daily(start, today)), ("pnl_stock", lambda: self.pnl_stock(start, today)),
                         ("realized", self.balance_rlz)):
            try:
                out[name] = fn()
            except Exception as e:  # noqa: BLE001
                out["errors"][name] = str(e)[:160]
            time.sleep(0.25)
        return out


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "check"
    k = KIS()
    if cmd == "check":
        print(f"[kis] mode={k.mode} account={k.cano}-{k.prdt}")
        print("[kis] price 005930:", k.price("005930"))
        b = k.balance()
        print(f"[kis] cash={b['cash']:,} d2_cash={b['d2_cash']:,} total={b['total']:,} holdings={len(b['holdings'])}")
        for h in b["holdings"]:
            print("   ", h)
        print(f"[kis] buying_power={k.buying_power():,}")
    elif cmd == "snapshot":  # 아침 재학습용 계좌 스냅샷(현금·평가·보유) → stock/data/kis_account.json
        from stock.common import DATA
        b = k.account_full(days=35)
        for name, err in (b.get("errors") or {}).items():
            print(f"[kis] {name} 조회 실패: {err}")
        # 최근 7일 체결 내역(날짜별) → 화면 '내 기록'에 자동 반영
        fills = {}
        for i in range(1, 8):
            d = time.strftime("%Y%m%d", time.gmtime(time.time() + 9 * 3600 - 86400 * i))
            if time.gmtime(time.time() + 9 * 3600 - 86400 * i).tm_wday >= 5:
                continue
            try:
                f = k.fills(d)
            except Exception as e:  # noqa: BLE001
                print(f"[kis] fills {d} 실패: {e}")
                continue
            if f:
                fills[d] = f
            time.sleep(0.3)
        b["fills"] = fills
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "kis_account.json").write_text(json.dumps(b, ensure_ascii=False))
        print(f"[kis] snapshot cash={b['cash']:,} total={b['total']:,} holdings={len(b['holdings'])}")
    elif cmd == "price":
        print(k.price(argv[1]))
        print(k.orderbook(argv[1]))
    elif cmd == "fills":
        print(json.dumps(k.fills(argv[1]), ensure_ascii=False, indent=1))
    elif cmd == "orders":
        print(json.dumps(k.orders(argv[1]), ensure_ascii=False, indent=1))
    elif cmd == "account":
        print(json.dumps(k.account_full(), ensure_ascii=False, indent=1))
    elif cmd == "pnl":
        print(json.dumps(k.pnl_daily(argv[1], argv[2]), ensure_ascii=False, indent=1))
        print(json.dumps(k.pnl_stock(argv[1], argv[2]), ensure_ascii=False, indent=1))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
