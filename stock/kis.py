"""한국투자증권 Open API(KIS Developers) 최소 연동: 토큰, 현재가·호가, 잔고·주문가능금액, 현금 주문, 체결 조회.

환경변수
  KIS_APP_KEY / KIS_APP_SECRET            실전투자 키
  KIS_PAPER_APP_KEY / KIS_PAPER_APP_SECRET 모의투자 키
  KIS_ACCOUNT                              계좌번호 "12345678-01"
  KIS_MODE                                 paper(기본) | real
  KIS_TOKEN_DIR                            토큰 캐시 폴더(토큰은 하루 유효, 분당 1회 발급 제한)

  python -m stock.kis check          # 연결 점검: 잔고·주문가능금액·삼성전자 현재가(읽기만)
  python -m stock.kis price 005930
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
    "ccld": ("TTTC8001R", "VTTC8001R"), "cancel": ("TTTC0013U", "VTTC0013U"),
}


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
        holds = [{"code": h["pdno"], "name": h["prdt_name"], "qty": int(h["hldg_qty"]), "avg": float(h["pchs_avg_pric"]),
                  "price": int(h["prpr"]), "eval": int(h["evlu_amt"]), "pnl": int(h["evlu_pfls_amt"])}
                 for h in j.get("output1") or [] if int(h.get("hldg_qty") or 0) > 0]
        return {"cash": int(o2.get("dnca_tot_amt") or 0), "d2_cash": int(o2.get("prvs_rcdl_excc_amt") or 0),
                "total": int(o2.get("tot_evlu_amt") or 0), "stock_eval": int(o2.get("scts_evlu_amt") or 0), "holdings": holds}

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

    def fills(self, day: str) -> list[dict]:
        """당일 체결 내역(YYYYMMDD)."""
        p = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": day, "INQR_END_DT": day, "SLL_BUY_DVSN_CD": "00",
             "INQR_DVSN": "00", "PDNO": "", "CCLD_DVSN": "01", "ORD_GNO_BRNO": "", "ODNO": "", "INQR_DVSN_3": "00", "INQR_DVSN_1": "",
             "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", p, self._tr("ccld"))
        return [{"code": r["pdno"], "name": r["prdt_name"], "side": "buy" if r["sll_buy_dvsn_cd"] == "02" else "sell",
                 "qty": int(r["tot_ccld_qty"] or 0), "price": float(r["avg_prvs"] or 0), "amount": int(r["tot_ccld_amt"] or 0),
                 "time": r.get("ord_tmd"), "order_no": r.get("odno")} for r in j.get("output1") or []]


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
    elif cmd == "price":
        print(k.price(argv[1]))
        print(k.orderbook(argv[1]))
    elif cmd == "fills":
        print(k.fills(argv[1]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
