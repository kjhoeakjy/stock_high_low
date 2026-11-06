"""당일 예측 리포트 생성: docs/stock/latest.json, history/YYYYMMDD.json, 전일 예측 채점."""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from stock.common import DATA, OUT, http_json, pmap

TOP = 25

# 특징 → 한국어 근거 문구 (값 포맷 함수)
def _pct(v): return f"{v*100:+.1f}%"
def _x(v): return f"{v:.1f}배"
def _n(v): return f"{v:.0f}"

LABELS = {
    "ret1": ("전일 등락", _pct), "ret2": ("2일 등락", _pct), "ret3": ("3일 등락", _pct), "ret5": ("5일 등락", _pct),
    "ret10": ("10일 등락", _pct), "ret20": ("20일 등락", _pct), "ret60": ("60일 등락", _pct), "ret120": ("120일 등락", _pct),
    "gap": ("전일 시가 갭", _pct), "range": ("전일 변동폭", _pct), "clv": ("종가의 일중 위치", lambda v: f"{v*100:.0f}%"),
    "body": ("전일 몸통", _pct), "uwick": ("윗꼬리", _pct), "lwick": ("아랫꼬리", _pct), "hi_ret": ("전일 고가 등락", _pct),
    "lo_ret": ("전일 저가 등락", _pct),
    "is_up": ("전일 상한가 마감", None), "is_dn": ("전일 하한가 마감", None), "touch_up": ("전일 장중 상한가 터치 후 풀림", None),
    "touch_dn": ("전일 장중 하한가 터치", None), "one_bar_up": ("전일 점상(시가=상한가)", None),
    "up_streak": ("연속 상한가", lambda v: f"{v:.0f}일"), "dn_streak": ("연속 하한가", lambda v: f"{v:.0f}일"),
    "since_up": ("마지막 상한가 이후", lambda v: f"{v:.0f}일"), "since_dn": ("마지막 하한가 이후", lambda v: f"{v:.0f}일"),
    "n_up5": ("최근 5일 상한가 횟수", _n), "n_up20": ("최근 20일 상한가 횟수", _n), "n_up60": ("최근 60일 상한가 횟수", _n),
    "n_touch5": ("5일 내 상한가 터치", _n), "n_touch20": ("20일 내 상한가 터치", _n), "n_touch60": ("60일 내 상한가 터치", _n),
    "n_dn20": ("20일 내 하한가", _n), "big_up": ("전일 15% 이상 급등", None), "n_big5": ("5일 내 15%↑ 급등 일수", _n),
    "n_big20": ("20일 내 15%↑ 급등 일수", _n), "updays5": ("5일 중 상승일", _n),
    "ma5": ("5일선 이격", _pct), "ma20": ("20일선 이격", _pct), "ma60": ("60일선 이격", _pct), "ma120": ("120일선 이격", _pct),
    "ma_align": ("정배열", None), "hi20": ("20일 고점 대비", _pct), "hi60": ("60일 고점 대비", _pct),
    "hi250": ("52주 고점 대비", _pct), "lo250": ("52주 저점 대비", _pct), "vol20": ("20일 변동성", _pct), "vol60": ("60일 변동성", _pct),
    "atr14": ("14일 평균 변동폭", _pct), "r20max": ("20일 내 최대 일등락", _pct), "r20min": ("20일 내 최저 일등락", _pct),
    "ret15": ("15일 등락", _pct), "alert3": ("3일 +100% (경고 예고 조건)", None), "alert5": ("5일 +60% (경고 예고 조건)", None),
    "alert15": ("15일 +100% (경고 예고 조건)", None), "heat40": ("40일 평균가 대비", _pct),
    "logval": ("거래대금", lambda v: f"{np.exp(v)/1e8:,.0f}억"), "vr5": ("거래량/5일 평균", _x), "vr20": ("거래량/20일 평균", _x),
    "vr60": ("거래량/60일 평균", _x), "vr_prev": ("전전일/전일 거래량", _x), "turn": ("회전율", _pct), "turn5": ("5일 평균 회전율", _pct),
    "turn2_40": ("2일 회전율/40일 (과열 조건)", _x), "vz20": ("거래량 z점수", lambda v: f"{v:.1f}"), "val20": ("20일 평균 거래대금", lambda v: f"{np.exp(v)/1e8:,.0f}억"),
    "zero_vol": ("거래 없음", None),
    "logmcap": ("시가총액", lambda v: f"{np.exp(v)/1e8:,.0f}억"), "logp": ("주가", lambda v: f"{np.exp(v):,.0f}원"), "penny": ("1,000원 미만 동전주", None),
    "frate": ("외국인 보유율", lambda v: f"{v:.1f}%"), "frate_d5": ("외국인 보유율 5일 변화", lambda v: f"{v:+.2f}%p"),
    "frate_d20": ("외국인 보유율 20일 변화", lambda v: f"{v:+.2f}%p"),
    "frgn1": ("외국인 순매수(시총 대비)", _pct), "frgn5": ("외국인 5일 순매수", _pct), "frgn20": ("외국인 20일 순매수", _pct),
    "inst1": ("기관 순매수(시총 대비)", _pct), "inst5": ("기관 5일 순매수", _pct), "inst20": ("기관 20일 순매수", _pct),
    "indiv1": ("개인 순매수(시총 대비)", _pct), "indiv5": ("개인 5일 순매수", _pct), "indiv20": ("개인 20일 순매수", _pct),
    "flow_flip": ("개인·외국인 매매 방향", lambda v: "같음" if v > 0 else "반대"),
    "m_nup": ("전일 시장 전체 상한가 수", _n), "m_ndn": ("전일 시장 전체 하한가 수", _n), "m_up_ratio": ("전일 상승 종목 비율", lambda v: f"{v*100:.0f}%"),
    "m_big": ("전일 15%↑ 급등 종목 수", _n), "m_streak": ("시장 최장 연속 상한가", _n), "m_ret_med": ("전일 종목 등락 중앙값", _pct),
    "m_nup5": ("5일 평균 상한가 수", lambda v: f"{v:.1f}"), "m_nup20": ("20일 평균 상한가 수", lambda v: f"{v:.1f}"),
    "ind_ret": ("업종 전일 등락", _pct), "ind_up": ("업종 내 상한가 수", _n), "ind_big": ("업종 내 15%↑ 급등 수", _n), "ind_n": ("업종 종목 수", _n),
    "ind_rel": ("업종 대비 초과 등락", _pct), "ind_up5": ("업종 5일 상한가 수", _n),
    "thm_ret": ("가장 뜨거운 소속 테마 전일 등락", _pct), "thm_up": ("테마 내 상한가 수", _n), "thm_big": ("테마 내 15%↑ 급등 수", _n),
    "thm_n": ("테마 종목 수", _n), "thm_rel": ("테마 대비 초과 등락", _pct), "thm_up5": ("테마 5일 상한가 수", _n),
    "KOSPI_r1": ("코스피 전일", _pct), "KOSPI_r5": ("코스피 5일", _pct), "KOSPI_r20": ("코스피 20일", _pct), "KOSDAQ_r1": ("코스닥 전일", _pct),
    "KOSDAQ_r5": ("코스닥 5일", _pct), "KOSDAQ_r20": ("코스닥 20일", _pct), "KOSDAQ_vol20": ("코스닥 변동성", _pct), "KOSPI_vol20": ("코스피 변동성", _pct),
    "KOSDAQ_ma250": ("코스닥 250일선 이격", _pct), "KOSPI_ma250": ("코스피 250일선 이격", _pct), "KOSDAQ_ma20": ("코스닥 20일선 이격", _pct),
    "KOSPI_ma20": ("코스피 20일선 이격", _pct), "KOSDAQ_dd": ("코스닥 고점 대비", _pct), "KOSPI_dd": ("코스피 고점 대비", _pct),
    "NASDAQ_r1": ("나스닥 밤사이", _pct), "NASDAQ_r5": ("나스닥 5일", _pct), "NASDAQ_r20": ("나스닥 20일", _pct), "SP500_r1": ("S&P500 밤사이", _pct),
    "SP500_r5": ("S&P500 5일", _pct), "SP500_r20": ("S&P500 20일", _pct), "SOX_r1": ("필라델피아 반도체 밤사이", _pct), "SOX_r5": ("반도체지수 5일", _pct),
    "SOX_r20": ("반도체지수 20일", _pct), "VIX_r1": ("VIX 변화", _pct), "VIX_r5": ("VIX 5일", _pct), "VIX_r20": ("VIX 20일", _pct),
    "VIX_lvl": ("VIX", lambda v: f"{v:.0f}"), "USDKRW_r1": ("원달러 환율", _pct), "USDKRW_r5": ("환율 5일", _pct), "USDKRW_r20": ("환율 20일", _pct),
    "RUSSELL_r1": ("러셀2000 밤사이", _pct), "RUSSELL_r5": ("러셀2000 5일", _pct), "RUSSELL_r20": ("러셀2000 20일", _pct),
    "WTI_r1": ("유가", _pct), "WTI_r5": ("유가 5일", _pct), "WTI_r20": ("유가 20일", _pct), "GOLD_r1": ("금", _pct), "GOLD_r5": ("금 5일", _pct), "GOLD_r20": ("금 20일", _pct),
    "US10Y_r1": ("미 10년물 금리", _pct), "US10Y_r5": ("미 금리 5일", _pct), "US10Y_r20": ("미 금리 20일", _pct), "BTC_r1": ("비트코인", _pct), "BTC_r5": ("비트코인 5일", _pct),
    "BTC_r20": ("비트코인 20일", _pct), "NIKKEI_r1": ("니케이", _pct), "NIKKEI_r5": ("니케이 5일", _pct), "NIKKEI_r20": ("니케이 20일", _pct),
    "SHANGHAI_r1": ("상하이", _pct), "SHANGHAI_r5": ("상하이 5일", _pct), "SHANGHAI_r20": ("상하이 20일", _pct), "HANGSENG_r1": ("항셍", _pct),
    "HANGSENG_r5": ("항셍 5일", _pct), "HANGSENG_r20": ("항셍 20일", _pct), "EWY_r1": ("한국 ETF(EWY) 밤사이", _pct), "EWY_r5": ("EWY 5일", _pct),
    "EWY_r20": ("EWY 20일", _pct), "NVDA_r1": ("엔비디아 밤사이", _pct), "NVDA_r5": ("엔비디아 5일", _pct), "NVDA_r20": ("엔비디아 20일", _pct),
    "is_pref": ("우선주", None), "age": ("상장 후 거래일", _n), "dte": ("옵션 만기까지", lambda v: f"{v:.0f}일"), "dom": ("일", _n),
    "d_alert_notice": ("투자경고·단기과열 지정 예고 공시", _n), "d_alert_set": ("투자경고·위험·과열 지정 공시", _n), "d_alert_release": ("지정 해제 공시", _n),
    "d_caution": ("투자주의 지정", _n), "d_manage": ("관리종목 지정 공시", _n), "d_delist": ("상장폐지·실질심사 관련 공시", _n), "d_spike": ("시황변동 조회공시", _n),
    "d_supply": ("공급계약 공시", _n), "d_bonus": ("무상증자 공시", _n), "d_rights": ("유상증자 공시", _n), "d_cb": ("전환사채 관련 공시", _n),
    "d_owner": ("최대주주 변경 공시", _n), "d_mna": ("합병·양수도 공시", _n), "d_buyback": ("자사주 취득 공시", _n), "d_dispose": ("자사주 처분 공시", _n),
    "d_earn": ("실적 공시", _n), "d_capital_cut": ("감자 공시", _n), "d_halt": ("거래정지 공시", _n), "d_resume": ("거래재개 공시", _n), "d_warn": ("시장경보 관련 공시", _n),
    "d_inquiry": ("조회공시", _n), "d_lawsuit": ("소송·횡령·회생 공시", _n), "d_pharma": ("임상·허가·특허 공시", _n), "d_contract_cancel": ("계약 해지 공시", _n),
    "d_limit": ("주식선물 가격제한폭 확대 공시", _n), "d_shareholder": ("주주총회 공시", _n), "d_dividend": ("배당 공시", _n), "d_split": ("액면분할·병합 공시", _n),
    "d_spac": ("스팩 공시", _n), "d_pledge": ("담보·대여 공시", _n), "d_lockup": ("보호예수 공시", _n), "d_fairdisc": ("공정공시", _n), "d_fundraise": ("자금조달 공시", _n),
    "d_major": ("주요사항보고서", _n), "d_any": ("당일 공시 수", _n), "d_any5": ("5일 공시 수", _n), "d_any20": ("20일 공시 수", _n),
    "d_supply20": ("20일 공급계약 공시", _n), "d_bonus20": ("20일 무상증자 공시", _n), "d_rights20": ("20일 유상증자 공시", _n), "d_cb20": ("20일 전환사채 공시", _n),
    "d_owner20": ("20일 최대주주 변경 공시", _n), "d_warn20": ("20일 시장경보 공시", _n), "d_halt20": ("20일 거래정지 공시", _n), "d_resume20": ("20일 거래재개 공시", _n),
    "n_feature": ("'특징주' 뉴스", _n), "n_limit_up": ("급등·상한가 뉴스", _n), "n_limit_dn": ("급락·하한가 뉴스", _n), "n_contract": ("계약·수주 뉴스", _n),
    "n_mna": ("인수·합병·지분 뉴스", _n), "n_bio": ("임상·허가 뉴스", _n), "n_theme": ("테마·관련주 뉴스", _n), "n_politics": ("정치 테마 뉴스", _n),
    "n_earn": ("실적 뉴스", _n), "n_bad": ("횡령·소송·조사 뉴스", _n), "n_rights": ("유상증자·CB 뉴스", _n), "n_bonus": ("무상증자 뉴스", _n),
    "n_foreign": ("수급 뉴스", _n), "n_treasury": ("자사주·배당 뉴스", _n), "n_ipo": ("상장·공모 뉴스", _n), "n_ai": ("AI·반도체 등 테마 뉴스", _n),
    "n_cnt": ("당일 뉴스 수", _n), "n_cnt3": ("3일 뉴스 수", _n), "n_cnt20": ("20일 평균 뉴스 수", lambda v: f"{v:.1f}"), "n_surge": ("뉴스 급증 배수", _x),
    "n_share": ("시장 뉴스 점유율", lambda v: f"{v*100:.2f}%"), "n_share_rk": ("뉴스 수 순위", lambda v: f"상위 {(1-v)*100:.0f}%"), "dow": ("요일", lambda v: "월화수목금토일"[int(v)]), "month": ("월", _n), "short_ban": ("공매도 금지 기간", None), "market_kq": ("코스닥", None),
    "rk_ret1": ("전일 등락 순위", lambda v: f"상위 {(1-v)*100:.0f}%"), "rk_vr20": ("거래량 배수 순위", lambda v: f"상위 {(1-v)*100:.0f}%"),
    "rk_turn": ("회전율 순위", lambda v: f"상위 {(1-v)*100:.0f}%"), "rk_logmcap": ("시총 순위", lambda v: f"상위 {(1-v)*100:.0f}%"),
    "rk_logval": ("거래대금 순위", lambda v: f"상위 {(1-v)*100:.0f}%"), "rk_ret5": ("5일 등락 순위", lambda v: f"상위 {(1-v)*100:.0f}%"),
}


def reasons(model, cols, row: pd.Series, k: int = 6) -> list[dict]:
    x = row[cols].to_numpy("float32").reshape(1, -1)
    contrib = model.predict(x, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(contrib))
    out = []
    for i in order:
        f = cols[i]
        v = row[f]
        if pd.isna(v) or abs(contrib[i]) < 0.02:
            continue
        label, fmt = LABELS.get(f, (f, lambda z: f"{z:.3g}"))
        if fmt is None:
            if not v:
                continue
            text = label
        else:
            text = f"{label} {fmt(float(v))}"
        out.append({"text": text, "dir": "+" if contrib[i] > 0 else "-", "w": round(float(contrib[i]), 3)})
        if len(out) >= k:
            break
    return out


def bucket(p: float, stats: dict) -> dict:
    for b in stats.get("buckets", []):
        if b["lo"] <= p < b["hi"]:
            return {"rate": b["rate"], "n": b["n"]}
    return {"rate": None, "n": 0}


def _evidence(code: str, name: str, dart: list[dict]) -> dict:
    ev = {"news": [], "disclosures": []}
    j = http_json(f"https://m.stock.naver.com/api/news/stock/{code}?pageSize=5&page=1")
    if isinstance(j, list):
        for grp in j:
            for it in grp.get("items", [])[:1]:
                ev["news"].append({"t": it.get("datetime", "")[:12], "title": it.get("titleFull") or it.get("title"),
                                   "url": it.get("mobileNewsUrl")})
    ev["news"] = ev["news"][:4]
    d = http_json(f"https://m.stock.naver.com/api/stock/{code}/disclosure?pageSize=4")
    if isinstance(d, list):
        ev["disclosures"] = [{"t": x.get("datetime", "")[:16], "title": x.get("title")} for x in d]
    key = re.sub(r"\s|\(주\)|주식회사", "", name)
    for x in dart:
        if re.sub(r"\s|\(주\)|주식회사", "", x["corp"]) == key:
            ev["disclosures"].append({"t": f'{x["date"]} {x["time"]}', "title": f'[DART] {x["tag"]}{x["title"]}'})
    ev["disclosures"] = ev["disclosures"][:6]
    return ev


def _load_json(name):
    try:
        return json.loads((DATA / name).read_text())
    except Exception:  # noqa: BLE001
        return None


def score_today(today: pd.DataFrame, res: dict, universe: list[dict]) -> dict:
    cols = res["features"]
    last_date = today["date"].max()
    today = today.copy()
    names = {u["code"]: u["name"] for u in universe}
    halted = {u["code"] for u in universe if (u.get("status") not in (None, "ok") and u.get("halt") == "HALTED")}
    status = _load_json("status_today.json") or {}
    over = _load_json("overnight.json") or {}
    halted |= {c for c, v in status.items() if v.get("halt")}
    today = today[(today["volume"] > 0) & ~today["code"].isin(halted)]
    out = {"as_of": last_date, "up": [], "n_scored": int(len(today))}
    try:
        dart = json.loads((DATA / "dart_today.json").read_text())
    except Exception:  # noqa: BLE001
        dart = []
    for target, key in (("y_up", "up"),):
        t = res["targets"][target]
        raw = t["model"].predict(today[cols].to_numpy("float32"))
        cal = t["iso"].predict(raw)
        today[f"p_{key}"] = cal.astype("float32")
        today[f"raw_{key}"] = raw.astype("float32")
        top = today.nlargest(TOP, f"p_{key}")
        items = []
        for _, r in top.iterrows():
            b = bucket(float(r[f"raw_{key}"]), t["stats"])
            items.append({
                "code": r["code"], "name": names.get(r["code"], r["code"]),
                "market": "코스닥" if r["market_kq"] else "코스피",
                "close": float(r["close"]), "ret1": float(r["ret1"]),
                "mcap": float(r["mcap"]) if pd.notna(r["mcap"]) else None,
                "p": round(float(r[f"p_{key}"]), 4), "raw": round(float(r[f"raw_{key}"]), 4),
                "hist_rate": b["rate"], "hist_n": b["n"],
                "halted": r["code"] in halted, "pref": bool(r.get("is_pref", 0)),
                "alert": (status.get(r["code"]) or {}).get("alert"), "managed": (status.get(r["code"]) or {}).get("managed"),
                "over_ret": (over.get(r["code"]) or {}).get("over_ret"), "over_status": (over.get(r["code"]) or {}).get("over_status"),
                "up_limit": (over.get(r["code"]) or {}).get("up_limit"), "dn_limit": (over.get(r["code"]) or {}).get("dn_limit"),
                "reasons": reasons(t["model"], cols, r),
            })
        ev = pmap(lambda it: _evidence(it["code"], it["name"], dart), items, workers=8, label=f"evidence-{key}")
        for it, e in zip(items, ev):
            it["evidence"] = e or {"news": [], "disclosures": []}
        try:
            from stock.llm import annotate
            annotate(items, key)
        except Exception as e:  # noqa: BLE001
            print(f"[report] llm skipped: {e}")
        out[key] = items
        out[f"expected_{key}"] = float(np.sum(cal))
    return out, today


def grade_history(actual: pd.DataFrame) -> list[dict]:
    """history/에 저장된 과거 예측을 실제 결과로 채점. actual: code,date,ret1,hi_ret,lo_ret"""
    hist_dir = OUT / "history"
    hist_dir.mkdir(exist_ok=True)
    graded = []
    for f in sorted(hist_dir.glob("*.json"))[-60:]:
        try:
            h = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        tgt = h.get("target_date")
        a = actual[actual["date"] == tgt].set_index("code") if tgt else None
        for key in ("up",):
            for it in h.get(key, []):
                if a is not None and it["code"] in a.index:
                    r = a.loc[it["code"]]
                    it["actual_ret"] = float(r["ret1"])
                    it["actual_gap"] = float(r["gap"]) if pd.notna(r["gap"]) else None
                    it["actual_oc"] = float((1 + r["ret1"]) / (1 + r["gap"]) - 1) if pd.notna(r["gap"]) else None
                    it["hit"] = bool(r["ret1"] >= 0.295)
                    it["touch"] = bool(r["hi_ret"] >= 0.295)
        if a is not None and len(a):
            h["graded"] = True
            h["actual_up_count"] = int((a["ret1"] >= 0.295).sum())
            f.write_text(json.dumps(h, ensure_ascii=False))
        graded.append({"target_date": tgt, "graded": h.get("graded", False),
                       "up_hits": sum(1 for it in h.get("up", [])[:10] if it.get("hit")),
                       "up_touch": sum(1 for it in h.get("up", [])[:10] if it.get("touch")),
                       "up_mean_ret": float(np.mean([it["actual_ret"] for it in h.get("up", [])[:10] if "actual_ret" in it])) if any("actual_ret" in it for it in h.get("up", [])[:10]) else None,
                       "up_mean_oc": float(np.mean([it["actual_oc"] for it in h.get("up", [])[:10] if it.get("actual_oc") is not None])) if any(it.get("actual_oc") is not None for it in h.get("up", [])[:10]) else None,
                       "actual_up_count": h.get("actual_up_count"),
                       "up": [{"name": it["name"], "p": it["p"], "ret": it.get("actual_ret"), "oc": it.get("actual_oc"), "hit": it.get("hit")} for it in h.get("up", [])[:10]]})
    return graded


HOLIDAYS = {
    # 2026
    "20260101", "20260216", "20260217", "20260218", "20260301", "20260302", "20260501", "20260505", "20260524", "20260525",
    "20260603", "20260606", "20260815", "20260817", "20260924", "20260925", "20260926", "20261003", "20261005", "20261009", "20261225", "20261231",
    # 2027
    "20270101", "20270206", "20270208", "20270209", "20270301", "20270505", "20270513", "20270603", "20270606", "20270607",
    "20270815", "20270816", "20270914", "20270915", "20270916", "20270917", "20271003", "20271004", "20271009", "20271011", "20271225", "20271227", "20271231",
}


def next_trading_day(as_of: str) -> str:
    d = dt.datetime.strptime(as_of, "%Y%m%d") + dt.timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y%m%d") in HOLIDAYS:
        d += dt.timedelta(days=1)
    return d.strftime("%Y%m%d")


def write(last: pd.DataFrame, actual: pd.DataFrame, res: dict, universe: list[dict], market: pd.DataFrame,
          n_train: int = 0, train_from: str = ""):
    """last: 마지막 거래일 행(특징 포함). actual: 전 기간 code,date,ret1,hi_ret,lo_ret."""
    today, scored = score_today(last, res, universe)
    target_date = next_trading_day(today["as_of"])
    stats = {k: {kk: vv for kk, vv in v["stats"].items()} for k, v in res["targets"].items()}
    imp = {k: v["importance"] for k, v in res["targets"].items()}
    # 시장 요약(화면 상단)
    mk = {}
    for sym in ("KOSPI", "KOSDAQ", "NASDAQ", "SOX", "USDKRW", "VIX"):
        g = market[market["symbol"] == sym].sort_values("date")
        if len(g) >= 2:
            mk[sym] = {"date": g["date"].iloc[-1], "close": float(g["close"].iloc[-1]),
                       "r1": float(g["close"].iloc[-1] / g["close"].iloc[-2] - 1)}
    mk["n_up_yesterday"] = int((last["ret1"] >= 0.295).sum())

    doc = {
        "generated_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M"),
        "as_of": today["as_of"], "target_date": target_date, "n_scored": today["n_scored"],
        "expected_up": round(today["expected_up"], 2),
        "up": today["up"], "market": mk,
        "stats": stats, "importance": imp, "n_features": len(res["features"]),
        "train_rows": n_train, "train_from": train_from,
    }
    (OUT / "history").mkdir(exist_ok=True)
    (OUT / "history" / f"{target_date}.json").write_text(json.dumps(doc, ensure_ascii=False))
    doc["history"] = grade_history(actual)
    (OUT / "latest.json").write_text(json.dumps(doc, ensure_ascii=False))
    # 전 종목 점수(검색용, 가벼운 버전)
    slim = scored[["code", "close", "ret1", "p_up"]].copy()
    names = {u["code"]: u["name"] for u in universe}
    slim["name"] = slim["code"].map(names)
    slim = slim.sort_values("p_up", ascending=False)
    (OUT / "scores.json").write_text(json.dumps(
        {"as_of": today["as_of"], "rows": [[r.code, r.name, float(r.close), round(float(r.ret1), 4), round(float(r.p_up), 4)]
                                          for r in slim.itertuples()]}, ensure_ascii=False))
    print(f"[report] {target_date}: up top {[(i['name'], i['p']) for i in doc['up'][:5]]}")
    return doc
