"""공시·뉴스 제목 → 일별 종목 특징. 거래일 t의 특징은 t 15:30 이후~t+1 08:30 이전 공시도 포함(아침 모델 기준).

규칙: 공시/뉴스 시각이 t일 15:30 이전이면 t일 귀속, 그 이후(다음 거래일 08:30까지)는 다음 거래일 아침 정보이므로
'직전 거래일 t' 행에 귀속한다(모델이 t 종가 기준으로 t+1을 예측하므로 t+1 아침까지의 정보는 t행에 들어가야 함).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# 공시 제목 키워드 범주 (DART·KOSCOM 제목 기준)
DISC_CATS = {
    "d_supply": r"단일판매|공급계약",
    "d_bonus": r"무상증자",
    "d_rights": r"유상증자",
    "d_cb": r"전환사채|신주인수권부사채|교환사채|전환청구|전환가액",
    "d_owner": r"최대주주\s*변경|경영권|지분양수도|주식양수도",
    "d_mna": r"합병|분할|영업양수|영업양도|타법인\s*주식|자산양수|자산양도",
    "d_buyback": r"자기주식\s*취득|자사주",
    "d_dispose": r"자기주식\s*처분",
    "d_earn": r"영업\(잠정\)실적|잠정실적|매출액\s*또는\s*손익구조|실적",
    "d_capital_cut": r"감자",
    "d_halt": r"매매거래\s*정지|거래정지",
    "d_resume": r"매매거래\s*재개|거래재개",
    "d_warn": r"투자주의|투자경고|투자위험|단기과열|불성실공시|관리종목|상장폐지|상장적격성|실질심사",
    "d_alert_notice": r"(투자경고|투자위험|단기과열)종목\s*지정\s*예고",
    "d_alert_set": r"(투자경고|투자위험|단기과열)종목\s*지정(?!\s*예고)",
    "d_alert_release": r"(투자경고|투자위험|단기과열)종목\s*지정\s*해제",
    "d_caution": r"투자주의종목",
    "d_manage": r"관리종목\s*지정",
    "d_delist": r"상장폐지|상장적격성|실질심사|정리매매",
    "d_spike": r"현저한\s*시황변동|주가\s*급등|주가\s*급락",
    "d_inquiry": r"조회공시|현저한\s*시황변동|풍문",
    "d_lawsuit": r"소송|횡령|배임|회생|파산|부도|감사의견|의견거절",
    "d_pharma": r"임상|품목허가|FDA|승인|특허",
    "d_contract_cancel": r"계약\s*해지|해제",
    "d_limit": r"가격제한폭\s*확대요건",
    "d_shareholder": r"주주총회|주주명부",
    "d_dividend": r"배당",
    "d_split": r"액면분할|액면병합|주식분할|주식병합",
    "d_spac": r"스팩|기업인수목적",
    "d_pledge": r"담보|대여|채무보증",
    "d_lockup": r"보호예수|의무보유",
    "d_fairdisc": r"공정공시",
    "d_fundraise": r"자금조달|차입",
    "d_major": r"주요사항보고서",
}
# 뉴스 제목 키워드 범주
NEWS_CATS = {
    "n_feature": r"특징주",
    "n_limit_up": r"상한가|上限|급등|강세|신고가|폭등|치솟",
    "n_limit_dn": r"하한가|급락|약세|폭락|곤두박질|신저가",
    "n_contract": r"계약|수주|공급|납품",
    "n_mna": r"인수|합병|M&A|지분|투자유치|경영권",
    "n_bio": r"임상|FDA|허가|승인|신약|기술이전|라이선스",
    "n_theme": r"테마|관련주|수혜",
    "n_politics": r"대선|총선|대통령|후보|정치|대장주",
    "n_earn": r"실적|영업이익|흑자|적자|어닝",
    "n_bad": r"횡령|배임|소송|거래정지|상장폐지|감사의견|회생|압수수색|구속|조사|논란|사기|분식",
    "n_rights": r"유상증자|전환사채|CB|BW|감자",
    "n_bonus": r"무상증자",
    "n_foreign": r"외국인|기관|순매수|매수세",
    "n_treasury": r"자사주|배당",
    "n_ipo": r"상장|공모|IPO|따상",
    "n_ai": r"AI|인공지능|반도체|로봇|양자|원전|방산|우주|2차전지|바이오",
}
_DISC = {k: re.compile(v) for k, v in DISC_CATS.items()}
_NEWS = {k: re.compile(v, re.I) for k, v in NEWS_CATS.items()}


def _assign_trading_day(dt_str: pd.Series, trading_days: np.ndarray) -> pd.Series:
    """YYYYMMDDHHMM → 귀속 거래일(t행). 15:30 이전이면 그날(또는 직전 거래일), 이후면 그날(다음 거래일 아침 정보 → 그날 행)."""
    d = dt_str.str[:8].to_numpy()
    hm = dt_str.str[8:12].fillna("0000").to_numpy()
    # 거래일 배열에서 '해당 날짜 이하의 마지막 거래일' 찾기
    pos = np.searchsorted(trading_days, d, side="right") - 1
    pos = np.clip(pos, 0, len(trading_days) - 1)
    day = trading_days[pos]
    # 공시가 거래일 당일 15:30 이전 → 그날 장중 정보 = 그날 종가에 이미 반영. 그래도 '그날' 행에 둔다(다음날 예측에 유효).
    # 공시가 거래일이 아닌 날(주말)이거나 15:30 이후 → 직전 거래일 행(=다음 거래일 아침 정보).
    # 둘 다 결과적으로 '해당 날짜 이하의 마지막 거래일' 행이므로 동일 처리. 단 08:30 이전 공시는 전 거래일 행으로.
    early = (d == day) & (hm < "0830")
    prev_pos = np.clip(pos - 1, 0, len(trading_days) - 1)
    day = np.where(early, trading_days[prev_pos], day)
    return pd.Series(day, index=dt_str.index)


def build_text_features(panel_keys: pd.DataFrame, disclosures: pd.DataFrame | None,
                        news: pd.DataFrame | None) -> pd.DataFrame:
    """panel_keys: code,date(거래일). 반환: code,date + 텍스트 특징."""
    trading_days = np.sort(panel_keys["date"].unique())
    out = panel_keys[["code", "date"]].copy()

    if disclosures is not None and len(disclosures):
        d = disclosures.copy()
        d["date"] = _assign_trading_day(d["dt"], trading_days)
        for k, rx in _DISC.items():
            d[k] = d["title"].str.contains(rx).astype("int8")
        d["d_any"] = 1
        agg = d.groupby(["code", "date"])[list(_DISC) + ["d_any"]].sum().reset_index()
        out = out.merge(agg, on=["code", "date"], how="left")
        cols = list(_DISC) + ["d_any"]
        out[cols] = out[cols].fillna(0).astype("float32")
        # 최근 5·20일 공시 수, 마지막 중요 공시 이후 경과일
        out = out.sort_values(["code", "date"])
        g = out.groupby("code", sort=False)
        out["d_any5"] = g["d_any"].transform(lambda s: s.rolling(5, min_periods=1).sum()).astype("float32")
        out["d_any20"] = g["d_any"].transform(lambda s: s.rolling(20, min_periods=1).sum()).astype("float32")
        for k in ("d_supply", "d_bonus", "d_rights", "d_cb", "d_owner", "d_warn", "d_halt", "d_resume"):
            out[f"{k}20"] = g[k].transform(lambda s: s.rolling(20, min_periods=1).sum()).astype("float32")

    if news is not None and len(news):
        n = news.copy()
        n = n[n["dt"].str.len() >= 8]
        n["date"] = _assign_trading_day(n["dt"], trading_days)
        for k, rx in _NEWS.items():
            n[k] = n["title"].str.contains(rx).astype("int8")
        n["n_cnt"] = 1
        agg = n.groupby(["code", "date"])[list(_NEWS) + ["n_cnt"]].sum().reset_index()
        out = out.merge(agg, on=["code", "date"], how="left")
        cols = list(_NEWS) + ["n_cnt"]
        # 뉴스 수집 시작 이전은 결측(0이 아님)으로 두어야 하므로 종목별 첫 뉴스일 이후만 0 채움
        first = n.groupby("code")["date"].min()
        has = out["date"] >= out["code"].map(first).fillna("99999999")
        for c in cols:
            out[c] = np.where(has, out[c].fillna(0), np.nan).astype("float32")
        out = out.sort_values(["code", "date"])
        g = out.groupby("code", sort=False)
        out["n_cnt3"] = g["n_cnt"].transform(lambda s: s.rolling(3, min_periods=1).sum()).astype("float32")
        out["n_cnt20"] = g["n_cnt"].transform(lambda s: s.rolling(20, min_periods=1).mean()).astype("float32")
        out["n_surge"] = (out["n_cnt"] / (out["n_cnt20"] + 0.5)).astype("float32")
        # 시장 전체 뉴스 대비 점유율(그날 관심 집중도)
        tot = out.groupby("date")["n_cnt"].transform("sum")
        out["n_share"] = (out["n_cnt"] / tot.replace(0, np.nan)).astype("float32")
        out["n_share_rk"] = out.groupby("date")["n_cnt"].rank(pct=True).astype("float32")
    return out
