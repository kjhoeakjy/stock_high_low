"""특징·라벨 생성. 모든 특징은 거래일 t 종가 기준(t+1 아침에 알 수 있는 정보만)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stock.common import DATA

LIM = 0.295  # 상·하한가 판정 임계(수정주가 오차 허용)
STOP = 0.05  # 손절선: 매수가 대비 -5% 도달 시 즉시 매도(사용자 규칙)
STOP_SLIP = 0.01  # 손절 체결 미끄러짐: 급락 중 시장가 매도는 손절선보다 낮게 체결 → -6%로 계산(보수 가정)


# ---------------------------------------------------------------- helpers
def _g(df, col):
    return df.groupby("code", sort=False)[col]


def _roll(df, col, n, fn="mean", minp=None):
    r = _g(df, col).rolling(n, min_periods=minp or max(2, n // 2))
    out = getattr(r, fn)()
    return out.reset_index(level=0, drop=True).astype("float32")


def _streak(flag: pd.Series, codes: pd.Series) -> np.ndarray:
    """연속 True 길이(종목별)."""
    f = flag.to_numpy(bool)
    c = codes.to_numpy()
    out = np.zeros(len(f), dtype=np.int16)
    run = 0
    prev = None
    for i in range(len(f)):
        if c[i] != prev:
            run = 0
            prev = c[i]
        run = run + 1 if f[i] else 0
        out[i] = run
    return out


def _days_since(flag: pd.Series, codes: pd.Series, cap=250) -> np.ndarray:
    f = flag.to_numpy(bool)
    c = codes.to_numpy()
    out = np.full(len(f), cap, dtype=np.int16)
    last = -10**9
    prev = None
    for i in range(len(f)):
        if c[i] != prev:
            last = -10**9
            prev = c[i]
        if f[i]:
            last = i
        out[i] = min(cap, i - last) if last > -10**8 else cap
    return out


# ---------------------------------------------------------------- market features
def market_features(mk: pd.DataFrame) -> pd.DataFrame:
    """symbol별 수익률 -> date 인덱스 wide 프레임(미국 지수는 해당 미국 날짜 = 한국 다음날 아침 정보)."""
    out = {}
    for sym, g in mk.groupby("symbol"):
        g = g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        c = g["close"].astype(float)
        r1 = c.pct_change()
        d = pd.DataFrame({f"{sym}_r1": r1, f"{sym}_r5": c.pct_change(5), f"{sym}_r20": c.pct_change(20)})
        if sym in ("KOSPI", "KOSDAQ"):
            d[f"{sym}_vol20"] = r1.rolling(20).std()
            d[f"{sym}_ma250"] = c / c.rolling(250, min_periods=120).mean() - 1
            d[f"{sym}_ma20"] = c / c.rolling(20).mean() - 1
            d[f"{sym}_dd"] = c / c.rolling(250, min_periods=60).max() - 1
            d[f"{sym}_r60"] = c.pct_change(60)
            d[f"{sym}_bull"] = (c > c.rolling(120, min_periods=60).mean()).astype("float32")  # 강세 국면 플래그
        if sym == "VIX":
            d["VIX_lvl"] = c
        out[sym] = d
    wide = pd.concat(out.values(), axis=1).sort_index()
    # 날짜 축: 전 영업일 캘린더(한국 날짜 기준). 미국 지수는 같은 달력일(=다음날 아침 정보)로 두고 결측은 직전값.
    return wide.ffill(limit=5).astype("float32")


# ---------------------------------------------------------------- build
def build(prices: pd.DataFrame, flows: pd.DataFrame | None, universe: list[dict],
          groups: dict, market: pd.DataFrame, start: str = "20190101",
          disclosures: pd.DataFrame | None = None, news: pd.DataFrame | None = None, board: pd.DataFrame | None = None) -> pd.DataFrame:
    df = prices[prices["date"] >= start].sort_values(["code", "date"]).reset_index(drop=True).copy()
    # 장중·장 마감 직후 실행이면 당일 행은 미완성이므로 제외(16:00 KST 이전)
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=9)
    if df["date"].max() == now.strftime("%Y%m%d") and now.hour < 16:
        df = df[df["date"] < now.strftime("%Y%m%d")].reset_index(drop=True)
        print(f"[features] dropped partial day {now:%Y%m%d}")
    uni = pd.DataFrame(universe).drop_duplicates("code").set_index("code")
    names = df["code"].map(uni["name"]).fillna("")
    df["is_pref"] = (names.str.endswith("우") | names.str.contains(r"우B$|우C$|\(전환\)") | df["code"].str[-1].isin(list("579K"))).astype("int8")
    df["market_kq"] = df["code"].map(uni["market"]).eq("KOSDAQ").astype("int8")
    shares = df["code"].map(uni["shares"]).astype("float64")

    c, o, h, l, v = (df[k].astype("float64") for k in ("close", "open", "high", "low", "volume"))
    pc = _g(df, "close").shift(1)
    df["ret1"] = (c / pc - 1).astype("float32")
    for n in (2, 3, 5, 10, 20, 60, 120):
        df[f"ret{n}"] = (c / _g(df, "close").shift(n) - 1).astype("float32")
    df["gap"] = (o / pc - 1).astype("float32")
    rng = (h - l)
    df["range"] = (rng / pc).astype("float32")
    df["clv"] = ((c - l) / rng.replace(0, np.nan)).astype("float32")
    df["body"] = ((c - o) / pc).astype("float32")
    df["uwick"] = ((h - np.maximum(c, o)) / pc).astype("float32")
    df["lwick"] = ((np.minimum(c, o) - l) / pc).astype("float32")
    df["hi_ret"] = (h / pc - 1).astype("float32")
    df["lo_ret"] = (l / pc - 1).astype("float32")

    # 상·하한가 상태
    up = df["ret1"] >= LIM
    dn = df["ret1"] <= -LIM
    df["is_up"] = up.astype("int8")
    df["is_dn"] = dn.astype("int8")
    df["touch_up"] = ((df["hi_ret"] >= LIM) & ~up).astype("int8")  # 장중 도달 후 풀림
    df["touch_dn"] = ((df["lo_ret"] <= -LIM) & ~dn).astype("int8")
    df["one_bar_up"] = (up & (o == c) & (h == l)).astype("int8")  # 점상
    df["up_streak"] = _streak(up, df["code"])
    df["dn_streak"] = _streak(dn, df["code"])
    df["since_up"] = _days_since(up, df["code"])
    df["since_dn"] = _days_since(dn, df["code"])
    for n in (5, 20, 60):
        df[f"n_up{n}"] = _roll(df, "is_up", n, "sum", 1)
        df[f"n_touch{n}"] = _roll(df, "touch_up", n, "sum", 1)
    df["n_dn20"] = _roll(df, "is_dn", 20, "sum", 1)
    # 상한가일을 제외한 모멘텀(Liu·Wu·Zhu 2022: 상한가일 과잉반응을 뺀 모멘텀이 유효)
    lr = np.log1p(df["ret1"].clip(-0.99, None)).where(~up, 0.0).fillna(0.0)
    df["_lr_ex"] = lr.astype("float64")
    df["ret20_ex"] = np.expm1(_roll(df, "_lr_ex", 20, "sum", 1)).astype("float32")
    df["ret60_ex"] = np.expm1(_roll(df, "_lr_ex", 60, "sum", 1)).astype("float32")
    # 상한가 잠김 강도: 상한가일 거래량/20일 평균(낮을수록 잠김이 단단함), 상한가일 종가 위치
    df["big_up"] = (df["ret1"] >= 0.15).astype("int8")
    df["n_big5"] = _roll(df, "big_up", 5, "sum", 1)
    df["n_big20"] = _roll(df, "big_up", 20, "sum", 1)
    df["updays5"] = _roll(df, "ret1", 5, "apply", 1) if False else _roll(df.assign(_u=(df["ret1"] > 0).astype("int8")), "_u", 5, "sum", 1)

    # 이동평균·고점 거리·변동성
    for n in (5, 20, 60, 120):
        df[f"ma{n}"] = (c / _roll(df, "close", n) - 1).astype("float32")
    df["ma_align"] = ((df["ma5"] > 0) & (df["ma20"] > 0) & (df["ma60"] > 0)).astype("int8")
    df["hi20"] = (c / _roll(df, "high", 20, "max") - 1).astype("float32")
    df["hi60"] = (c / _roll(df, "high", 60, "max") - 1).astype("float32")
    df["hi250"] = (c / _roll(df, "high", 250, "max", 60) - 1).astype("float32")
    df["lo250"] = (c / _roll(df, "low", 250, "min", 60) - 1).astype("float32")
    df["vol20"] = _roll(df, "ret1", 20, "std")
    df["vol60"] = _roll(df, "ret1", 60, "std")
    df["atr14"] = (_roll(df, "range", 14) ).astype("float32")
    df["r20max"] = _roll(df, "ret1", 20, "max")
    df["r20min"] = _roll(df, "ret1", 20, "min")
    # 복권형·관심 변수(문헌: MAX 효과, 상·하위 20위 출현 횟수, 전일 등락 제곱)
    df["r5max"] = _roll(df, "ret1", 5, "max", 2)
    df["r20q80"] = _g(df, "ret1").rolling(20, min_periods=10).quantile(0.8).reset_index(level=0, drop=True).astype("float32")
    df["ret1_sq"] = (df["ret1"] ** 2).astype("float32")
    drk = df.groupby("date")["ret1"].rank(ascending=False, method="min")
    df["_top20"] = (drk <= 20).astype("int8")
    df["_bot20"] = (df.groupby("date")["ret1"].rank(ascending=True, method="min") <= 20).astype("int8")
    df["n_top20_20"] = _roll(df, "_top20", 20, "sum", 1)
    df["n_bot20_20"] = _roll(df, "_bot20", 20, "sum", 1)
    df["n_tb20_20"] = (df["n_top20_20"] + df["n_bot20_20"]).astype("float32")
    # KRX 시장경보 근사 조건
    df["ret15"] = (c / _g(df, "close").shift(15) - 1).astype("float32")
    df["alert3"] = (df["ret3"] >= 1.0).astype("int8")
    df["alert5"] = (df["ret5"] >= 0.6).astype("int8")
    df["alert15"] = (df["ret15"] >= 1.0).astype("int8")
    df["heat40"] = (c / _roll(df, "close", 40) - 1).astype("float32")  # 단기과열 종가 조건(≥0.3)

    # 거래량·회전율·거래대금
    val = (c * v)
    df["logval"] = np.log1p(val).astype("float32")
    df["_v"] = v.astype("float64")
    for n in (5, 20, 60):
        df[f"vr{n}"] = (v / _roll(df, "_v", n).replace(0, np.nan)).astype("float32")
    df["vr_prev"] = (_g(df, "_v").shift(1) / v.replace(0, np.nan)).astype("float32")
    df["turn"] = (v / shares).astype("float32")
    df["turn5"] = _roll(df, "turn", 5)
    df["turn2_40"] = (_roll(df, "turn", 2, "mean", 1) / _roll(df, "turn", 40).replace(0, np.nan)).astype("float32")
    df["vz20"] = ((v - _roll(df, "_v", 20)) / _roll(df, "_v", 20, "std").replace(0, np.nan)).astype("float32")
    df["vz_small"] = (df["vz20"] * (c * shares < 2e11)).astype("float32")  # 소형주 거래량 충격(문헌: 1~2일 반전)
    df["val20"] = _roll(df, "logval", 20)
    df["zero_vol"] = (v == 0).astype("int8")

    # 규모·가격
    df["mcap"] = (c * shares).astype("float64")
    df["logmcap"] = np.log(df["mcap"].replace(0, np.nan)).astype("float32")
    df["logp"] = np.log(c).astype("float32")
    df["penny"] = (c < 1000).astype("int8")
    df["frate"] = df["frate"].astype("float32")
    df["frate_d5"] = (df["frate"] - _g(df, "frate").shift(5)).astype("float32")
    df["frate_d20"] = (df["frate"] - _g(df, "frate").shift(20)).astype("float32")

    # 수급
    if flows is not None and len(flows):
        f = flows[["code", "date", "frgn", "inst", "indiv"]].copy()
        df = df.merge(f, on=["code", "date"], how="left")
        # 개인 매매 강도 대용(개인 순매수 수량 절대값/거래량, 20일 평균) — 거래 비중 자료가 없어 대용 변수
        df["_indiv_abs"] = (df["indiv"].abs() / v.replace(0, np.nan)).clip(upper=1.0).fillna(0).astype("float32")
        df["indiv_act20"] = _roll(df, "_indiv_abs", 20, "mean", 5)
        df["indiv_act5"] = _roll(df, "_indiv_abs", 5, "mean", 2)
        for k in ("frgn", "inst", "indiv"):
            x = (df[k] * c / df["mcap"].replace(0, np.nan)).astype("float32")  # 시총 대비 순매수
            df[f"{k}1"] = x
            df[f"_{k}"] = x.fillna(0)
            df[f"{k}5"] = _roll(df, f"_{k}", 5, "sum", 1)
            df[f"{k}20"] = _roll(df, f"_{k}", 20, "sum", 1)
            df.drop(columns=[k, f"_{k}"], inplace=True)
        df["flow_flip"] = (np.sign(df["indiv1"].fillna(0)) * np.sign(df["frgn1"].fillna(0))).astype("float32")

    # 시장 전체(날짜별 집계) — 당일 상한가 수·등락 종목 비율
    day = df.groupby("date").agg(m_nup=("is_up", "sum"), m_ndn=("is_dn", "sum"),
                                 m_up_ratio=("ret1", lambda s: float((s > 0).mean())),
                                 m_big=("big_up", "sum"), m_streak=("up_streak", "max"),
                                 m_ret_med=("ret1", "median"))
    day["m_nup5"] = day["m_nup"].rolling(5, min_periods=1).mean()
    day["m_nup20"] = day["m_nup"].rolling(20, min_periods=1).mean()
    df = df.merge(day.astype("float32").reset_index(), on="date", how="left")

    # 업종·테마 열기
    df = _group_feats(df, groups, "industry", "ind")
    df = _group_feats(df, groups, "theme", "thm")

    # 해외·지수
    mkf = market_features(market)
    df = df.merge(mkf, left_on="date", right_index=True, how="left")

    # 달력·제도
    d = pd.to_datetime(df["date"], format="%Y%m%d")
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["dom"] = d.dt.day.astype("int8")
    # 상장(데이터 시작) 후 경과 거래일: 신규상장 직후 변동성
    df["age"] = df.groupby("code", sort=False).cumcount().clip(upper=2000).astype("int16")
    # 옵션 만기(둘째 목요일)까지 남은 일수
    first = d.values.astype("datetime64[M]").astype("datetime64[D]")
    first_dow = (pd.DatetimeIndex(first).dayofweek).to_numpy()
    second_thu = first + np.timedelta64(1, "D") * (((3 - first_dow) % 7) + 7)
    dte = (second_thu - d.values.astype("datetime64[D]")).astype("timedelta64[D]").astype(int)
    nxt_first = (d.values.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
    nf_dow = pd.DatetimeIndex(nxt_first).dayofweek.to_numpy()
    nxt_thu = nxt_first + np.timedelta64(1, "D") * (((3 - nf_dow) % 7) + 7)
    dte2 = (nxt_thu - d.values.astype("datetime64[D]")).astype("timedelta64[D]").astype(int)
    df["dte"] = np.where(dte >= 0, dte, dte2).astype("int8")
    df["short_ban"] = (((df["date"] >= "20200316") & (df["date"] <= "20210502")) |
                       ((df["date"] >= "20231106") & (df["date"] <= "20250330"))).astype("int8")

    # 공시·뉴스 제목 특징
    if (disclosures is not None and len(disclosures)) or (news is not None and len(news)):
        from stock.text_features import build_text_features
        tf = build_text_features(df[["code", "date"]], disclosures, news)
        df = df.merge(tf, on=["code", "date"], how="left")
    if board is not None and len(board):
        from stock.text_features import build_board_features
        bf = build_board_features(df[["code", "date"]], board)
        df = df.merge(bf, on=["code", "date"], how="left")
        # 투자경고·위험 지정 공시 후 경과일(문헌: 지정 후 3일 고점, 20일 하락)
        if "d_alert_set" in df.columns:
            df["since_alert"] = _days_since(df["d_alert_set"].fillna(0) > 0, df["code"], cap=60).astype("int16")
            df["since_alert_notice"] = _days_since(df["d_alert_notice"].fillna(0) > 0, df["code"], cap=60).astype("int16")

    # 횡단면 순위(날짜별)
    for k in ("ret1", "vr20", "turn", "logmcap", "logval", "ret5"):
        df[f"rk_{k}"] = df.groupby("date")[k].rank(pct=True).astype("float32")

    # 라벨: 다음 거래일
    nxt = _g(df, "ret1").shift(-1)
    nxt_hi = _g(df, "hi_ret").shift(-1)
    nxt_lo = _g(df, "lo_ret").shift(-1)
    nxt_vol = _g(df, "volume").shift(-1)
    valid = nxt.notna() & (nxt.abs() < 0.31) & (nxt_vol > 0)
    df["y_up"] = np.where(valid, (nxt >= LIM).astype("int8"), -1).astype("int8")
    df["y_dn"] = np.where(valid, (nxt <= -LIM).astype("int8"), -1).astype("int8")
    df["y_touch_up"] = np.where(valid, (nxt_hi >= LIM).astype("int8"), -1).astype("int8")
    df["y_ret"] = nxt.astype("float32")
    nxt_gap = _g(df, "gap").shift(-1)
    df["y_gap"] = np.where(valid, nxt_gap, np.nan).astype("float32")          # 다음날 시가 갭
    df["y_oc"] = np.where(valid, (1 + nxt) / (1 + nxt_gap) - 1, np.nan).astype("float32")  # 시가→종가
    # 매도 규칙 수익(시가 매수 기준): 매수가 대비 -5% 도달 시 손절(STOP), 장중 상한가 도달 시 즉시 매도, 아니면 종가(15:15 근사).
    # 같은 날 손절선과 상한가가 모두 나오면 손절이 먼저라고 보수적으로 가정.
    nxt_lo = _g(df, "lo_ret").shift(-1)
    oc_close = (1 + nxt) / (1 + nxt_gap) - 1
    oc_lo = (1 + nxt_lo) / (1 + nxt_gap) - 1
    oc_hi = (1 + nxt_hi) / (1 + nxt_gap) - 1
    df["y_ocx"] = np.where(valid, np.where(oc_lo <= -STOP, -(STOP + STOP_SLIP), np.where(nxt_hi >= LIM, oc_hi, oc_close)), np.nan).astype("float32")
    df["next_date"] = _g(df, "date").shift(-1)

    df.drop(columns=["_v", "_lr_ex", "_top20", "_bot20", "_indiv_abs"], inplace=True, errors="ignore")
    return df


def _group_feats(df: pd.DataFrame, groups: dict, kind: str, pre: str) -> pd.DataFrame:
    glist = groups.get(kind) or []
    if not glist:
        return df
    mem = pd.DataFrame([(g["no"], c) for g in glist for c in g["codes"]], columns=["gno", "code"])
    if mem.empty:
        return df
    base = df[["code", "date", "ret1", "is_up", "big_up"]].merge(mem, on="code", how="inner")
    agg = base.groupby(["gno", "date"]).agg(gret=("ret1", "mean"), gup=("is_up", "sum"),
                                            gbig=("big_up", "sum"), gn=("ret1", "size")).reset_index()
    agg = agg[agg["gn"] >= 2]
    x = mem.merge(agg, on="gno").drop(columns="gno")
    # 종목이 여러 그룹에 속하면 가장 뜨거운 그룹 기준
    best = x.groupby(["code", "date"]).agg(**{f"{pre}_ret": ("gret", "max"), f"{pre}_up": ("gup", "max"),
                                             f"{pre}_big": ("gbig", "max"), f"{pre}_n": ("gn", "max")}).reset_index()
    df = df.merge(best, on=["code", "date"], how="left")
    # 그룹 내 당일 순위(가장 뜨거운 그룹 기준이 아닌 평균적 위치): 종목 ret1 - 그룹 평균
    df[f"{pre}_rel"] = (df["ret1"] - df[f"{pre}_ret"]).astype("float32")
    for k in (f"{pre}_ret", f"{pre}_up", f"{pre}_big", f"{pre}_n"):
        df[k] = df[k].astype("float32")
    # 그룹 열기의 5일 이동(테마 지속성)
    df[f"{pre}_up5"] = _roll(df.assign(_t=df[f"{pre}_up"].fillna(0)), "_t", 5, "sum", 1)
    return df


EXCLUDE = {"code", "date", "open", "high", "low", "close", "volume", "mcap", "y_up", "y_dn",
           "y_touch_up", "y_ret", "y_gap", "y_oc", "y_ocx", "next_date", "name"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    """특징 열: 키·가격 원자료·라벨(y_로 시작하는 모든 열) 제외. 라벨 열이 변수로 새어 들어가면 검증이 무의미해진다."""
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("y_") and df[c].dtype.kind in "fiu"]


def load_all():
    prices = pd.read_parquet(DATA / "prices.parquet")
    flows = pd.read_parquet(DATA / "flows.parquet") if (DATA / "flows.parquet").exists() else None
    universe = json.loads((DATA / "universe.json").read_text())
    groups = json.loads((DATA / "groups.json").read_text()) if (DATA / "groups.json").exists() else {}
    market = pd.read_parquet(DATA / "market.parquet")
    disc = pd.read_parquet(DATA / "disclosures.parquet") if (DATA / "disclosures.parquet").exists() else None
    news = pd.read_parquet(DATA / "news.parquet") if (DATA / "news.parquet").exists() else None
    board = pd.read_parquet(DATA / "board.parquet") if (DATA / "board.parquet").exists() else None
    return prices, flows, universe, groups, market, disc, news, board


if __name__ == "__main__":
    import time
    t0 = time.time()
    prices, flows, universe, groups, market, disc, news, board = load_all()
    df = build(prices, flows, universe, groups, market, disclosures=disc, news=news)
    print(df.shape, f"{time.time()-t0:.0f}s")
    print("positives up/dn:", (df.y_up == 1).sum(), (df.y_dn == 1).sum(), "valid", (df.y_up >= 0).sum())
    df.to_parquet(DATA / "panel.parquet", index=False)
