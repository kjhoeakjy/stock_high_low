"""학습·검증·예측. LightGBM 상한가 이진 모델 + 워크포워드 OOS 검증 + 등온 보정.

메모리: 특징 행렬을 float32 numpy로 한 번만 만들고 패널은 라벨·키만 남긴다.
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from stock.common import DATA
from stock.features import feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    max_bin=255, verbose=-1, num_threads=0, seed=7,
)
ROUNDS = 600
TRAIN_START = "20200101"
NEG_KEEP = 0.25  # 음성 다운샘플 비율(가중치로 보정)
HALF_LIFE = {"y_oc10": 24}  # 표본 가중 반감기(개월). 실험(2026-11-06, 6개월): 24개월 반감기가 최악일·낙폭을 줄임
N_SEEDS = {"y_up": 3, "y_oc10": 3}  # 운영 모델 시드 앙상블 수. 실험: 3시드 +0.4%p
WF_SEEDS = {"y_oc10": 3}  # 검증(워크포워드) 시드 수: 매매 모델은 운영과 동일, 표시용 y_up은 1(시간 절약)


class Panel:
    """특징 행렬 + 키."""

    def __init__(self, df: pd.DataFrame):
        self.cols = feature_cols(df)
        # 학습 시작일 이전 행은 룩백 계산용이라 행렬에서 뺀다(메모리)
        df = df[df["date"] >= TRAIN_START]
        self.X = df[self.cols].to_numpy("float32")
        self.date = df["date"].to_numpy()
        self.code = df["code"].to_numpy()
        self.close = df["close"].to_numpy("float32")
        self.y = {t: df[t].to_numpy("int8") for t in ("y_up", "y_touch_up")}
        self.y_gap = df["y_gap"].to_numpy("float32")
        self.y_oc = df["y_oc"].to_numpy("float32")
        # 매도 규칙 수익(상한가 도달 즉시 매도, 아니면 종가). 없으면 종가 기준
        self.y_ocx = df["y_ocx"].to_numpy("float32") if "y_ocx" in df.columns else self.y_oc
        # 단타 라벨: 익일 시가 매수 → 매도 규칙 수익 +10% 이상
        self.y["y_oc10"] = np.where(np.isnan(self.y_ocx) | (self.y["y_up"] < 0), -1, (self.y_ocx >= 0.10)).astype("int8")
        # 시가 상한가(매수 불가) 라벨: 익일 시가가 이미 상한가
        self.y["y_lock"] = np.where(np.isnan(self.y_gap) | (self.y["y_up"] < 0), -1, (self.y_gap >= 0.295)).astype("int8")
        self.next_date = df["next_date"].to_numpy()
        self.ym = np.array([d[:6] for d in self.date])
        self.age_days = ((pd.to_datetime(self.date.max(), format="%Y%m%d")
                          - pd.to_datetime(self.date, format="%Y%m%d")).days.to_numpy()).astype("float32")


HARD_NEG = 3.0  # 장중 상한가 도달 후 실패한 음성 가중(실험: 시가 매수 수익 +1.2%p)


def _weights(p: Panel, idx: np.ndarray, y: np.ndarray, sampled: bool, half_life: float | None = None) -> np.ndarray:
    w = np.ones(len(idx), dtype="float32")
    if sampled:
        w = np.where(y == 1, 1.0, 1.0 / NEG_KEEP).astype("float32")
    hard = (y == 0) & (p.y["y_touch_up"][idx] == 1)
    w = np.where(hard, w * HARD_NEG, w).astype("float32")
    if half_life:  # 지수 감쇠(반감기 개월). 기준일이 달라도 비율은 같아 워크포워드·운영 모두 동일 효과
        return w * (0.5 ** (p.age_days[idx] / (half_life * 30.4))).astype("float32")
    # 최근 가중(5년 전 0.5 → 지금 1.0)
    return w * (1.0 - 0.5 * np.clip(p.age_days[idx] / 1825.0, 0, 1)).astype("float32")


def _sample(p: Panel, target: str, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    y = p.y[target]
    idx = np.flatnonzero(mask & (y >= 0))
    rng = np.random.default_rng(seed)
    keep = (y[idx] == 1) | (rng.random(len(idx)) < NEG_KEEP)
    return idx[keep]


def train_one(p: Panel, target: str, mask: np.ndarray, rounds=ROUNDS, params=PARAMS, seed: int = 0):
    idx = _sample(p, target, mask, seed)
    y = p.y[target][idx]
    ds = lgb.Dataset(p.X[idx], y, weight=_weights(p, idx, y, True, HALF_LIFE.get(target)), feature_name=p.cols, free_raw_data=True)
    prm = dict(params, seed=params["seed"] + seed)
    return lgb.train(prm, ds, num_boost_round=rounds)


def precision_at_k(d: pd.DataFrame, score: str, target: str, k: int) -> tuple[float, float]:
    hits, n = [], []
    for _, g in d.groupby("date"):
        top = g.nlargest(k, score)
        hits.append(top[target].sum())
        n.append(len(top))
    hits, n = np.array(hits), np.array(n)
    return float(hits.sum() / max(1, n.sum())), float(hits.mean())


WF_CACHE_VERSION = 1  # 검증 결과 재사용 캐시 버전(과거 자료가 소급 변경되면 올려서 전체 재계산)


def _wf_key(p: Panel, target: str, rounds: int, n_seeds: int) -> str:
    """캐시 유효성 키: 모델·특징 코드, 변수 목록, 대상, 설정이 모두 같을 때만 지난달 결과를 재사용."""
    import hashlib
    h = hashlib.sha1()
    for f in ("model.py", "features.py", "text_features.py"):
        fp = Path(__file__).with_name(f)
        if fp.exists():
            h.update(fp.read_bytes())
    h.update("|".join(p.cols).encode())
    h.update(f"{target}|{rounds}|{n_seeds}|{TRAIN_START}|{WF_CACHE_VERSION}".encode())
    return h.hexdigest()


def walk_forward(p: Panel, target: str, months: int = 12, rounds: int = ROUNDS, n_seeds: int | None = None) -> pd.DataFrame:
    """월 단위 워크포워드: 각 달은 그 이전 데이터로만 학습한 모델(운영과 같은 시드 수 앙상블)로 예측.
    완료된 달의 결과는 stock/data/oos_<target>.parquet에 저장해 코드·변수가 같으면 다음 실행에서 재사용(이번 달만 계산)."""
    n_seeds = n_seeds or WF_SEEDS.get(target, 1)
    y = p.y[target]
    valid = y >= 0
    test_months = sorted(set(p.ym[valid]))[-months:]
    key = _wf_key(p, target, rounds, n_seeds)
    cache_f, key_f = DATA / f"oos_{target}.parquet", DATA / f"oos_{target}.key"
    cached = None
    if cache_f.exists() and key_f.exists() and key_f.read_text().strip() == key:
        cached = pd.read_parquet(cache_f)
        cached = cached.assign(ym=cached["date"].astype(str).str[:6])
    outs, reused = [], []
    for m in test_months:
        tr = valid & (p.ym < m) & (p.date >= TRAIN_START)
        te = valid & (p.ym == m)
        if te.sum() == 0 or y[tr].sum() < 50:
            continue
        if cached is not None:
            c = cached[cached["ym"] == m]
            if len(c) == int(te.sum()):  # 그달 행이 전부 있으면(완료된 달) 재사용
                outs.append(c.drop(columns=["ym"]))
                reused.append(m)
                continue
        # 엠바고: 학습 마지막 거래일의 라벨은 테스트 달과 겹치므로 제외
        tr &= p.date < p.date[tr].max()
        t0 = time.time()
        mdl = Ensemble([train_one(p, target, tr, rounds, seed=s) for s in range(n_seeds)])
        ti = np.flatnonzero(te)
        pr = mdl.predict(p.X[ti]).astype("float32")
        o = pd.DataFrame({"code": p.code[ti], "date": p.date[ti], "next_date": p.next_date[ti],
                          target: y[ti], "close": p.close[ti], "p": pr,
                          "y_gap": p.y_gap[ti], "y_oc": p.y_oc[ti], "y_ocx": p.y_ocx[ti], "ret1": p.X[ti, p.cols.index("ret1")],
                          "y_up": p.y["y_up"][ti], "pref": p.X[ti, p.cols.index("is_pref")],
                          "val": np.expm1(p.X[ti, p.cols.index("logval")])})
        outs.append(o)
        print(f"[wf {target}] {m} seeds={n_seeds} train={int(tr.sum())} pos={int(y[tr].sum())} test={len(ti)} "
              f"pos={int(y[ti].sum())} ap={average_precision_score(y[ti], pr):.4f} {time.time()-t0:.0f}s", flush=True)
        del mdl
        gc.collect()
    if reused:
        print(f"[wf {target}] 재사용 {len(reused)}개월({reused[0]}~{reused[-1]}), 새로 계산 {len(test_months) - len(reused)}개월", flush=True)
    oos = pd.concat(outs, ignore_index=True)
    oos.to_parquet(cache_f, index=False)
    key_f.write_text(key)
    return oos


COST = 0.0053  # 왕복 거래비용 가정: 거래세 0.20% + 수수료 0.03% + 체결 미끄러짐 0.30%(매수·매도 합계)
MIN_VAL = 1e9  # 후보 최소 거래대금(전일)


def tradable(df: pd.DataFrame) -> pd.DataFrame:
    """화면 후보와 동일한 사전 필터: 보통주, 전일 거래대금 10억 이상."""
    if "pref" in df.columns and "val" in df.columns:
        return df[(df["pref"] == 0) & (df["val"] >= MIN_VAL)]
    return df


def net_policy(oos: pd.DataFrame, k: int = 5, q: float = 0.3, warm: int = 20, trade_days: set | None = None) -> dict:
    """상위 k 시가 매수·종가 매도 순수익(비용 차감)과 미거래일 규칙.
    규칙: 그날 상위 k의 평균 점수가 직전 거래일들의 하위 q 분위 미만이면 거래하지 않음(확장 창, 초기 warm일은 거래)."""
    t = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    buy = t[t["y_gap"] < 0.295]
    day = pd.DataFrame({"score": t.groupby("date")["p"].mean()})
    day["net"] = (buy.groupby("date")["y_ocx"].mean() - COST).reindex(day.index).fillna(0.0)
    day["n_buy"] = buy.groupby("date").size().reindex(day.index).fillna(0).astype(int)
    day = day.sort_index()
    thr = day["score"].expanding().quantile(q).shift(1)
    day["trade"] = (day.index.isin(day.index[:warm])) | (day["score"] >= thr.fillna(-np.inf))
    if trade_days is not None:  # 전체 검증 기간에서 정한 거래일을 부분 기간에 그대로 적용
        day["trade"] = day.index.isin(trade_days)
    tr = day[day["trade"]]
    thr_now = float(day["score"].quantile(q)) if len(day) >= warm else None
    return {"cost": COST, "k": k, "days": int(len(day)), "q": q, "thr": thr_now,
            "all_day": float(day["net"].mean()), "all_cum": float(day["net"].sum()),
            "all_loss_days": float((day["net"] < 0).mean()), "all_worst": float(day["net"].min()),
            "trade_frac": float(day["trade"].mean()), "trade_day": float(tr["net"].mean()) if len(tr) else None,
            "trade_cum": float(tr["net"].sum()), "trade_loss_days": float((tr["net"] < 0).mean()) if len(tr) else None,
            "trade_worst": float(tr["net"].min()) if len(tr) else None,
            "skip_day": float(day.loc[~day["trade"], "net"].mean()) if (~day["trade"]).any() else None,
            "daily": [{"d": str(i), "net": round(float(r.net), 4), "trade": bool(r.trade)} for i, r in day.iterrows()]}


START_CAPITAL = 1_000_000  # 사용자 시작 자금


ADV_CAP = 0.01  # 종목당 매수 금액 한도: 전일 거래대금의 1%(문헌: 시장 충격 억제)


def allocate(scores: np.ndarray, prices: np.ndarray, capital: float, buyable: np.ndarray, cap: np.ndarray | None = None) -> np.ndarray:
    """화면의 매수 계획과 같은 규칙: 점수 가중(10~40%로 제한) 비중 → 종목당 한도(cap, 원) 적용 → 주수 내림 →
    남은 현금을 목표 금액 부족분이 큰 순서로 1주씩 채움(한도 내)."""
    n = np.zeros(len(scores), dtype=np.int64)
    ok = buyable & (prices > 0)
    if not ok.any():
        return n
    cap = np.full(len(scores), np.inf) if cap is None else np.where(np.isfinite(cap), cap, np.inf)
    w = np.where(ok, np.clip(scores, 0, None), 0.0)
    w = w / w.sum() if w.sum() > 0 else np.where(ok, 1.0, 0.0) / ok.sum()
    w = np.where(ok, np.clip(w, 0.1, 0.4), 0.0)
    w = w / w.sum()
    target = np.minimum(w * capital, cap)
    n = np.where(ok, np.floor(target / np.where(prices > 0, prices, np.inf)), 0).astype(np.int64)
    cash = capital - float((n * prices).sum())
    for _ in range(200):
        room = ok & (prices <= cash) & ((n + 1) * prices <= cap)
        short = np.where(room, target - n * prices, -np.inf)
        j = int(np.argmax(short))
        if not np.isfinite(short[j]):
            break
        n[j] += 1
        cash -= prices[j]
    return n


def sim_capital(oos: pd.DataFrame, k: int = 5, start: float = START_CAPITAL, trade_days: set | None = None) -> dict:
    """사용자 상황 그대로의 복리 시뮬레이션: 매일 아침 잔고 전액을 상위 k 중 매수 가능 종목(시가 상한가 제외)에 주수 단위로 배분,
    시가 매수·종가 매도, 왕복 비용 COST 차감. trade_days가 있으면 그 날만 거래(미거래일 규칙)."""
    t = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    cap = float(start)
    fixed_pnl = 0.0  # 매일 시작 자금(start)만 투자했을 때의 누적 손익(복리 없음)
    peak, mdd, rows, capped = cap, 0.0, [], 0
    for d, g in t.groupby("date", sort=True):
        if trade_days is not None and d not in trade_days:
            rows.append({"d": str(d), "bal": round(cap), "ret": 0.0, "n": 0, "trade": False, "fixed": round(start + fixed_pnl)})
            continue
        gap = g["y_gap"].to_numpy("float64")
        open_ = g["close"].to_numpy("float64") * (1 + np.nan_to_num(gap, nan=1.0))
        buyable = np.nan_to_num(gap, nan=1.0) < 0.295
        lim = ADV_CAP * g["val"].to_numpy("float64") if "val" in g.columns else None  # 종목당 전일 거래대금 1%
        r = np.nan_to_num(g["y_ocx"].to_numpy("float64"), nan=0.0)
        n = allocate(g["p"].to_numpy("float64"), open_, cap, buyable, lim)
        spent = float((n * open_).sum())
        pnl = float((n * open_ * r).sum()) - spent * COST
        ret = pnl / cap if cap > 0 else 0.0
        if buyable.any() and spent < 0.8 * cap:
            capped += 1
        nf = allocate(g["p"].to_numpy("float64"), open_, start, buyable, lim)
        fixed_pnl += float((nf * open_ * r).sum()) - float((nf * open_).sum()) * COST
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, cap / peak - 1)
        rows.append({"d": str(d), "bal": round(cap), "ret": round(ret, 4), "n": int((n > 0).sum()), "trade": True, "fixed": round(start + fixed_pnl)})
    rets = np.array([r["ret"] for r in rows if r["trade"]])
    return {"start": start, "final": round(cap), "cum": cap / start - 1, "mdd": float(mdd), "days": int(len(rows)),
            "trade_days": int(len(rets)), "win": float((rets > 0).mean()) if len(rets) else None,
            "day_avg": float(rets.mean()) if len(rets) else None, "worst": float(rets.min()) if len(rets) else None,
            "best": float(rets.max()) if len(rets) else None, "capped_days": int(capped), "adv_cap": ADV_CAP,
            "fixed_pnl": round(fixed_pnl), "fixed_cum": fixed_pnl / start, "daily": rows}


def summarize(oos: pd.DataFrame, target: str, trade_days: set | None = None) -> dict:
    """검증 요약. trade_days: 미거래일 규칙의 거래일 집합(전체 기간 기준, 부분 기간 요약 시 전달)."""
    oos = tradable(oos).reset_index(drop=True)
    y, p = oos[target].to_numpy(), oos["p"].to_numpy()
    res = {"n": int(len(y)), "pos": int(y.sum()), "base_rate": float(y.mean()),
           "ap": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p)),
           "days": int(oos["date"].nunique())}
    for k in (3, 5, 10, 20):
        pr, cnt = precision_at_k(oos, "p", target, k)
        res[f"p@{k}"] = pr
        res[f"hits@{k}"] = cnt
    # 시가 매수 기준 결과(상위 k): 시가 갭, 시가→종가, 시가가 이미 제한가인 비율
    sign = 1.0
    for k in (5, 10):
        tops = oos.groupby("date", group_keys=False).apply(lambda g: g.nlargest(k, "p"))
        gap, oc, occ = tops["y_gap"].astype(float), tops["y_ocx"].astype(float), tops["y_oc"].astype(float)
        res[f"open_gap@{k}"] = float(gap.mean())
        res[f"open_to_close@{k}"] = float(oc.mean())
        res[f"oc_win@{k}"] = float(((sign * oc) > 0).mean())
        res[f"oc_med@{k}"] = float(oc.median())
        res[f"unbuyable@{k}"] = float(((sign * gap) >= 0.295).mean())
        res[f"oc_mean_ex_unbuyable@{k}"] = float(oc[(sign * gap) < 0.295].mean()) if ((sign * gap) < 0.295).any() else None
        # 상한가 외 상승 지표: 익일 종가 +10% 이상 비율, 시가 매수 후 종가 +10% 이상 비율(시가 상한가 제외)
        cc = (1 + gap) * (1 + occ) - 1
        res[f"up10@{k}"] = float((cc >= 0.10).mean())
        res[f"up5@{k}"] = float((cc >= 0.05).mean())
        res[f"oc10@{k}"] = float((oc[gap < 0.295] >= 0.10).mean()) if (gap < 0.295).any() else None
        res[f"oc5@{k}"] = float((oc[gap < 0.295] >= 0.05).mean()) if (gap < 0.295).any() else None
        if "y_up" in tops.columns:
            res[f"up@{k}"] = float(tops["y_up"].astype(float).mean())
            res[f"net@{k}"] = float((oc[gap < 0.295] - COST).mean()) if (gap < 0.295).any() else None
    res["net"] = net_policy(oos, 5, trade_days=trade_days)
    # 100만 원 시작 복리 시뮬레이션(항상 거래 / 미거래일 규칙 적용)
    res["sim"] = sim_capital(oos, 5)
    res["sim"]["start_date"] = str(oos["date"].min())
    res["sim_policy"] = sim_capital(oos, 5, trade_days={r["d"] for r in res["net"]["daily"] if r["trade"]})
    res["sim_policy"].pop("daily", None)
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    cut = pd.cut(p, bins, right=False)
    gap_all, oc_all = oos["y_gap"].astype(float).to_numpy(), oos["y_ocx"].astype(float).to_numpy()
    tab = pd.DataFrame({"y": y, "b": cut, "gap": gap_all, "oc": oc_all})
    res["buckets"] = []
    for i, g in tab.groupby("b", observed=True):
        buyable = g[g["gap"] < 0.295]
        res["buckets"].append({"lo": float(i.left), "hi": float(i.right), "rate": float(g["y"].mean()), "n": int(len(g)),
                               "unb": float((g["gap"] >= 0.295).mean()),
                               "oc": float(buyable["oc"].mean()) if len(buyable) else None,
                               "oc_win": float((buyable["oc"] > 0).mean()) if len(buyable) else None})
    oos = oos.assign(ym=oos["date"].str[:6])
    res["monthly"] = []
    for m, g in oos.groupby("ym"):
        t10 = g.groupby("date", group_keys=False).apply(lambda x: x.nlargest(10, "p"))
        t5 = g.groupby("date", group_keys=False).apply(lambda x: x.nlargest(5, "p"))
        res["monthly"].append({"ym": m, "p@10": precision_at_k(g, "p", target, 10)[0], "p@5": precision_at_k(g, "p", target, 5)[0],
                               "pos": int(g[target].sum()),
                               "days": int(g["date"].nunique()), "oc@10": float(t10["y_ocx"].astype(float).mean()),
                               "gap@10": float(t10["y_gap"].astype(float).mean()),
                               "oc@5": float(t5.loc[t5["y_gap"] < 0.295, "y_ocx"].astype(float).mean()),
                               "gap@5": float(t5["y_gap"].astype(float).mean()),
                               "up10@5": float((((1 + t5["y_gap"].astype(float)) * (1 + t5["y_oc"].astype(float)) - 1) >= 0.10).mean()),
                               "up5@5": float((((1 + t5["y_gap"].astype(float)) * (1 + t5["y_oc"].astype(float)) - 1) >= 0.05).mean()),
                               "up@5": float(t5["y_up"].astype(float).mean()) if "y_up" in t5.columns else None,
                               "net@5": float((t5.loc[t5["y_gap"] < 0.295, "y_ocx"].astype(float) - COST).mean())})
    return res


HORIZONS = (1, 6, 12, 36)  # 성능 표기 기간(개월)


def summarize_horizons(oos: pd.DataFrame, target: str, horizons=HORIZONS) -> dict:
    """같은 검증 결과를 최근 1·6·12·36개월(개월당 21거래일)로 잘라 요약. 자료가 짧으면 가능한 기간만."""
    dates = sorted(oos["date"].astype(str).unique())
    full = net_policy(tradable(oos), 5)
    days = {r["d"] for r in full["daily"] if r["trade"]}
    out = {}
    for h in horizons:
        n = h * 21
        if len(dates) >= n * 0.95:
            sub = oos[oos["date"] >= dates[-min(n, len(dates))]]
            out[str(h)] = summarize(sub, target, trade_days=days)
            out[str(h)]["months"] = h
    if not out:
        out[str(max(1, round(len(dates) / 21)))] = summarize(oos, target, trade_days=days)
    return out


def fit_calibrator(oos: pd.DataFrame, target: str) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oos["p"].to_numpy(), oos[target].to_numpy())
    return iso


class Ensemble:
    """시드가 다른 모델 평균."""

    def __init__(self, models):
        self.models = models

    def predict(self, X, **kw):
        return np.mean([m.predict(X, **kw) for m in self.models], axis=0)

    def feature_importance(self, kind="gain"):
        return np.mean([m.feature_importance(kind) for m in self.models], axis=0)


def run(df, months: int = 12, rounds: int = ROUNDS) -> dict:
    """df: 특징 DataFrame 또는 미리 만든 Panel(호출자가 DataFrame을 지워 메모리를 아낄 수 있음)."""
    p = df if isinstance(df, Panel) else Panel(df)
    print(f"[model] {len(p.cols)} features, X {p.X.shape} ({p.X.nbytes/1e9:.1f}GB)")
    result = {"features": p.cols, "targets": {}}
    for target in ("y_up", "y_oc10", "y_lock"):
        # 매수 불가 모델은 제외 판단용: 검증은 최근 4개월만(시간 절약), 확률 보정만 필요
        oos = walk_forward(p, target, months if target != "y_lock" else min(months, 4), rounds)
        hz = summarize_horizons(oos, target)
        stats = dict(hz[str(max(int(k) for k in hz))])
        stats["horizons"] = hz
        print(f"[model] {target} OOS: {json.dumps({k: v for k, v in stats.items() if k not in ('buckets', 'monthly', 'horizons', 'net', 'sim', 'sim_policy')}, ensure_ascii=False)}", flush=True)
        iso = fit_calibrator(oos, target)
        full = (p.y[target] >= 0) & (p.date >= TRAIN_START)
        models = [train_one(p, target, full, rounds, seed=s) for s in range(N_SEEDS.get(target, 1))]
        mdl = Ensemble(models)
        imp = pd.Series(mdl.feature_importance("gain"), index=p.cols).sort_values(ascending=False)
        result["targets"][target] = {"model": mdl, "iso": iso, "oos": oos, "stats": stats,
                                     "importance": imp.head(40).to_dict()}
        models[0].save_model(str(DATA / f"model_{target}.txt"))
        gc.collect()
    return result
