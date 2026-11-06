"""학습·검증·예측. LightGBM 상한가 이진 모델 + 워크포워드 OOS 검증 + 등온 보정.

메모리: 특징 행렬을 float32 numpy로 한 번만 만들고 패널은 라벨·키만 남긴다.
"""
from __future__ import annotations

import gc
import json
import time
import warnings

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
        # 단타 라벨: 익일 시가 매수 → 종가 +10% 이상(실험 결과 시가 매수 수익에 가장 유리)
        self.y["y_oc10"] = np.where(np.isnan(self.y_oc) | (self.y["y_up"] < 0), -1, (self.y_oc >= 0.10)).astype("int8")
        # 시가 상한가(매수 불가) 라벨: 익일 시가가 이미 상한가
        self.y["y_lock"] = np.where(np.isnan(self.y_gap) | (self.y["y_up"] < 0), -1, (self.y_gap >= 0.295)).astype("int8")
        self.next_date = df["next_date"].to_numpy()
        self.ym = np.array([d[:6] for d in self.date])
        self.age_days = ((pd.to_datetime(self.date.max(), format="%Y%m%d")
                          - pd.to_datetime(self.date, format="%Y%m%d")).days.to_numpy()).astype("float32")


HARD_NEG = 3.0  # 장중 상한가 도달 후 실패한 음성 가중(실험: 시가 매수 수익 +1.2%p)


def _weights(p: Panel, idx: np.ndarray, y: np.ndarray, sampled: bool) -> np.ndarray:
    w = np.ones(len(idx), dtype="float32")
    if sampled:
        w = np.where(y == 1, 1.0, 1.0 / NEG_KEEP).astype("float32")
    hard = (y == 0) & (p.y["y_touch_up"][idx] == 1)
    w = np.where(hard, w * HARD_NEG, w).astype("float32")
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
    ds = lgb.Dataset(p.X[idx], y, weight=_weights(p, idx, y, True), feature_name=p.cols, free_raw_data=True)
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


def walk_forward(p: Panel, target: str, months: int = 12, rounds: int = ROUNDS) -> pd.DataFrame:
    """월 단위 워크포워드: 각 달은 그 이전 데이터로만 학습한 모델로 예측."""
    y = p.y[target]
    valid = y >= 0
    test_months = sorted(set(p.ym[valid]))[-months:]
    outs = []
    for m in test_months:
        tr = valid & (p.ym < m) & (p.date >= TRAIN_START)
        te = valid & (p.ym == m)
        if te.sum() == 0 or y[tr].sum() < 50:
            continue
        # 엠바고: 학습 마지막 거래일의 라벨은 테스트 달과 겹치므로 제외
        tr &= p.date < p.date[tr].max()
        t0 = time.time()
        mdl = train_one(p, target, tr, rounds)
        ti = np.flatnonzero(te)
        pr = mdl.predict(p.X[ti]).astype("float32")
        o = pd.DataFrame({"code": p.code[ti], "date": p.date[ti], "next_date": p.next_date[ti],
                          target: y[ti], "close": p.close[ti], "p": pr,
                          "y_gap": p.y_gap[ti], "y_oc": p.y_oc[ti], "ret1": p.X[ti, p.cols.index("ret1")],
                          "y_up": p.y["y_up"][ti], "pref": p.X[ti, p.cols.index("is_pref")],
                          "val": np.expm1(p.X[ti, p.cols.index("logval")])})
        outs.append(o)
        print(f"[wf {target}] {m} train={int(tr.sum())} pos={int(y[tr].sum())} test={len(ti)} "
              f"pos={int(y[ti].sum())} ap={average_precision_score(y[ti], pr):.4f} {time.time()-t0:.0f}s", flush=True)
        del mdl
        gc.collect()
    return pd.concat(outs, ignore_index=True)


COST = 0.0053  # 왕복 거래비용 가정: 거래세 0.20% + 수수료 0.03% + 체결 미끄러짐 0.30%(매수·매도 합계)
MIN_VAL = 1e9  # 후보 최소 거래대금(전일)


def tradable(df: pd.DataFrame) -> pd.DataFrame:
    """화면 후보와 동일한 사전 필터: 보통주, 전일 거래대금 10억 이상."""
    if "pref" in df.columns and "val" in df.columns:
        return df[(df["pref"] == 0) & (df["val"] >= MIN_VAL)]
    return df


def net_policy(oos: pd.DataFrame, k: int = 5, q: float = 0.3, warm: int = 20) -> dict:
    """상위 k 시가 매수·종가 매도 순수익(비용 차감)과 미거래일 규칙.
    규칙: 그날 상위 k의 평균 점수가 직전 거래일들의 하위 q 분위 미만이면 거래하지 않음(확장 창, 초기 warm일은 거래)."""
    t = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    buy = t[t["y_gap"] < 0.295]
    day = pd.DataFrame({"score": t.groupby("date")["p"].mean()})
    day["net"] = (buy.groupby("date")["y_oc"].mean() - COST).reindex(day.index).fillna(0.0)
    day["n_buy"] = buy.groupby("date").size().reindex(day.index).fillna(0).astype(int)
    day = day.sort_index()
    thr = day["score"].expanding().quantile(q).shift(1)
    day["trade"] = (day.index.isin(day.index[:warm])) | (day["score"] >= thr.fillna(-np.inf))
    tr = day[day["trade"]]
    return {"cost": COST, "k": k, "days": int(len(day)),
            "all_day": float(day["net"].mean()), "all_cum": float(day["net"].sum()),
            "all_loss_days": float((day["net"] < 0).mean()), "all_worst": float(day["net"].min()),
            "trade_frac": float(day["trade"].mean()), "trade_day": float(tr["net"].mean()) if len(tr) else None,
            "trade_cum": float(tr["net"].sum()), "trade_loss_days": float((tr["net"] < 0).mean()) if len(tr) else None,
            "trade_worst": float(tr["net"].min()) if len(tr) else None,
            "skip_day": float(day.loc[~day["trade"], "net"].mean()) if (~day["trade"]).any() else None,
            "daily": [{"d": str(i), "net": round(float(r.net), 4), "trade": bool(r.trade)} for i, r in day.iterrows()]}


def summarize(oos: pd.DataFrame, target: str) -> dict:
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
        gap, oc = tops["y_gap"].astype(float), tops["y_oc"].astype(float)
        res[f"open_gap@{k}"] = float(gap.mean())
        res[f"open_to_close@{k}"] = float(oc.mean())
        res[f"oc_win@{k}"] = float(((sign * oc) > 0).mean())
        res[f"oc_med@{k}"] = float(oc.median())
        res[f"unbuyable@{k}"] = float(((sign * gap) >= 0.295).mean())
        res[f"oc_mean_ex_unbuyable@{k}"] = float(oc[(sign * gap) < 0.295].mean()) if ((sign * gap) < 0.295).any() else None
        # 상한가 외 상승 지표: 익일 종가 +10% 이상 비율, 시가 매수 후 종가 +10% 이상 비율(시가 상한가 제외)
        cc = (1 + gap) * (1 + oc) - 1
        res[f"up10@{k}"] = float((cc >= 0.10).mean())
        res[f"up5@{k}"] = float((cc >= 0.05).mean())
        res[f"oc10@{k}"] = float((oc[gap < 0.295] >= 0.10).mean()) if (gap < 0.295).any() else None
        res[f"oc5@{k}"] = float((oc[gap < 0.295] >= 0.05).mean()) if (gap < 0.295).any() else None
        if "y_up" in tops.columns:
            res[f"up@{k}"] = float(tops["y_up"].astype(float).mean())
            res[f"net@{k}"] = float((oc[gap < 0.295] - COST).mean()) if (gap < 0.295).any() else None
    res["net"] = net_policy(oos, 5)
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    cut = pd.cut(p, bins, right=False)
    gap_all, oc_all = oos["y_gap"].astype(float).to_numpy(), oos["y_oc"].astype(float).to_numpy()
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
                               "days": int(g["date"].nunique()), "oc@10": float(t10["y_oc"].astype(float).mean()),
                               "gap@10": float(t10["y_gap"].astype(float).mean()),
                               "oc@5": float(t5.loc[t5["y_gap"] < 0.295, "y_oc"].astype(float).mean()),
                               "gap@5": float(t5["y_gap"].astype(float).mean()),
                               "up10@5": float((((1 + t5["y_gap"].astype(float)) * (1 + t5["y_oc"].astype(float)) - 1) >= 0.10).mean()),
                               "up5@5": float((((1 + t5["y_gap"].astype(float)) * (1 + t5["y_oc"].astype(float)) - 1) >= 0.05).mean()),
                               "up@5": float(t5["y_up"].astype(float).mean()) if "y_up" in t5.columns else None,
                               "net@5": float((t5.loc[t5["y_gap"] < 0.295, "y_oc"].astype(float) - COST).mean())})
    return res


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


def run(df, months: int = 12, rounds: int = ROUNDS, n_seeds: int = 3) -> dict:
    """df: 특징 DataFrame 또는 미리 만든 Panel(호출자가 DataFrame을 지워 메모리를 아낄 수 있음)."""
    p = df if isinstance(df, Panel) else Panel(df)
    print(f"[model] {len(p.cols)} features, X {p.X.shape} ({p.X.nbytes/1e9:.1f}GB)")
    result = {"features": p.cols, "targets": {}}
    for target in ("y_up", "y_oc10", "y_lock"):
        # 매수 불가 모델은 제외 판단용: 검증은 최근 4개월만(시간 절약), 확률 보정만 필요
        oos = walk_forward(p, target, months if target != "y_lock" else min(months, 4), rounds)
        stats = summarize(oos, target)
        print(f"[model] {target} OOS: {json.dumps({k: v for k, v in stats.items() if k not in ('buckets', 'monthly')}, ensure_ascii=False)}", flush=True)
        iso = fit_calibrator(oos, target)
        full = (p.y[target] >= 0) & (p.date >= TRAIN_START)
        models = [train_one(p, target, full, rounds, seed=s) for s in range(n_seeds if target == "y_up" else 1)]
        mdl = Ensemble(models)
        imp = pd.Series(mdl.feature_importance("gain"), index=p.cols).sort_values(ascending=False)
        result["targets"][target] = {"model": mdl, "iso": iso, "oos": oos, "stats": stats,
                                     "importance": imp.head(40).to_dict()}
        oos.to_parquet(DATA / f"oos_{target}.parquet", index=False)
        models[0].save_model(str(DATA / f"model_{target}.txt"))
        gc.collect()
    return result
