"""모델 실험 하네스: 설정 이름을 받아 워크포워드 검증을 돌리고 요약을 한 줄 JSON으로 출력한다.

  python -m stock.experiment base rank oc10 ...   (여러 개면 순서대로)
평가는 단타 관점: 상위 5개의 상한가 마감 비율(p@5), 시가에 살 수 있던 종목의 시가→종가 평균(oc@5),
상승 마감 비율(win@5), 시가가 상한가라 못 산 비율(unb@5). 보통주·거래대금 10억↑만 추린 결과(_lc)도 함께.
"""
from __future__ import annotations

import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stock import features
from stock.model import PARAMS, ROUNDS, TRAIN_START, Panel, _weights

CONFIGS = {
    "base": {},
    "lr02_r1200": {"params": {"learning_rate": 0.02}, "rounds": 1200},
    "leaves127": {"params": {"num_leaves": 127, "min_data_in_leaf": 100}},
    "leaves31": {"params": {"num_leaves": 31}},
    "ff05": {"params": {"feature_fraction": 0.5}},
    "neg50": {"neg_keep": 0.5},
    "neg100": {"neg_keep": 1.0},
    "recent_only": {"train_start": "20230101"},
    "norecency": {"recency": False},
    "rank": {"objective": "lambdarank"},
    "rank_graded": {"objective": "lambdarank", "graded": True},
    "buyable": {"target": "up_buyable"},
    "oc10": {"target": "oc10"},
    "oc_reg": {"target": "oc_reg"},
    "no_text": {"drop_prefix": ("d_", "n_")},
    "no_flow": {"drop_prefix": ("frgn", "inst", "indiv", "frate", "flow_")},
    "no_group": {"drop_prefix": ("ind_", "thm_")},
    "no_rank": {"drop_prefix": ("rk_",)},
    "dart": {"params": {"boosting": "dart", "drop_rate": 0.1}, "rounds": 400},
    "goss": {"params": {"boosting": "goss", "bagging_fraction": 1.0, "bagging_freq": 0}},
    "l2_1": {"params": {"lambda_l2": 1.0}},
    "minleaf50": {"params": {"min_data_in_leaf": 50}},
    "pos_w3": {"pos_weight": 3.0},
}


def labels(p: Panel, target: str):
    yu = p.y["y_up"]
    valid = yu >= 0
    if target == "up":
        return yu.astype("int8"), valid
    if target == "up_buyable":
        y = ((yu == 1) & (p.y_gap < 0.295)).astype("int8")
        return y, valid & ~np.isnan(p.y_gap)
    if target == "oc10":
        y = (p.y_oc >= 0.10).astype("int8")
        return y, valid & ~np.isnan(p.y_oc)
    if target == "oc_reg":
        y = np.clip(np.nan_to_num(p.y_oc, nan=0.0), -0.3, 0.3).astype("float32")
        return y, valid & ~np.isnan(p.y_oc)
    raise ValueError(target)


def fit(p: Panel, cfg: dict, tr: np.ndarray, y_all, seed=0):
    neg_keep = cfg.get("neg_keep", 0.25)
    objective = cfg.get("objective", "binary")
    idx = np.flatnonzero(tr)
    rng = np.random.default_rng(seed)
    pos = (y_all[idx] > 0) if objective != "regression" else (np.abs(y_all[idx]) > 0.1)
    keep = pos | (rng.random(len(idx)) < neg_keep)
    idx = idx[keep]
    y = y_all[idx]
    params = dict(PARAMS, **cfg.get("params", {}))
    params["seed"] = PARAMS["seed"] + seed
    cols = p.cols
    X = p.X[idx]
    if cfg.get("drop_prefix"):
        keep_c = [i for i, c in enumerate(cols) if not c.startswith(tuple(cfg["drop_prefix"]))]
        X = X[:, keep_c]
        cols = [cols[i] for i in keep_c]
    w = np.where(y > 0, 1.0, 1.0 / neg_keep).astype("float32") if objective != "regression" else np.ones(len(idx), "float32")
    if cfg.get("recency", True):
        w = w * (1.0 - 0.5 * np.clip(p.age_days[idx] / 1825.0, 0, 1)).astype("float32")
    if cfg.get("pos_weight"):
        w = np.where(y > 0, w * cfg["pos_weight"], w).astype("float32")
    if objective == "lambdarank":
        order = np.argsort(p.date[idx], kind="stable")
        idx, X, y, w = idx[order], X[order], y[order], w[order]
        if cfg.get("graded"):
            y = (p.y["y_up"][idx] * 2 + (np.nan_to_num(p.y_oc[idx]) >= 0.10)).astype("int8")
        _, counts = np.unique(p.date[idx], return_counts=True)
        params.update(objective="lambdarank", metric="ndcg", lambdarank_truncation_level=30, label_gain=[0, 1, 3])
        ds = lgb.Dataset(X, y, group=counts, feature_name=cols, free_raw_data=True)
    else:
        if objective == "regression":
            params.update(objective="huber", alpha=0.1)
        ds = lgb.Dataset(X, y, weight=w, feature_name=cols, free_raw_data=True)
    mdl = lgb.train(params, ds, num_boost_round=cfg.get("rounds", ROUNDS))
    return mdl, cols


def evaluate(oos: pd.DataFrame, k: int = 5) -> dict:
    res = {"n": int(len(oos)), "days": int(oos["date"].nunique()), "ap": float(average_precision_score(oos["y_up"], oos["p"]))}
    for name, d in (("", oos), ("_lc", oos[(oos["pref"] == 0) & (oos["val"] >= 1e9)])):
        tops = d.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
        buy = tops[tops["y_gap"] < 0.295]
        res[f"p@{k}{name}"] = float(tops["y_up"].mean())
        res[f"oc@{k}{name}"] = float(buy["y_oc"].mean())
        res[f"win@{k}{name}"] = float((buy["y_oc"] > 0).mean())
        res[f"unb@{k}{name}"] = float((tops["y_gap"] >= 0.295).mean())
        res[f"med@{k}{name}"] = float(buy["y_oc"].median())
        daily = buy.groupby("date")["y_oc"].mean() - 0.003
        res[f"dayavg@{k}{name}"] = float(daily.mean())
        res[f"worst@{k}{name}"] = float(daily.min())
    t3 = oos.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(3)
    res["p@3"] = float(t3["y_up"].mean())
    return res


def run(name: str, p: Panel, months: int, seed: int = 0) -> dict:
    cfg = CONFIGS[name]
    target = cfg.get("target", "up")
    y_all, valid = labels(p, target)
    train_start = cfg.get("train_start", TRAIN_START)
    yms = sorted(set(p.ym[valid]))
    test_months = yms[-months:]
    outs = []
    t0 = time.time()
    i_pref, i_val = p.cols.index("is_pref"), p.cols.index("logval")
    for m in test_months:
        tr = valid & (p.ym < m) & (p.date >= train_start)
        te = valid & (p.ym == m)
        if te.sum() == 0:
            continue
        tr &= p.date < p.date[tr].max()
        mdl, cols = fit(p, cfg, tr, y_all, seed)
        ti = np.flatnonzero(te)
        Xte = p.X[ti]
        if cols != p.cols:
            Xte = Xte[:, [p.cols.index(c) for c in cols]]
        pr = mdl.predict(Xte).astype("float32")
        outs.append(pd.DataFrame({"date": p.date[ti], "y_up": p.y["y_up"][ti], "p": pr, "y_gap": p.y_gap[ti], "y_oc": p.y_oc[ti],
                                  "pref": p.X[ti, i_pref], "val": np.expm1(p.X[ti, i_val])}))
        del mdl, Xte
        gc.collect()
        print(f"[exp {name}] {m} done {time.time()-t0:.0f}s", flush=True)
    oos = pd.concat(outs, ignore_index=True)
    res = {"name": name, "months": months, "sec": int(time.time() - t0), **evaluate(oos)}
    print("[exp] " + json.dumps(res), flush=True)
    return res


def main(argv):
    months = 6
    names = []
    for a in argv:
        if a.startswith("--months="):
            months = int(a.split("=")[1])
        else:
            names.append(a)
    prices, flows, universe, groups, market, disc, news = features.load_all()
    panel = features.build(prices, flows, universe, groups, market, disclosures=disc, news=news)
    del prices, flows
    p = Panel(panel)
    del panel
    gc.collect()
    print(f"[exp] panel X {p.X.shape}", flush=True)
    for n in names:
        run(n, p, months)


if __name__ == "__main__":
    main(sys.argv[1:])
