"""모델 실험 하네스: 설정 이름을 받아 워크포워드 검증을 돌리고 요약을 한 줄 JSON으로 출력한다.

  python -m stock.experiment base rank oc10 ...   (여러 개면 순서대로)
평가는 단타 관점: 상위 5개의 상한가 마감 비율(p@5), 시가에 살 수 있던 종목의 시가→종가 평균(oc@5),
상승 마감 비율(win@5), 시가가 상한가라 못 산 비율(unb@5). 보통주·거래대금 10억↑만 추린 결과(_lc)도 함께.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from stock import features
from stock.model import COST, PARAMS, ROUNDS, TRAIN_START, Panel, _weights, sim_capital

NEW_FEATS = ("r5max", "r20q80", "ret1_sq", "n_top20_20", "n_bot20_20", "n_tb20_20", "vz_small", "indiv_act", "since_alert")

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
    "no_newfeat": {"drop_prefix": ("ret20_ex", "ret60_ex", "KOSDAQ_r60", "KOSPI_r60", "KOSDAQ_bull", "KOSPI_bull")},
    "dart": {"params": {"boosting": "dart", "drop_rate": 0.1}, "rounds": 400},
    "goss": {"params": {"boosting": "goss", "bagging_fraction": 1.0, "bagging_freq": 0}},
    "l2_1": {"params": {"lambda_l2": 1.0}},
    "minleaf50": {"params": {"min_data_in_leaf": 50}},
    "pos_w3": {"pos_weight": 3.0},
    "hardneg": {"hard_neg": 3.0},            # 다음날 상한가 터치 후 실패한 음성에 가중
    "twostage": {"two_stage": True},         # P(터치) × P(마감|터치)
    "blend": {"blend": True},                # 이진 + 랭킹 순위 평균
    "seeds3": {"n_seeds": 3},
    "touch": {"target": "touch"},            # 장중 터치 자체를 예측
    # 2차: 단타 수익(시가→종가) 지향 조합
    "oc5": {"target": "oc5"},
    "oc15": {"target": "oc15"},
    "oc10_hn": {"target": "oc10", "hard_neg": 3.0},
    "oc10_nr": {"target": "oc10", "recency": False},
    "oc10_hn_nr": {"target": "oc10", "hard_neg": 3.0, "recency": False},
    "oc10_full": {"target": "oc10", "hard_neg": 3.0, "recency": False, "params": {"feature_fraction": 0.5}},
    "hn_nr": {"hard_neg": 3.0, "recency": False},
    "hn_nr_ff05": {"hard_neg": 3.0, "recency": False, "params": {"feature_fraction": 0.5}},
    "mix": {"mix": True},                    # 상한가 모델 + oc10 모델 순위 평균
    "mix_hn": {"mix": True, "hard_neg": 3.0, "recency": False},
    # 3차: 문헌 조사 반영(변수 추가 기여, 순위 학습, 앙상블, 학습 창·반감기, 규칙 필터)
    "oc10_nonew": {"target": "oc10", "hard_neg": 3.0, "drop_prefix": NEW_FEATS},   # 새 변수 제외(기여도 비교용)
    "oc_rank": {"target": "oc_grade", "objective": "lambdarank", "pos_min": 2},    # 등급 라벨(0/1/2/3) 순위 학습
    "oc_rank_t10": {"target": "oc_grade", "objective": "lambdarank", "pos_min": 2, "trunc": 10},
    "oc10_xgb": {"target": "oc10", "hard_neg": 3.0, "lib": "xgb"},
    "oc10_seeds3": {"target": "oc10", "hard_neg": 3.0, "n_seeds": 3},
    "oc10_ens": {"target": "oc10", "hard_neg": 3.0, "ens": [{}, {"seed_off": 1}, {"lib": "xgb"}]},  # LGBM 2시드 + XGB 순위 평균
    "oc_rank_ens": {"target": "oc10", "hard_neg": 3.0, "ens": [{}, {"target": "oc_grade", "objective": "lambdarank", "pos_min": 2}]},
    "oc10_win12": {"target": "oc10", "hard_neg": 3.0, "window_months": 12},
    "oc10_win24": {"target": "oc10", "hard_neg": 3.0, "window_months": 24},
    "oc10_win36": {"target": "oc10", "hard_neg": 3.0, "window_months": 36},
    "oc10_hl6": {"target": "oc10", "hard_neg": 3.0, "half_life": 6},
    "oc10_hl12": {"target": "oc10", "hard_neg": 3.0, "half_life": 12},
    "oc10_hl24": {"target": "oc10", "hard_neg": 3.0, "half_life": 24},
    "oc10_rules": {"target": "oc10", "hard_neg": 3.0, "rules": True},   # 예측 후 규칙 필터(경고 지정 3일 초과·소형주 거래량 충격 등)
    # 4차: 3차에서 우세한 요소 결합(다중 시드 +0.4%p, 반감기 24개월 +0.3%p, 순위 학습 앙상블 +0.2%p)
    "oc10_s2_hl24": {"target": "oc10", "hard_neg": 3.0, "n_seeds": 2, "half_life": 24},
    "oc10_s3_hl24": {"target": "oc10", "hard_neg": 3.0, "n_seeds": 3, "half_life": 24},
    "oc10_s5": {"target": "oc10", "hard_neg": 3.0, "n_seeds": 5},
    "oc10_s3_hl24_rank": {"target": "oc10", "hard_neg": 3.0, "half_life": 24,
                          "ens": [{}, {"seed_off": 1}, {"seed_off": 2}, {"target": "oc_grade", "objective": "lambdarank", "pos_min": 2}]},
    "oc10_s3_hl36": {"target": "oc10", "hard_neg": 3.0, "n_seeds": 3, "half_life": 36},
    # 5차(2026-11-07): 다른 학습기·DoubleEnsemble 구성원
    "oc10_cat": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "lib": "cat"},                       # CatBoost 단독
    "oc10_cat_ens": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "ens": [{}, {"seed_off": 1}, {"lib": "cat"}]},  # LGBM 2시드 + CatBoost 순위 평균
    "oc10_de": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "de": 3},                             # DoubleEnsemble-lite(손실 재가중+변수 부분집합 3회)
    "oc10_de_s3": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "de": 3, "n_seeds": 3},
    # 하이퍼파라미터 격자(운영 구성 s3_hl24 기준)
    "hp_lr02_r1200": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"learning_rate": 0.02}, "rounds": 1200},
    "hp_lr05_r400": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"learning_rate": 0.05}, "rounds": 400},
    "hp_leaves127": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"num_leaves": 127, "min_data_in_leaf": 400}},
    "hp_leaves31": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"num_leaves": 31, "min_data_in_leaf": 100}},
    "hp_ff05_bag07": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"feature_fraction": 0.5, "bagging_fraction": 0.7}},
    "hp_l2_30_minleaf500": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "params": {"lambda_l2": 30.0, "min_data_in_leaf": 500}},
    "hp_neg50": {"target": "oc10", "hard_neg": 3.0, "half_life": 24, "n_seeds": 3, "neg_keep": 0.5},
    "hp_hn5": {"target": "oc10", "hard_neg": 5.0, "half_life": 24, "n_seeds": 3},
}


def labels(p: Panel, target: str):
    yu = p.y["y_up"]
    valid = yu >= 0
    if target == "up":
        return yu.astype("int8"), valid
    if target == "touch":
        return p.y["y_touch_up"].astype("int8"), valid & (p.y["y_touch_up"] >= 0)
    if target == "up_hold":  # 터치한 날만: 마감 여부
        yt = p.y["y_touch_up"]
        return yu.astype("int8"), valid & (yt == 1)
    if target == "up_buyable":
        y = ((yu == 1) & (p.y_gap < 0.295)).astype("int8")
        return y, valid & ~np.isnan(p.y_gap)
    if target.startswith("oc") and target[2:].isdigit():
        y = (p.y_ocx >= int(target[2:]) / 100).astype("int8")
        return y, valid & ~np.isnan(p.y_ocx)
    if target == "oc_grade":  # 시가 매수 수익 등급: 매수 불가·하락 0, 0~5% 1, 5~10% 2, 10%↑ 3
        oc = np.nan_to_num(p.y_ocx, nan=-1.0)
        g = np.where(oc >= 0.10, 3, np.where(oc >= 0.05, 2, np.where(oc >= 0, 1, 0)))
        g = np.where(np.nan_to_num(p.y_gap, nan=1.0) >= 0.295, 0, g)
        return g.astype("int8"), valid & ~np.isnan(p.y_ocx)
    if target == "oc_reg":
        y = np.clip(np.nan_to_num(p.y_oc, nan=0.0), -0.3, 0.3).astype("float32")
        return y, valid & ~np.isnan(p.y_oc)
    raise ValueError(target)


_DNUM = {}


def _days(p: Panel) -> np.ndarray:
    if "d" not in _DNUM:
        _DNUM["d"] = pd.to_datetime(pd.Series(p.date), format="%Y%m%d").values.astype("datetime64[D]").astype("int64")
    return _DNUM["d"]


def fit(p: Panel, cfg: dict, tr: np.ndarray, y_all, seed=0):
    neg_keep = cfg.get("neg_keep", 0.25)
    objective = cfg.get("objective", "binary")
    seed = seed + cfg.get("seed_off", 0)
    idx = np.flatnonzero(tr)
    rng = np.random.default_rng(seed)
    pos = (y_all[idx] >= cfg.get("pos_min", 1)) if objective != "regression" else (np.abs(y_all[idx]) > 0.1)
    keep = pos | (rng.random(len(idx)) < neg_keep)
    idx = idx[keep]
    y = y_all[idx]
    params = dict(PARAMS, **cfg.get("params", {}))
    params["seed"] = PARAMS["seed"] + seed
    ref_day = _days(p)[np.flatnonzero(tr)].max()  # 학습 자료의 마지막 날(검증 달 직전) 기준 경과일
    age = (ref_day - _days(p)[idx]).astype("float32")
    cols = p.cols
    X = p.X[idx]
    if cfg.get("drop_prefix"):
        keep_c = [i for i, c in enumerate(cols) if not c.startswith(tuple(cfg["drop_prefix"]))]
        X = X[:, keep_c]
        cols = [cols[i] for i in keep_c]
    w = np.where(y > 0, 1.0, 1.0 / neg_keep).astype("float32") if objective != "regression" else np.ones(len(idx), "float32")
    if cfg.get("half_life"):
        w = w * (0.5 ** (age / (cfg["half_life"] * 30.4))).astype("float32")
    elif cfg.get("recency", True):
        w = w * (1.0 - 0.5 * np.clip(age / 1825.0, 0, 1)).astype("float32")
    if cfg.get("pos_weight"):
        w = np.where(y > 0, w * cfg["pos_weight"], w).astype("float32")
    if cfg.get("hard_neg"):
        hard = (y <= 0) & (p.y["y_touch_up"][idx] == 1)
        w = np.where(hard, w * cfg["hard_neg"], w).astype("float32")
    if objective == "lambdarank":
        order = np.argsort(p.date[idx], kind="stable")
        idx, X, y, w = idx[order], X[order], y[order], w[order]
        if cfg.get("graded"):
            y = (p.y["y_up"][idx] * 2 + (np.nan_to_num(p.y_oc[idx]) >= 0.10)).astype("int8")
        _, counts = np.unique(p.date[idx], return_counts=True)
        params.update(objective="lambdarank", metric="ndcg", eval_at=[5], lambdarank_truncation_level=cfg.get("trunc", 20),
                      label_gain=[0, 1, 3, 7])
        ds = lgb.Dataset(X, y, group=counts, feature_name=cols, free_raw_data=True)
    elif cfg.get("lib") == "cat":
        from catboost import CatBoostClassifier, Pool
        pool = Pool(_finite(X), label=y, weight=w, feature_names=list(cols))
        cb = CatBoostClassifier(iterations=cfg.get("rounds", 1500), learning_rate=0.05, depth=8, l2_leaf_reg=10.0, border_count=128,
                                random_seed=params["seed"], thread_count=-1, verbose=0, allow_writing_files=False)
        cb.fit(pool)
        del pool
        return _Cat(cb), cols
    elif cfg.get("de"):
        return _double_ensemble(X, y, w, cols, params, cfg, seed), cols
    elif cfg.get("lib") == "xgb":
        import xgboost as xgb
        dm = xgb.DMatrix(_finite(X), label=y, weight=w, feature_names=cols, nthread=0)
        xp = {"objective": "binary:logistic", "tree_method": "hist", "eta": params["learning_rate"], "max_depth": 8,
              "min_child_weight": 200, "subsample": 0.8, "colsample_bytree": 0.7, "lambda": 10.0, "max_bin": 255,
              "seed": params["seed"], "verbosity": 0}
        bst = xgb.train(xp, dm, num_boost_round=cfg.get("rounds", ROUNDS))
        del dm
        return _XGB(bst, cols), cols
    else:
        if objective == "regression":
            params.update(objective="huber", alpha=0.1)
        ds = lgb.Dataset(X, y, weight=w, feature_name=cols, free_raw_data=True)
    mdl = lgb.train(params, ds, num_boost_round=cfg.get("rounds", ROUNDS))
    return mdl, cols


class _Cat:
    def __init__(self, cb):
        self.cb = cb

    def predict(self, X):
        return self.cb.predict_proba(_finite(X))[:, 1]


class _DE:
    """변수 부분집합별 모델 평균."""
    def __init__(self, parts):
        self.parts = parts  # [(model, col_idx)]

    def predict(self, X):
        return np.mean([m.predict(X[:, ci]) for m, ci in self.parts], axis=0)


def _double_ensemble(X, y, w, cols, params, cfg, seed):
    """DoubleEnsemble(Zhang 외 2020) 간소판: 회차마다 (1) 직전 회차 앙상블의 표본 손실 순위로 가중(상위 50% 손실 ×1.5),
    (2) 변수 60% 무작위 부분집합으로 LightGBM 학습, 예측은 평균."""
    rng = np.random.default_rng(100 + seed)
    k, frac = int(cfg["de"]), cfg.get("de_frac", 0.6)
    parts, pred = [], None
    for r in range(k):
        wr = w.copy()
        if pred is not None:
            loss = -(y * np.log(np.clip(pred, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-6, 1)))
            hard = loss >= np.quantile(loss, 0.5)
            wr = np.where(hard, wr * 1.5, wr).astype("float32")
        ci = np.sort(rng.choice(len(cols), int(len(cols) * frac), replace=False))
        prm = dict(params, seed=params["seed"] + r, feature_fraction=min(1.0, params.get("feature_fraction", 0.7) / frac))
        ds = lgb.Dataset(X[:, ci], y, weight=wr, feature_name=[cols[i] for i in ci], free_raw_data=True)
        m = lgb.train(prm, ds, num_boost_round=cfg.get("rounds", ROUNDS))
        parts.append((m, ci))
        pr = m.predict(X[:, ci])
        pred = pr if pred is None else (pred * len(parts[:-1]) + pr) / len(parts)
    return _DE(parts)


class _XGB:
    def __init__(self, bst, cols):
        self.bst, self.cols = bst, cols

    def predict(self, X):
        import xgboost as xgb
        return self.bst.predict(xgb.DMatrix(_finite(X), feature_names=self.cols, nthread=0))


def _finite(X: np.ndarray) -> np.ndarray:
    """XGBoost는 inf를 받지 않으므로 결측으로 바꾼다(LightGBM은 큰 값으로 처리)."""
    return np.where(np.isfinite(X), X, np.nan).astype("float32")


def apply_rules(p: Panel, ti: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """문헌 기반 규칙: 투자경고 지정 후 3일 초과 경과(20일 하락 구간), 소형주 거래량 충격(z>3) 종목은 순위에서 뒤로 보낸다."""
    pen = np.zeros(len(ti), "float32")
    for name, lo, hi, val in (("since_alert", 3, 20, 1.0), ("vz_small", 3.0, 1e9, 0.5)):
        if name in p.cols:
            x = p.X[ti, p.cols.index(name)]
            pen += np.where((x > lo) & (x <= hi), val, 0.0).astype("float32")
    return pr - pen


def evaluate(oos: pd.DataFrame, k: int = 5) -> dict:
    res = {"n": int(len(oos)), "days": int(oos["date"].nunique()), "ap": float(average_precision_score(oos["y_up"], oos["p"]))}
    for name, d in (("", oos), ("_lc", oos[(oos["pref"] == 0) & (oos["val"] >= 1e9)])):
        tops = d.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
        buy = tops[tops["y_gap"] < 0.295]
        res[f"p@{k}{name}"] = float(tops["y_up"].mean())
        res[f"oc@{k}{name}"] = float(buy["y_ocx"].mean())
        res[f"win@{k}{name}"] = float((buy["y_ocx"] > 0).mean())
        res[f"unb@{k}{name}"] = float((tops["y_gap"] >= 0.295).mean())
        res[f"med@{k}{name}"] = float(buy["y_ocx"].median())
        daily = buy.groupby("date")["y_ocx"].mean() - COST  # 운영과 같은 왕복 비용
        res[f"dayavg@{k}{name}"] = float(daily.mean())
        res[f"worst@{k}{name}"] = float(daily.min())
        if "close" in d.columns:
            sim = sim_capital(d, k)
            res[f"sim@{k}{name}"] = float(sim["cum"])
            res[f"simfix@{k}{name}"] = float(sim["fixed_cum"])
            res[f"simmdd@{k}{name}"] = float(sim["mdd"])
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
    all_yms = sorted(set(p.ym))
    # 월별 중간 저장(EXP_CKPT_DIR): 전원 차단 등으로 중단된 실험을 다시 실행하면 완료된 달은 건너뛴다.
    # 키에 자료 범위(마지막 날짜·행 수)를 넣어 자료가 바뀌면 처음부터 계산.
    ckpt = None
    if os.environ.get("EXP_CKPT_DIR"):
        ckpt = Path(os.environ["EXP_CKPT_DIR"]) / f"{name}_{months}_{int(p.date.max())}_{len(p.date)}_{seed}"
        ckpt.mkdir(parents=True, exist_ok=True)
    for m in test_months:
        if ckpt is not None and (ckpt / f"{m}.parquet").exists():
            outs.append(pd.read_parquet(ckpt / f"{m}.parquet"))
            print(f"[exp {name}] {m} 중간 저장 사용", flush=True)
            continue
        tr = valid & (p.ym < m) & (p.date >= train_start)
        if cfg.get("window_months"):
            tr &= p.ym >= all_yms[max(0, all_yms.index(m) - cfg["window_months"])]
        te = valid & (p.ym == m)
        if te.sum() == 0:
            continue
        tr &= p.date < p.date[tr].max()
        ti = np.flatnonzero(te)

        def _pred(c, yv, trm, s=seed):
            mdl, cols = fit(p, c, trm, yv, s)
            Xte = p.X[ti]
            if cols != p.cols:
                Xte = Xte[:, [p.cols.index(c2) for c2 in cols]]
            out = mdl.predict(Xte).astype("float32")
            del mdl
            return out

        if cfg.get("two_stage"):
            yt, vt = labels(p, "touch")
            yh, vh = labels(p, "up_hold")
            pr = _pred({}, yt, tr & vt) * _pred({}, yh, tr & vh)
        elif cfg.get("mix"):
            sub = {k: v for k, v in cfg.items() if k != "mix"}
            yo, vo = labels(p, "oc10")
            a, b = _pred(sub, y_all, tr), _pred(sub, yo, tr & vo)
            d = pd.DataFrame({"date": p.date[ti], "a": a, "b": b})
            pr = (d.groupby("date")["a"].rank(pct=True) + d.groupby("date")["b"].rank(pct=True)).to_numpy("float32") / 2
        elif cfg.get("ens"):
            base = {k: v for k, v in cfg.items() if k != "ens"}
            parts = []
            for sub in cfg["ens"]:
                c2 = dict(base, **sub)
                ys, vs = (y_all, valid) if c2.get("target", target) == target else labels(p, c2["target"])
                parts.append(_pred(c2, ys, tr & vs))
            d = pd.DataFrame({"date": p.date[ti], **{f"p{i}": a for i, a in enumerate(parts)}})
            pr = sum(d.groupby("date")[f"p{i}"].rank(pct=True) for i in range(len(parts))).to_numpy("float32") / len(parts)
        elif cfg.get("blend"):
            a, b = _pred({}, y_all, tr), _pred({"objective": "lambdarank"}, y_all, tr)
            d = pd.DataFrame({"date": p.date[ti], "a": a, "b": b})
            pr = (d.groupby("date")["a"].rank(pct=True) + d.groupby("date")["b"].rank(pct=True)).to_numpy("float32") / 2
        elif cfg.get("n_seeds", 1) > 1:
            pr = np.mean([_pred(cfg, y_all, tr, s) for s in range(cfg["n_seeds"])], axis=0).astype("float32")
        else:
            pr = _pred(cfg, y_all, tr)
        if cfg.get("rules"):
            pr = apply_rules(p, ti, pr)
        outs.append(pd.DataFrame({"date": p.date[ti], "y_up": p.y["y_up"][ti], "p": pr, "y_gap": p.y_gap[ti], "y_oc": p.y_oc[ti], "y_ocx": p.y_ocx[ti], "close": p.close[ti],
                                  "pref": p.X[ti, i_pref], "val": np.expm1(p.X[ti, i_val])}))
        if ckpt is not None:
            tmp = ckpt / f"{m}.parquet.tmp"
            outs[-1].to_parquet(tmp, index=False)
            os.replace(tmp, ckpt / f"{m}.parquet")
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
    prices, flows, universe, groups, market, disc, news, board = features.load_all()
    panel = features.build(prices, flows, universe, groups, market, disclosures=disc, news=news, board=board)
    del prices, flows
    p = Panel(panel)
    del panel
    gc.collect()
    print(f"[exp] panel X {p.X.shape}", flush=True)
    for n in names:
        run(n, p, months)


if __name__ == "__main__":
    main(sys.argv[1:])
