"""2단계 재순위 실험: 1단계 LightGBM(y_oc10) 검증 캐시의 일별 상위 K 후보만 대상으로
(a) 원시 60일 시퀀스 GRU, (b) 1단계 확률·GRU 점수·교차 단면 변수를 합친 2단계 LightGBM을 월 단위 전진 검증으로 학습하고
상위 5 선정 성능(적중률·순수익·승률·최악일)을 1단계 단독과 비교한다. CPU torch.

  python -m stock.rerank [--top=200] [--seq=60] [--min_train=6] [--refit=3] [--epochs=4] [--backend=gru|tabpfn|lgb|mlp] [--feats=seq|panel]
  --feats=panel: 1단계와 같은 258변수를 후보 행에 붙여 모형만 바꿔 비교(공정 비교). 패널 생성에 전체 자료(stock-data) 필요.
출력: [rerank] {...} JSON 줄
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from stock.common import DATA
from stock.model import COST, tradable

SEQ_FEATS = ["r_close", "gap", "range", "pos", "dvol", "turn"]


def candidates(top: int) -> pd.DataFrame:
    oos = pd.read_parquet(DATA / "oos_y_oc10.parquet")
    df = tradable(oos).copy()
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["date", "p"], ascending=[True, False])
    df["rank"] = df.groupby("date").cumcount() + 1
    c = df[df["rank"] <= top].copy()
    g = c.groupby("date")["p"]
    c["p_top1"], c["p_top5"], c["p_top20"] = g.transform("max"), g.transform(lambda s: s.head(5).mean()), g.transform(lambda s: s.head(20).mean())
    c["p_rel"] = c["p"] / c["p_top1"]
    c["logval"] = np.log1p(c["val"].fillna(0))
    c["buyable"] = c["y_gap"] < 0.295
    c["net"] = np.where(c["buyable"], c["y_ocx"] - COST, 0.0)
    c["label"] = (c["buyable"] & (c["y_ocx"] >= 0.10)).astype("float32")
    c["ym"] = c["date"].str[:6]
    return c.reset_index(drop=True)


def sequences(c: pd.DataFrame, seq: int) -> np.ndarray:
    """후보별 (seq, 6) 시퀀스: 종가 로그수익, 시가 갭, 고저 범위, 종가 위치, 거래량 로그 변화, 회전 대용(거래대금/60일 평균). 종목 내 표준화."""
    pr = pd.read_parquet(DATA / "prices.parquet", columns=["code", "date", "open", "high", "low", "close", "volume"])
    pr = pr[pr["code"].isin(set(c["code"]))].sort_values(["code", "date"])
    out = np.zeros((len(c), seq, len(SEQ_FEATS)), "float32")
    pos_map = {}
    for code, g in pr.groupby("code", sort=False):
        o, h, l, cl, v = (g[k].to_numpy("float64") for k in ("open", "high", "low", "close", "volume"))
        cl_prev = np.r_[cl[0], cl[:-1]]
        amt = cl * v
        f = np.column_stack([
            np.log(np.clip(cl / np.clip(cl_prev, 1, None), 1e-3, 1e3)),
            np.log(np.clip(o / np.clip(cl_prev, 1, None), 1e-3, 1e3)),
            (h - l) / np.clip(cl, 1, None),
            np.where(h > l, (cl - l) / np.clip(h - l, 1e-9, None), 0.5) - 0.5,
            np.log1p(v) - np.log1p(np.r_[v[0], v[:-1]]),
            np.log1p(amt) - np.log1p(pd.Series(amt).rolling(60, min_periods=5).mean().bfill().to_numpy()),
        ]).astype("float32")
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        pos_map[code] = (g["date"].to_numpy(), f)
    for i, (code, d) in enumerate(zip(c["code"].to_numpy(), c["date"].to_numpy())):
        if code not in pos_map:
            continue
        dates, f = pos_map[code]
        j = np.searchsorted(dates, d, side="right")  # d 포함 이전 구간
        s = f[max(0, j - seq):j]
        if len(s):
            out[i, seq - len(s):] = s
    mu, sd = out.reshape(-1, out.shape[-1]).mean(0), out.reshape(-1, out.shape[-1]).std(0) + 1e-6
    return (out - mu) / sd


def panel_features(c: pd.DataFrame) -> np.ndarray:
    """후보 행에 1단계와 같은 특징 행렬(features.build, 258변수)을 붙인다. 공정 비교용: 같은 입력으로 모형만 바꿔 본다."""
    from stock import features
    from stock.model import Panel
    prices, flows, universe, groups, market, disc, news, board = features.load_all()
    df = features.build(prices, flows, universe, groups, market, disclosures=disc, news=news, board=board)
    del prices, flows
    p = Panel(df)
    del df
    key = pd.Series(np.arange(len(p.date)), index=pd.MultiIndex.from_arrays([p.code, p.date]))
    idx = key.reindex(pd.MultiIndex.from_arrays([c["code"].to_numpy(), c["date"].to_numpy()])).to_numpy()
    ok = ~np.isnan(idx)
    out = np.full((len(c), p.X.shape[1]), np.nan, "float32")
    out[ok] = p.X[idx[ok].astype(int)]
    print(f"[rerank] 패널 변수 {p.X.shape[1]}개 · 후보 매칭 {ok.mean():.3f}", flush=True)
    return out


class _LGB:
    """같은 후보·같은 변수에 LightGBM(1단계와 비슷한 설정)을 학습: TabPFN과의 모형 비교 기준."""
    def __init__(self, X, y, seed):
        X2 = X.reshape(len(X), -1)
        ds = lgb.Dataset(X2, y, free_raw_data=True)
        self.m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                                bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=seed, num_threads=8), ds, num_boost_round=600)

    def predict(self, X):
        return self.m.predict(X.reshape(len(X), -1)).astype("float32")


class _MLP:
    """torch가 없을 때의 대체: 시퀀스를 펼친 sklearn MLP(코드 경로 점검·저사양 환경용)."""
    def __init__(self, X, y, epochs, seed):
        from sklearn.neural_network import MLPClassifier
        self.m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=max(20, epochs * 5), random_state=seed, early_stopping=False)
        self.m.fit(np.nan_to_num(X.reshape(len(X), -1)), y)

    def predict(self, X):
        return self.m.predict_proba(np.nan_to_num(X.reshape(len(X), -1)))[:, 1]


class _TabPFN:
    """TabPFN v2(CPU): 시퀀스 요약 통계(각 변수의 최근 1·5·20·60일 평균·표준편차)를 표 입력으로, 컨텍스트는 최근 표본 n_ctx행(양성 전부+음성 표본)."""
    def __init__(self, X, y, seed, n_ctx=10000):
        from tabpfn import TabPFNClassifier
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[rerank] TabPFN device={dev}", flush=True)
        self.m = TabPFNClassifier(device=dev, n_estimators=4, random_state=seed, ignore_pretraining_limits=True)
        F = self.flat(X)
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y > 0)
        neg = np.flatnonzero(y <= 0)
        n_neg = max(0, n_ctx - len(pos))
        if len(pos) > n_ctx // 2:
            pos = rng.choice(pos, n_ctx // 2, replace=False)
            n_neg = n_ctx - len(pos)
        neg = neg[-n_neg * 3:] if len(neg) > n_neg * 3 else neg  # 최근 쪽 우선
        neg = rng.choice(neg, min(n_neg, len(neg)), replace=False)
        idx = np.sort(np.r_[pos, neg])
        self.m.fit(F[idx], y[idx].astype(int))

    @staticmethod
    def flat(X):
        if X.ndim == 2:
            return X
        parts = []
        for w in (1, 5, 20, 60):
            seg = X[:, -w:, :]
            parts.append(seg.mean(1))
            if w > 1:
                parts.append(seg.std(1))
        return np.concatenate(parts, axis=1).astype("float32")

    def predict(self, X):
        F = self.flat(X)
        out = np.zeros(len(F), "float32")
        for i in range(0, len(F), 2000):
            out[i:i + 2000] = self.m.predict_proba(F[i:i + 2000])[:, 1]
        return out


def fit_gru(X: np.ndarray, y: np.ndarray, epochs: int, seed: int = 0):
    if BACKEND.get("name") == "tabpfn":
        return _TabPFN(X, y, seed, BACKEND.get("n_ctx", 10000))
    if BACKEND.get("name") == "lgb":
        return _LGB(X, y, seed)
    if BACKEND.get("name") == "mlp":
        return _MLP(X, y, epochs, seed)
    if X.ndim == 2:
        raise SystemExit("[rerank] 패널 변수 입력(--feats=panel)은 backend tabpfn|lgb|mlp만 지원")
    try:
        import torch
    except ImportError:
        print("[rerank] torch 없음 → MLP 대체", flush=True)
        return _MLP(X, y, epochs, seed)
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(X.shape[-1], 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    net = Net()
    pos = max(1.0, (1 - y.mean()) / max(y.mean(), 1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(pos, 10.0)))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"[rerank]   epoch {ep + 1} loss {tot / n:.4f}", flush=True)
    net.eval()
    return net


BACKEND: dict = {}


def predict_gru(net, X: np.ndarray) -> np.ndarray:
    if isinstance(net, (_MLP, _TabPFN, _LGB)):
        return net.predict(X)
    import torch
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X))).numpy()


STACK = ["p", "rank", "p_top1", "p_top5", "p_top20", "p_rel", "ret1", "logval", "gru"]


def walk(c: pd.DataFrame, X: np.ndarray, min_train: int, refit: int, epochs: int) -> pd.DataFrame:
    months = sorted(c["ym"].unique())
    out, net, last_fit = [], None, -999
    for i, m in enumerate(months):
        if i < min_train:
            continue
        if net is None or i - last_fit >= refit:
            tr = np.flatnonzero(c["ym"] < m)
            t0 = time.time()
            net = fit_gru(X[tr], c["label"].to_numpy()[tr], epochs, seed=i)
            last_fit = i
            print(f"[rerank] {m} GRU 학습 {len(tr)}건 {time.time() - t0:.0f}s", flush=True)
        te = np.flatnonzero(c["ym"] == m)
        sub = c.iloc[te].copy()
        sub["gru"] = predict_gru(net, X[te])
        out.append(sub)
    r = pd.concat(out, ignore_index=True)
    # 2단계 LightGBM(스태킹): 이전 달까지의 (1단계 확률·GRU 점수·교차 단면) → 라벨
    r["stack"] = np.nan
    ms = sorted(r["ym"].unique())
    for i, m in enumerate(ms):
        if i < 3:
            continue
        tr, te = r[r["ym"] < m], r["ym"] == m
        ds = lgb.Dataset(tr[STACK].astype("float32").values, tr["label"].values, feature_name=STACK, free_raw_data=True)
        mdl = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7), ds, num_boost_round=300)
        r.loc[te, "stack"] = mdl.predict(r.loc[te, STACK].astype("float32").values)
    return r


def evaluate(r: pd.DataFrame, score: str, k: int = 5) -> dict:
    s = r.dropna(subset=[score]).sort_values(["date", score], ascending=[True, False]).groupby("date").head(k)
    day = pd.DataFrame({"net": s.groupby("date")["net"].mean(), "hit": s.groupby("date")["label"].mean(),
                        "unbuy": s.groupby("date")["buyable"].apply(lambda b: 1 - b.mean())}).sort_index()
    cum = day["net"].cumsum()
    return {"score": score, "days": int(len(day)), "p@5": round(float(day["hit"].mean()), 4), "net_day": round(float(day["net"].mean()), 4),
            "win": round(float((day["net"] > 0).mean()), 3), "worst": round(float(day["net"].min()), 4), "cum": round(float(day["net"].sum()), 3),
            "mdd_sum": round(float((cum - cum.cummax()).min()), 3), "unbuyable": round(float(day["unbuy"].mean()), 4)}


def main(argv):
    arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    top, seq, min_train, refit, epochs = int(arg("top", 200)), int(arg("seq", 60)), int(arg("min_train", 6)), int(arg("refit", 3)), int(arg("epochs", 4))
    BACKEND.update(name=arg("backend", "gru"), n_ctx=int(arg("n_ctx", 10000)), feats=arg("feats", "seq"))
    print(f"[rerank] backend={BACKEND['name']} feats={BACKEND['feats']}")
    c = candidates(top)
    print(f"[rerank] 후보 {len(c)}건 · 일수 {c['date'].nunique()} · 양성 {c['label'].mean():.3f}")
    t0 = time.time()
    X = panel_features(c) if BACKEND["feats"] == "panel" else sequences(c, seq)
    print(f"[rerank] 입력 {X.shape} {time.time() - t0:.0f}s")
    r = walk(c, X, min_train, refit, epochs)
    r["rk_p"] = r.groupby("date")["p"].rank(ascending=False)
    r["rk_g"] = r.groupby("date")["gru"].rank(ascending=False)
    r["avg_rank"] = -(r["rk_p"] + r["rk_g"])
    r["p_x_gru"] = r["p"] * r["gru"]
    sub = r.dropna(subset=["stack"])
    for sc in ("p", "gru", "avg_rank", "p_x_gru", "stack"):
        print("[rerank]", json.dumps(evaluate(sub, sc), ensure_ascii=False))
    for y, g in sub.groupby(sub["date"].str[:4]):
        print(f"[rerank] {y} " + " · ".join(f"{sc}: p@5 {evaluate(g, sc)['p@5']} net {evaluate(g, sc)['net_day']}" for sc in ("p", "avg_rank", "stack")))
    r[["date", "code", "p", "gru", "stack", "label", "net"]].to_parquet(DATA / f"rerank_oos_{BACKEND['name']}_{BACKEND['feats']}.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
