"""메타 라벨링 게이트 실험: 1단계 모형(y_oc10)의 검증 캐시(stock/data/oos_y_oc10.parquet)만으로
'이 추천이 비용 차감 후 이익일 확률'을 2단계 LightGBM으로 학습하고, 재선정·미거래일 규칙의 효과를 월 단위 전진 검증으로 비교.

  python -m stock.meta [--top=20] [--min_train=6]
출력: [meta] {정책별 거래일 평균 순수익·승률·최악일·거래일 비율·누적}
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from stock.common import DATA
from stock.model import COST, tradable

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=0, seed=7)
ROUNDS = 300


def build(oos: pd.DataFrame, top: int) -> pd.DataFrame:
    """일별 상위 top 종목 행 + 당일 교차 단면 변수. 모든 변수는 t일 종가 시점에 알 수 있는 값만."""
    df = tradable(oos).copy()
    df["date"] = df["date"].astype(str)
    g_all = df.groupby("date")
    mkt = pd.DataFrame({"m_breadth": g_all["ret1"].apply(lambda s: float((s > 0).mean())), "m_ret1": g_all["ret1"].mean(),
                        "m_n_p30": g_all["p"].apply(lambda s: int((s > 0.3).sum())), "m_n_p50": g_all["p"].apply(lambda s: int((s > 0.5).sum())),
                        "m_p_q99": g_all["p"].quantile(0.99), "m_n": g_all.size()})
    df = df.sort_values(["date", "p"], ascending=[True, False])
    df["rank"] = df.groupby("date").cumcount() + 1
    t = df[df["rank"] <= top].copy()
    g = t.groupby("date")["p"]
    t["p_top1"] = g.transform("max")
    t["p_top5"] = g.transform(lambda s: s.head(5).mean())
    t["p_top20"] = g.transform("mean")
    t["p_gap5"] = t["p_top5"] - g.transform(lambda s: s.iloc[4] if len(s) > 4 else s.iloc[-1])
    t["p_rel"] = t["p"] / t["p_top1"]
    t["logval"] = np.log1p(t["val"].fillna(0))
    t["dow"] = pd.to_datetime(t["date"]).dt.dayofweek
    t = t.join(mkt, on="date")
    # 시장 흐름: 최근 5·20일 평균 breadth(전진 창, 당일 포함)
    mk = mkt.sort_index()
    for w in (5, 20):
        t[f"m_breadth{w}"] = t["date"].map(mk["m_breadth"].rolling(w, min_periods=1).mean())
        t[f"m_ret{w}"] = t["date"].map(mk["m_ret1"].rolling(w, min_periods=1).sum())
    t["buyable"] = (t["y_gap"] < 0.295)
    t["net"] = np.where(t["buyable"], t["y_ocx"] - COST, 0.0)
    t["label"] = (t["buyable"] & (t["y_ocx"] - COST > 0)).astype("int8")
    t["ym"] = t["date"].str[:6]
    return t


LLM_FEATS = ["llm_has", "llm_dir", "llm_abs", "llm_novelty", "llm_risk"]
FEATS = ["p", "rank", "p_top1", "p_top5", "p_top20", "p_gap5", "p_rel", "ret1", "logval", "dow",
         "m_breadth", "m_ret1", "m_n_p30", "m_n_p50", "m_p_q99", "m_n", "m_breadth5", "m_ret5", "m_breadth20", "m_ret20"]


def add_llm(t: pd.DataFrame) -> pd.DataFrame:
    """LLM 읽기 점수(llm_scores.parquet)가 있으면 병합. 점수 없는 행은 has=0."""
    f = DATA / "llm_scores.parquet"
    if not f.exists():
        return t
    s = pd.read_parquet(f)
    s["date"], s["code"] = s["date"].astype(str), s["code"].astype(str)
    s = s.rename(columns={"has": "llm_has", "dir": "llm_dir", "novelty": "llm_novelty", "risk": "llm_risk"})
    s["llm_abs"] = s["llm_dir"].abs()
    t = t.merge(s[["date", "code"] + LLM_FEATS], on=["date", "code"], how="left")
    t["llm_cov"] = t["llm_has"].notna()
    for c in LLM_FEATS:
        t[c] = t[c].fillna(0)
    return t


def walk(t: pd.DataFrame, min_train: int, feats: list[str] | None = None) -> pd.DataFrame:
    feats = feats or FEATS
    months = sorted(t["ym"].unique())
    out = []
    for i, m in enumerate(months):
        if i < min_train:
            continue
        tr, te = t[t["ym"] < m], t[t["ym"] == m]
        ds = lgb.Dataset(tr[feats].astype("float32").values, tr["label"].values, feature_name=feats, free_raw_data=True)
        mdl = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
        te = te.copy()
        te["q"] = mdl.predict(te[feats].astype("float32").values)
        out.append(te)
        print(f"[meta] {m} train={len(tr)} test={len(te)} pos={tr['label'].mean():.3f}", flush=True)
    return pd.concat(out, ignore_index=True)


def policy(t: pd.DataFrame, pick: str, gate: str | None, k: int = 5, q: float = 0.3, warm: int = 20, thr: float | None = None) -> dict:
    """pick: 'p'(1단계 점수) | 'q'(이익 확률) | 'pq'(곱). gate: None | 'score'(기존 점수 분위) | 'q'(상위 k 평균 이익 확률 분위) | 'q_abs'(thr 절대)."""
    s = t.assign(pq=t["p"] * t["q"]).sort_values(["date", pick], ascending=[True, False]).groupby("date").head(k)
    day = pd.DataFrame({"score": s.groupby("date")["p"].mean(), "qm": s.groupby("date")["q"].mean(),
                        "net": s.groupby("date")["net"].mean()}).sort_index()
    if gate is None:
        day["trade"] = True
    elif gate == "score":
        thr_s = day["score"].expanding().quantile(q).shift(1)
        day["trade"] = day.index.isin(day.index[:warm]) | (day["score"] >= thr_s.fillna(-np.inf))
    elif gate == "q":
        thr_s = day["qm"].expanding().quantile(q).shift(1)
        day["trade"] = day.index.isin(day.index[:warm]) | (day["qm"] >= thr_s.fillna(-np.inf))
    else:
        day["trade"] = day["qm"] >= thr
    tr = day[day["trade"]]
    cum = tr["net"].cumsum()
    mdd = float((cum - cum.cummax()).min()) if len(tr) else 0.0
    return {"pick": pick, "gate": gate or "none", "days": int(len(day)), "trade_frac": round(float(day["trade"].mean()), 3),
            "trade_day": round(float(tr["net"].mean()), 4) if len(tr) else None, "all_day": round(float(day["net"].mean()), 4),
            "win": round(float((tr["net"] > 0).mean()), 3) if len(tr) else None, "worst": round(float(tr["net"].min()), 4) if len(tr) else None,
            "cum": round(float(tr["net"].sum()), 3), "mdd_sum": round(mdd, 3),
            "skip_day": round(float(day.loc[~day["trade"], "net"].mean()), 4) if (~day["trade"]).any() else None}


def main(argv):
    top = next((int(a.split("=")[1]) for a in argv if a.startswith("--top=")), 20)
    min_train = next((int(a.split("=")[1]) for a in argv if a.startswith("--min_train=")), 6)
    oos = pd.read_parquet(DATA / "oos_y_oc10.parquet")
    print(f"[meta] oos rows={len(oos)} days={oos['date'].nunique()} {oos['date'].min()}~{oos['date'].max()}")
    t = build(oos, top)
    if "--llm" in argv:
        t = add_llm(t)
        cov = t[t.get("llm_cov", pd.Series(False, index=t.index))]
        if len(cov):
            dates = sorted(cov["date"].unique())
            print(f"[meta] LLM 점수 구간 {dates[0]}~{dates[-1]} {len(dates)}일 · 판독 있음 {int(cov['llm_has'].sum())}/{len(cov)}")
            print("[meta] LLM 방향별 상위 20 평균 순수익·적중:")
            print(cov.groupby("llm_dir").agg(n=("net", "size"), net=("net", "mean"), win=("label", "mean")).round(4).to_string())
            # 구간 내 비교: 기본 변수 vs 기본+LLM (학습은 구간 이전 달 포함 전체 캐시, 평가는 LLM 구간)
            a = walk(t, min_train)
            b = walk(t, min_train, FEATS + LLM_FEATS)
            for name, r in (("meta_base", a), ("meta_llm", b)):
                sub = r[r["date"].isin(set(dates))]
                for pick, gate in (("pq", "q"), ("q", None), ("pq", None)):
                    print(f"[meta] {name} {pick}/{gate}", json.dumps(policy(sub, pick, gate), ensure_ascii=False))
            print("[meta] 기준(1단계 p, 점수 게이트)", json.dumps(policy(a[a["date"].isin(set(dates))], "p", "score"), ensure_ascii=False))
            return
    t = walk(t, min_train)
    print(f"[meta] 평가 구간 {t['date'].min()}~{t['date'].max()} 일수 {t['date'].nunique()} 메타 AUC 대용(이익 확률 분위별 실제 승률):")
    t["qb"] = pd.qcut(t["q"], 5, labels=False, duplicates="drop")
    print(t.groupby("qb").agg(n=("label", "size"), win=("label", "mean"), net=("net", "mean"), q=("q", "mean")).round(4).to_string())
    res = [policy(t, "p", None), policy(t, "p", "score"), policy(t, "p", "q"), policy(t, "q", None), policy(t, "q", "q"),
           policy(t, "pq", None), policy(t, "pq", "q"), policy(t, "pq", "score")]
    for thr in (0.3, 0.35, 0.4, 0.45):
        res.append(policy(t, "pq", "q_abs", thr=thr))
    for r in res:
        print("[meta]", json.dumps(r, ensure_ascii=False))
    # 연도별(pq + q 게이트 vs 기준)
    for name, (pick, gate) in {"base": ("p", "score"), "meta": ("pq", "q")}.items():
        for y, sub in t.groupby(t["date"].str[:4]):
            r = policy(sub, pick, gate)
            print(f"[meta] {name} {y} trade_day={r['trade_day']} win={r['win']} worst={r['worst']} frac={r['trade_frac']}")


if __name__ == "__main__":
    main(sys.argv[1:])
