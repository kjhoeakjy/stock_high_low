"""전체 파이프라인: 특징 → 학습·검증 → 리포트."""
from __future__ import annotations

import sys
import time

from stock import features, model, report
from stock.common import DATA


def main(argv):
    t0 = time.time()
    months = 12
    rounds = model.ROUNDS
    if "--fast" in argv:
        months, rounds = 3, 200
    prices, flows, universe, groups, market, disc, news = features.load_all()
    panel = features.build(prices, flows, universe, groups, market, disclosures=disc, news=news)
    del prices, flows
    print(f"[run] panel {panel.shape} in {time.time()-t0:.0f}s; up={int((panel.y_up==1).sum())} dn={int((panel.y_dn==1).sum())}")
    if "--save-panel" in argv:
        panel.to_parquet(DATA / "panel.parquet", index=False)
    last_date = panel["date"].max()
    last = panel[panel["date"] == last_date].copy()
    actual = panel[["code", "date", "ret1", "hi_ret", "lo_ret", "gap"]].copy()
    n_train, train_from = int((panel["y_up"] >= 0).sum()), str(panel["date"].min())
    res = model.run(panel, months=months, rounds=rounds)
    del panel
    import gc; gc.collect()
    report.write(last, actual, res, universe, market, n_train, train_from)
    print(f"[run] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
