"""장기(36개월) 워크포워드 검증: 매일 학습과 같은 절차로 매매 모델(y_oc10)을 월 단위 재학습해
1·6·12·36개월 성능을 docs/backtest.json에 기록한다(주 1회, backtest.yml).

  python -m stock.backtest [--months=36]
"""
from __future__ import annotations

import datetime as dt
import gc
import sys
import time

from stock import crypt, features, model
from stock.common import OUT


def main(argv):
    t0 = time.time()
    months = 36
    for a in argv:
        if a.startswith("--months="):
            months = int(a.split("=", 1)[1])
    prices, flows, universe, groups, market, disc, news, board = features.load_all()
    panel = features.build(prices, flows, universe, groups, market, disclosures=disc, news=news, board=board)
    del prices, flows
    gc.collect()
    print(f"[backtest] panel {panel.shape} in {time.time()-t0:.0f}s")
    p = model.Panel(panel)
    del panel
    gc.collect()
    oos = model.walk_forward(p, "y_oc10", months)
    del p
    gc.collect()
    hz = model.summarize_horizons(oos, "y_oc10")
    doc = {"generated_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M"),
           "as_of": str(oos["date"].max()), "months": months, "target": "y_oc10", "horizons": hz}
    for h, st in hz.items():
        sim = st.get("sim") or {}
        print(f"[backtest] {h}개월: days={st['days']} p@5={st.get('p@5')} oc@5={st.get('oc_mean_ex_unbuyable@5')} "
              f"net_day={(st.get('net') or {}).get('trade_day')} sim_final={sim.get('final')} fixed={sim.get('fixed_pnl')} mdd={sim.get('mdd')}", flush=True)
    crypt.write_json(OUT / "backtest.json", doc)
    print(f"[backtest] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
