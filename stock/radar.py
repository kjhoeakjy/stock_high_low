"""장중 실시간 레이더 (PC에서 실행, 표준 라이브러리만 사용).

    python radar.py            → http://localhost:8787 자동 열림
    python radar.py --port 9000

7초마다 네이버 시세로 코스피·코스닥 상승·하락 상위 종목을 읽어
상한가·하한가까지 남은 거리, 신규 상한가·하한가 진입/이탈, 거래대금 급증을 표시한다.
아침 예측(latest.json)이 있으면 후보 표시를 겹쳐 보여준다. 투자 권유 아님.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
      "Referer": "https://m.stock.naver.com/"}
NAVER = "https://m.stock.naver.com/api"
PAGES_URL = "https://kjhoeakjy.github.io/stock_high_low/latest.json"
INTERVAL = 7
STATE = {"ts": None, "rows": [], "events": [], "market": {}, "morning": {}, "error": None}
LOCK = threading.Lock()


def get(url, timeout=15):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        STATE["error"] = f"{time.strftime('%H:%M:%S')} {url.split('/api/')[-1][:40]}: {e}"
        return None


def num(s):
    try:
        return float(str(s).replace(",", "").replace("+", ""))
    except Exception:  # noqa: BLE001
        return None


def tick(p):
    return 1 if p < 2000 else 5 if p < 5000 else 10 if p < 20000 else 50 if p < 50000 else 100 if p < 200000 else 500 if p < 500000 else 1000


def limits(prev):
    up = prev * 1.3
    dn = prev * 0.7
    up = math.floor(up / tick(up)) * tick(up)
    dn = math.ceil(dn / tick(dn)) * tick(dn)
    return up, dn


def fetch_lists():
    rows = {}
    for mkt in ("KOSPI", "KOSDAQ"):
        for kind in ("up", "down"):
            for page in (1, 2, 3):
                j = get(f"{NAVER}/stocks/{kind}/{mkt}?page={page}&pageSize=100")
                if not j or not j.get("stocks"):
                    break
                stop = False
                for s in j["stocks"]:
                    if s.get("stockEndType") != "stock":
                        continue
                    price = num(s.get("closePriceRaw"))
                    chg = num(s.get("compareToPreviousClosePriceRaw"))
                    if not price or chg is None:
                        continue
                    prev = price - chg
                    if prev <= 0:
                        continue
                    ret = price / prev - 1
                    if abs(ret) < 0.10:
                        stop = True  # 정렬된 목록이므로 10% 미만부터는 불필요
                        break
                    up, dn = limits(prev)
                    rows[s["itemCode"]] = {
                        "code": s["itemCode"], "name": s["stockName"], "mkt": "코스피" if mkt == "KOSPI" else "코스닥",
                        "price": price, "prev": prev, "ret": ret, "up": up, "dn": dn,
                        "to_up": up / price - 1, "to_dn": dn / price - 1,
                        "at_up": price >= up, "at_dn": price <= dn,
                        "value": (num(s.get("accumulatedTradingValueRaw")) or 0) / 1e8,
                        "mcap": (num(s.get("marketValueRaw")) or 0) / 1e8,
                        "halt": (s.get("tradeStopType") or {}).get("name") == "HALTED",
                        "t": (s.get("localTradedAt") or "")[11:19],
                    }
                if stop or len(j["stocks"]) < 100:
                    break
    return rows


def fetch_market():
    out = {}
    for idx in ("KOSPI", "KOSDAQ"):
        j = get(f"{NAVER}/index/{idx}/basic")
        if j:
            out[idx] = {"v": num(j.get("closePrice")), "r": (num(j.get("fluctuationsRatio")) or 0) / 100,
                        "status": j.get("marketStatus")}
    return out


def loop():
    prev_rows = {}
    last_market = 0
    while True:
        t0 = time.time()
        rows = fetch_lists()
        events = []
        now = time.strftime("%H:%M:%S")
        for c, r in rows.items():
            p = prev_rows.get(c)
            if r["at_up"] and not (p and p["at_up"]):
                events.append({"t": now, "kind": "up", "text": f'{r["name"]} 상한가 진입 {r["price"]:,.0f}원'})
            elif p and p["at_up"] and not r["at_up"]:
                events.append({"t": now, "kind": "warn", "text": f'{r["name"]} 상한가 풀림 {r["ret"]*100:+.1f}%'})
            if r["at_dn"] and not (p and p["at_dn"]):
                events.append({"t": now, "kind": "dn", "text": f'{r["name"]} 하한가 진입 {r["price"]:,.0f}원'})
            elif p and p["at_dn"] and not r["at_dn"]:
                events.append({"t": now, "kind": "warn", "text": f'{r["name"]} 하한가 풀림 {r["ret"]*100:+.1f}%'})
            if not r["at_up"] and r["to_up"] <= 0.03 and not (p and p["to_up"] <= 0.03):
                events.append({"t": now, "kind": "near", "text": f'{r["name"]} 상한가까지 {r["to_up"]*100:.1f}%'})
            if not r["at_dn"] and r["to_dn"] >= -0.03 and not (p and p["to_dn"] >= -0.03):
                events.append({"t": now, "kind": "near", "text": f'{r["name"]} 하한가까지 {abs(r["to_dn"])*100:.1f}%'})
        prev_rows = rows
        market = STATE["market"]
        if time.time() - last_market > 30:
            market = fetch_market() or market
            last_market = time.time()
        with LOCK:
            STATE["ts"] = now
            STATE["rows"] = sorted(rows.values(), key=lambda r: -r["ret"])
            STATE["events"] = (events + STATE["events"])[:200]
            STATE["market"] = market
        time.sleep(max(1, INTERVAL - (time.time() - t0)))


def load_morning():
    j = get(PAGES_URL + f"?{int(time.time())}")
    if j:
        with LOCK:
            STATE["morning"] = {"target_date": j.get("target_date"),
                                "up": {x["code"]: x["p"] for x in j.get("up", [])}}


HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>상한가 레이더</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--fg:#f2f3f5;--mut:#8b919c;--up:#ff4d4d;--dn:#4d8cff;--card:#171a21;--line:#262a33}
body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,"Malgun Gothic","Apple SD Gothic Neo",sans-serif;font-size:20px}
.wrap{max-width:1300px;margin:0 auto;padding:20px 16px}
h1{font-size:30px;margin:0 0 4px}.sub{color:var(--mut);font-size:16px;margin-bottom:16px}
.mk{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}.mk div{background:#222631;border-radius:10px;padding:8px 12px;font-size:17px}
.pos{color:var(--up)}.neg{color:var(--dn)}
.cols{display:grid;grid-template-columns:1.2fr 1.2fr 1fr;gap:16px}@media(max-width:1000px){.cols{grid-template-columns:1fr}}
h2{font-size:22px;margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-size:18px}td,th{padding:6px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:14px}td:first-child,th:first-child{text-align:left}
.lim{font-weight:800}.bar{height:4px;background:#2a2f3a;border-radius:2px;overflow:hidden;margin-top:3px}.bar i{display:block;height:100%}
.ev{font-size:17px;line-height:1.5;max-height:80vh;overflow:auto}.ev div{padding:4px 0;border-bottom:1px solid var(--line)}
.ev b{color:var(--mut);font-weight:400;font-size:14px;margin-right:8px}
.tag{font-size:12px;border-radius:5px;padding:1px 5px;margin-left:5px;vertical-align:middle;background:#3a2a2a;color:#ffb3b3}
.tag.m{background:#2a3350;color:#b3c8ff}
.foot{color:var(--mut);font-size:14px;margin-top:20px}
</style></head><body><div class="wrap">
<h1>상한가 · 하한가 레이더</h1><div class="sub" id="sub">연결 중</div><div class="mk" id="mk"></div>
<div class="cols">
<section><h2 class="pos">상승 (상한가까지 거리)</h2><table id="up"></table></section>
<section><h2 class="neg">하락 (하한가까지 거리)</h2><table id="dn"></table></section>
<section><h2>방금 일어난 일</h2><div class="ev" id="ev"></div></section>
</div>
<div class="foot">네이버 시세 7초 주기. 투자 권유 아님. ★는 오늘 아침 모델 후보(확률).</div></div>
<script>
const pct=(v,d=1)=>(v*100).toFixed(d)+'%';const sg=v=>(v>0?'+':'')+(v*100).toFixed(1)+'%';
let lastEv='';
async function tick(){
 try{const d=await (await fetch('/data')).json();
 document.getElementById('sub').textContent=(d.ts?`${d.ts} 갱신 · ${d.rows.length}종목 추적`:'대기')+(d.error?' · '+d.error:'');
 const m=d.market;document.getElementById('mk').innerHTML=Object.entries(m).map(([k,v])=>`<div><b>${k=='KOSPI'?'코스피':'코스닥'}</b> ${v.v?v.v.toLocaleString():''} <span class="${v.r>0?'pos':'neg'}">${sg(v.r)}</span></div>`).join('')
  +`<div>상한가 <b>${d.rows.filter(r=>r.at_up).length}</b></div><div>하한가 <b>${d.rows.filter(r=>r.at_dn).length}</b></div>`;
 const star=(r,k)=>{const p=d.morning&&d.morning[k]&&d.morning[k][r.code];return p!=null?`<span class="tag ${k=='dn'?'m':''}">★ ${pct(p,0)}</span>`:''};
 const row=(r,k)=>{const dist=k=='up'?r.to_up:-r.to_dn;const at=k=='up'?r.at_up:r.at_dn;const col=k=='up'?'var(--up)':'var(--dn)';
  return `<tr><td>${r.name}${star(r,k)}${r.halt?'<span class="tag">정지</span>':''}<div style="color:var(--mut);font-size:13px">${r.mkt} · 시총 ${r.mcap>=10000?(r.mcap/10000).toFixed(1)+'조':Math.round(r.mcap).toLocaleString()+'억'}</div></td>
  <td>${r.price.toLocaleString()}</td><td class="${k=='up'?'pos':'neg'}" style="font-weight:700">${sg(r.ret)}</td>
  <td class="${at?'lim':''}" style="color:${at?col:'inherit'}">${at?(k=='up'?'상한가':'하한가'):pct(dist)}<div class="bar"><i style="width:${Math.max(0,Math.min(100,(1-dist/0.2)*100))}%;background:${col}"></i></div></td><td>${Math.round(r.value).toLocaleString()}억</td></tr>`};
 const head='<tr><th>종목</th><th>현재가</th><th>등락</th><th>남은 거리</th><th>거래대금</th></tr>';
 document.getElementById('up').innerHTML=head+d.rows.filter(r=>r.ret>0).slice(0,40).map(r=>row(r,'up')).join('');
 document.getElementById('dn').innerHTML=head+d.rows.filter(r=>r.ret<0).sort((a,b)=>a.ret-b.ret).slice(0,40).map(r=>row(r,'dn')).join('');
 document.getElementById('ev').innerHTML=d.events.map(e=>`<div class="${e.kind=='up'?'pos':e.kind=='dn'?'neg':''}"><b>${e.t}</b>${e.text}</div>`).join('');
 if(d.events.length&&d.events[0].t+d.events[0].text!==lastEv){lastEv=d.events[0].t+d.events[0].text;if(d.events[0].kind!='near')beep();}
 }catch(e){document.getElementById('sub').textContent='서버 연결 끊김 '+e;}
}
function beep(){try{const a=new (window.AudioContext||window.webkitAudioContext)();const o=a.createOscillator();o.connect(a.destination);o.frequency.value=880;o.start();o.stop(a.currentTime+0.12);}catch(e){}}
tick();setInterval(tick,3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            with LOCK:
                body = json.dumps(STATE, ensure_ascii=False).encode()
            ctype = "application/json; charset=utf-8"
        else:
            body = HTML.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main(argv):
    port = 8787
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=lambda: (load_morning(), [time.sleep(600) or load_morning() for _ in iter(int, 1)]), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"레이더 실행 중: http://localhost:{port}  (종료: Ctrl+C)")
    if "--no-browser" not in argv:
        webbrowser.open(f"http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
