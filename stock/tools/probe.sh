#!/bin/bash
# Actions에서 네이버 묶음 시세 형식 점검
H1='User-Agent: Mozilla/5.0'; H2='Referer: https://m.stock.naver.com/'
codes=$(curl -s -H "$H1" -H "$H2" "https://m.stock.naver.com/api/stocks/marketValue/KOSDAQ?page=1&pageSize=100" | python3 -c "import sys,json;print(','.join(x['itemCode'] for x in json.load(sys.stdin)['stocks']))")
c100=$codes; c20=$(echo "$codes" | cut -d, -f1-20)
show() { python3 - "$1" <<'PY'
import json,sys
raw=open(sys.argv[1]).read()
try:
    d=json.loads(raw)
except Exception:
    print(raw[:300]); sys.exit()
ds=d.get('datas') if isinstance(d,dict) else None
if ds is None and isinstance(d,dict): ds=d.get('result',{}).get('areas',[{}])[0].get('datas')
if ds is None: print(json.dumps(d,ensure_ascii=False)[:800])
else:
    print('n=',len(ds)); print(json.dumps(ds[0],ensure_ascii=False)[:1200] if ds else '')
PY
}
echo "=== realtime/domestic/stock 20"; curl -s -H "$H1" "https://polling.finance.naver.com/api/realtime/domestic/stock/$c20" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
echo "=== realtime/domestic/stock 100"; curl -s -H "$H1" "https://polling.finance.naver.com/api/realtime/domestic/stock/$c100" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
q=$(echo "$c20" | tr ',' '\n' | sed 's/^/SERVICE_ITEM:/' | paste -sd'|' | sed 's/|/%7C/g')
echo "=== realtime?query %7C 20"; curl -s -H "$H1" "https://polling.finance.naver.com/api/realtime?query=$q" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
q2=$(echo "$c20" | tr ',' '\n' | sed 's/^/SERVICE_ITEM:/' | paste -sd',')
echo "=== realtime?query comma 20"; curl -s -H "$H1" "https://polling.finance.naver.com/api/realtime?query=$q2" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
echo "=== NXT polling 100"; curl -s -H "$H1" -H "$H2" "https://stock.naver.com/api/polling/domestic/NXT/stock?itemCodes=$c100" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
echo "=== KRX polling 100"; curl -s -H "$H1" -H "$H2" "https://stock.naver.com/api/polling/domestic/KRX/stock?itemCodes=$c100" -o /tmp/b -w "%{http_code}\n"; show /tmp/b
echo "=== up list sample"; curl -s -H "$H1" -H "$H2" "https://m.stock.naver.com/api/stocks/up/KOSDAQ?page=1&pageSize=2" -o /tmp/b -w "%{http_code}\n"; python3 -c "import json;d=json.load(open('/tmp/b'));print(json.dumps(d['stocks'][0],ensure_ascii=False)[:1500])"
echo "=== index"; curl -s -H "$H1" -H "$H2" "https://m.stock.naver.com/api/index/KOSDAQ/basic" | head -c 500; echo
