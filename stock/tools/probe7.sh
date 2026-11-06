#!/bin/bash
# 토론방: 네이버 자체 토론 API 경로를 PC·모바일 페이지와 JS에서 읽는다(추측 호출 없음)
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
C=005930
mkdir -p /tmp/pr
curl -s -H "$PC" "https://stock.naver.com/domestic/stock/$C/discussion" -o /tmp/pr/pc
curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/pr/m
for f in pc m; do echo "=== $f page $(wc -c </tmp/pr/$f) bytes: dehydrated keys"; python3 - /tmp/pr/$f <<'PY'
import re,sys,json
s=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',s,re.S)
if not m: print('no __NEXT_DATA__'); sys.exit()
d=json.loads(m.group(1))
qs=d.get('props',{}).get('pageProps',{}).get('dehydratedState',{}).get('queries',[])
for q in qs:
    k=json.dumps(q.get('queryKey'),ensure_ascii=False)
    data=json.dumps(q.get('state',{}).get('data'),ensure_ascii=False)
    print('KEY',k[:200],'| len',len(data))
    if 'iscussion' in k or 'post' in k.lower() or 'comment' in k.lower(): print('   ',data[:1200])
# 페이지 안의 API 경로 문자열
for x in sorted(set(re.findall(r'"(/[A-Za-z0-9_/{}\-]*(?:discussion|post|comment|opentalk)[A-Za-z0-9_/{}\-]*)"',s))): print('  path',x)
PY
done
echo "=== JS chunk strings (both pages)"
cat /tmp/pr/pc /tmp/pr/m | grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/[^"]+\.js' | sort -u > /tmp/pr/urls
echo "chunks $(wc -l </tmp/pr/urls)"
: > /tmp/pr/all.txt
i=0; while read -r u; do i=$((i+1)); curl -s "$u" -o /tmp/pr/j$i.js; echo "## $u" >> /tmp/pr/all.txt; cat /tmp/pr/j$i.js >> /tmp/pr/all.txt; echo >> /tmp/pr/all.txt; done < /tmp/pr/urls
python3 - <<'PY'
import re
s=open('/tmp/pr/all.txt',encoding='utf-8',errors='ignore').read()
paths=sorted(set(re.findall(r'"(/[A-Za-z0-9_/{}\-\$]*(?:discussion|post|comment|opentalk|board)[A-Za-z0-9_/{}\-\$]*)"',s)))
print('paths',len(paths)); [print('  ',p) for p in paths[:80]]
hosts=sorted(set(re.findall(r'"(https?://[a-z0-9.-]*naver\.com[^"]{0,60})"',s)))
print('hosts',len(hosts)); [print('  ',h) for h in hosts if 'api' in h or 'stock' in h][:60]
for kw in ('/discussion/comment/counts','discussion/posts','/posts','baseURL','front-api'):
    ms=list(re.finditer(re.escape(kw),s)); print(f'[{kw}] {len(ms)}')
    for m in ms[:3]: print('    ',s[max(0,m.start()-250):m.start()+250].replace('\n',' '))
PY
