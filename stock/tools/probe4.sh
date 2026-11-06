#!/bin/bash
# 종목토론방 = 네이버 댓글(cbox) 서비스. SSR 데이터와 JS 청크에서 호출 인자 추출 후 목록 API 시도
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
C=005930
curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/m
echo "=== dehydrated queries"; python3 - <<'PY'
import re,json
s=open('/tmp/m',encoding='utf-8',errors='ignore').read()
m=re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',s,re.S)
d=json.loads(m.group(1)); qs=d['props']['pageProps']['dehydratedState']['queries']
for q in qs:
    key=json.dumps(q.get('queryKey'),ensure_ascii=False)[:160]
    data=json.dumps(q.get('state',{}).get('data'),ensure_ascii=False)
    print('KEY',key,'| len',len(data)); print('   ',data[:700])
print('rankingToken', d['props']['pageProps'].get('rankingToken'))
PY
echo "=== js chunk cbox params"
js=$(grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/pages/domestic/%5BstockEndType%5D/%5Bcode%5D/discussion-[a-z0-9]+\.js' /tmp/m | head -1); echo "$js"
curl -s "$js" -o /tmp/j; echo "bytes $(wc -c < /tmp/j)"
grep -oE '.{80}(ticket|templateId|pool|objectId|commentBox|cbox|web_naver_list|lang=ko).{120}' /tmp/j | head -20
echo "=== all chunks referencing cbox"; grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/[^"]+\.js' /tmp/m | sort -u > /tmp/chunks; wc -l < /tmp/chunks
while read u; do curl -s "$u" -o /tmp/c; if grep -q 'commentBox\|cbox' /tmp/c; then echo "## $u"; grep -oE '.{60}(ticket|templateId|pool|web_naver_list|commentBox/cbox)[^;]{0,160}' /tmp/c | head -12; fi; done < /tmp/chunks
echo "=== try cbox list (guess)"; for pool in cbox12 cbox5; do for tk in finance stock; do u="https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json?ticket=$tk&templateId=default&pool=$pool&lang=ko&country=KR&objectId=$C&pageSize=20&indexSize=10&page=1&sort=NEW"; curl -s -H "$MO" -H "Referer: https://m.stock.naver.com/domestic/stock/$C/discussion" "$u" -o /tmp/b -w "$pool/$tk http %{http_code} "; head -c 250 /tmp/b | tr '\n' ' '; echo; done; done
echo "=== board.naver 302 location"; curl -s -I -H "$PC" "https://finance.naver.com/item/board.naver?code=$C&page=1" | grep -i '^location\|^HTTP'
