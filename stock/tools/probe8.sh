#!/bin/bash
# 토론방: JS에서 토론 목록 API의 인자 이름을 읽고, 그 이름으로 공개 목록을 1회씩 조회
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
C=005930
mkdir -p /tmp/pr
curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/pr/m
grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/[^"]+\.js' /tmp/pr/m | sort -u > /tmp/pr/urls
: > /tmp/pr/all.txt
i=0; while read -r u; do i=$((i+1)); curl -s "$u" >> /tmp/pr/all.txt; echo >> /tmp/pr/all.txt; done < /tmp/pr/urls
python3 - <<'PY'
import re
s=open('/tmp/pr/all.txt',encoding='utf-8',errors='ignore').read()
for kw in ('"/discussion/list"','"/discussion/list/itemCodes"','"/discussion/rankings/itemCodes"','"/discussion/list/hot/itemCodes"','"/discussion/detail/view"','"/discussion/list/hot"','DISCUSSION_LIST:','itemCode:','discussionType','sortType','excludeNews'):
    ms=list(re.finditer(re.escape(kw),s)); print(f'[{kw}] {len(ms)}')
    for m in ms[:3]: print('    ',s[max(0,m.start()-300):m.start()+400].replace('\n',' '))
PY
echo "=== calls"
for q in "discussion/list?itemCode=$C&stockEndType=stock" "discussion/list?itemCode=$C&stockEndType=stock&pageSize=20" "discussion/list/itemCodes?itemCodes=$C" "discussion/rankings/itemCodes?itemCodes=$C" "discussion/list/hot/itemCodes?itemCodes=$C"; do
  for base in "https://m.stock.naver.com/front-api" "https://m.stock.naver.com/api"; do
    echo "-- $base/$q"; curl -s -H "$MO" -H "Referer: https://m.stock.naver.com/domestic/stock/$C/discussion" "$base/$q" -w " [http %{http_code}]\n" | cut -c1-700
  done
done
