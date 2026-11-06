#!/bin/bash
# Actions에서 종목토론방·뉴스 본문 접근 점검
H1='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
H2='Referer: https://m.stock.naver.com/'
try() { echo "=== $1"; curl -s -L -H "$H1" -H "$H2" -H 'Accept: application/json, text/plain, */*' "$2" -o /tmp/b -w "http %{http_code} bytes %{size_download}\n"; head -c ${3:-600} /tmp/b | tr '\n' ' '; echo; echo; }
C=005930
try "discussion A" "https://m.stock.naver.com/api/stock/$C/discussion?page=1&pageSize=20"
try "discussion B" "https://m.stock.naver.com/api/discussion/stock/$C?page=1&pageSize=20"
try "discussion C" "https://m.stock.naver.com/api/discussion/$C/list?page=1&pageSize=20"
try "discussion D" "https://m.stock.naver.com/api/discussions/$C?page=1&pageSize=20"
try "discussion E" "https://m.stock.naver.com/api/stock/$C/discussion/list?page=1&pageSize=20"
try "discussion F(html app)" "https://m.stock.naver.com/domestic/stock/$C/discussion" 400
try "board html pc" "https://finance.naver.com/item/board.naver?code=$C&page=1" 300
echo "--- board html rows:"; grep -c 'board_read\|item/board_read' /tmp/b; grep -o 'board_read[^"]*' /tmp/b | head -3; grep -o '<span class="tah p10 gray03">[^<]*' /tmp/b | head -3
try "board html pc page 500" "https://finance.naver.com/item/board.naver?code=$C&page=500" 200
echo "--- oldest date on page 500:"; grep -o '<span class="tah p10 gray03">[^<]*' /tmp/b | tail -2
try "news list" "https://m.stock.naver.com/api/news/stock/$C?pageSize=3&page=1" 1200
python3 - <<'PY'
import json
try:
    d=json.load(open('/tmp/b'))
    items=d[0]['items'] if isinstance(d,list) else d.get('items',[])
    it=items[0]; print('keys', list(it.keys())); print({k:it.get(k) for k in ('officeId','articleId','title','body','summary','linkUrl','datetime','imageOriginLink')})
    open('/tmp/ids','w').write(f"{it.get('officeId')} {it.get('articleId')}")
except Exception as e: print('parse fail', e)
PY
read OID AID < /tmp/ids
try "news body mobile api" "https://m.stock.naver.com/api/news/article/$OID/$AID" 1500
try "news body n.news" "https://n.news.naver.com/mnews/article/$OID/$AID" 300
echo "--- article body length:"; grep -o 'id="dic_area"[^>]*>' /tmp/b | head -1; python3 -c "
import re;s=open('/tmp/b',encoding='utf-8',errors='ignore').read();m=re.search(r'id=\"dic_area\"[^>]*>(.*?)</article>',s,re.S);print(len(re.sub('<[^>]+>','',m.group(1))) if m else 'no dic_area')"
try "news body finance.naver" "https://finance.naver.com/news/news_read.naver?article_id=$AID&office_id=$OID" 300
try "datalab(no key)" "https://openapi.naver.com/v1/datalab/search" 200
try "google trends" "https://trends.google.com/trends/api/explore?hl=ko&tz=-540&req=%7B%22comparisonItem%22%3A%5B%7B%22keyword%22%3A%22%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90%22%2C%22geo%22%3A%22KR%22%2C%22time%22%3A%22today%203-m%22%7D%5D%2C%22category%22%3A0%2C%22property%22%3A%22%22%7D" 200
try "dcinside stock gallery" "https://gall.dcinside.com/board/lists/?id=stockus&page=1" 200
