#!/bin/bash
# 종목토론방 접근 2차 점검: PC UA, 모바일 앱 HTML 내 API 경로 탐색
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
C=005930
echo "=== board.naver PC UA page1"; curl -s -H "$PC" -H 'Referer: https://finance.naver.com/' "https://finance.naver.com/item/board.naver?code=$C&page=1" -o /tmp/b -w "http %{http_code} bytes %{size_download}\n"
iconv -f euc-kr -t utf-8 /tmp/b 2>/dev/null > /tmp/bu || cp /tmp/b /tmp/bu
echo "rows: $(grep -c 'board_read' /tmp/bu)"; grep -o '<span class="tah p10 gray03">[^<]*' /tmp/bu | head -3; grep -o 'title="[^"]*"' /tmp/bu | head -5; grep -o '<td class="num">[^<]*' /tmp/bu | head -6 | tr '\n' ' '; echo
echo "=== board.naver PC UA page300"; curl -s -H "$PC" -H 'Referer: https://finance.naver.com/' "https://finance.naver.com/item/board.naver?code=$C&page=300" -o /tmp/b -w "http %{http_code}\n"; iconv -f euc-kr -t utf-8 /tmp/b 2>/dev/null > /tmp/bu || cp /tmp/b /tmp/bu; grep -o '<span class="tah p10 gray03">[^<]*' /tmp/bu | tail -2
echo "=== mobile discussion HTML: api paths"; curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/m -w "http %{http_code}\n"
grep -oE '"[^"]*(discussion|board|community|comment)[^"]*"' /tmp/m | sort -u | head -30
grep -oE 'https?://[a-z0-9.-]*naver\.com[^" ]*' /tmp/m | grep -iE 'discuss|board|comm' | sort -u | head
echo "=== __NEXT_DATA__ keys"; python3 - <<'PY'
import re,json
s=open('/tmp/m',encoding='utf-8',errors='ignore').read()
m=re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',s,re.S)
if m:
    d=json.loads(m.group(1)); print(list(d.keys())); pp=d.get('props',{}).get('pageProps',{}); print(list(pp.keys())[:30]); print(json.dumps(pp,ensure_ascii=False)[:1500])
else:
    print('no next data'); print(re.findall(r'/api/[a-zA-Z0-9/_\-]+',s)[:40])
PY
for u in "https://m.stock.naver.com/api/stock/$C/discussion/posts?page=1&pageSize=20" "https://m.stock.naver.com/api/community/stock/$C?page=1&pageSize=20" "https://m.stock.naver.com/front-api/discussion/stock/$C?page=1&pageSize=20" "https://api.stock.naver.com/discussion/stock/$C?page=1&pageSize=20" "https://m.stock.naver.com/api/discussion/domestic/stock/$C?page=1&pageSize=20"; do echo "=== $u"; curl -s -H "$MO" -H 'Referer: https://m.stock.naver.com/' "$u" -o /tmp/b -w "http %{http_code} bytes %{size_download}\n"; head -c 300 /tmp/b; echo; done
echo "=== dcinside 주식갤러리"; for g in stock_new2 neostock kospi; do curl -s -H "$PC" "https://gall.dcinside.com/board/lists/?id=$g&page=1" -o /tmp/b -w "$g http %{http_code} bytes %{size_download} "; grep -o '<title>[^<]*' /tmp/b | head -1; done
