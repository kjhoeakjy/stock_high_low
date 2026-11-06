#!/bin/bash
# 토론방(cbox) 호출 인자를 네이버 모바일 페이지의 JS에서 그대로 읽는다(추측 호출 없음). PC 게시판 페이지 조회 여부도 확인.
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
C=005930
mkdir -p /tmp/pr
curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/pr/m
echo "page bytes $(wc -c </tmp/pr/m)"
grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/[^"]+\.js' /tmp/pr/m | sort -u > /tmp/pr/urls
echo "chunks $(wc -l </tmp/pr/urls)"
i=0
while read -r u; do
  i=$((i+1)); curl -s "$u" -o /tmp/pr/c$i.js
  if grep -q 'cbox\|commentBox\|objectId' /tmp/pr/c$i.js; then echo "## $u ($(wc -c </tmp/pr/c$i.js) bytes)"; python3 - /tmp/pr/c$i.js <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
for kw in ('cbox12','pool:','ticket:','ticket=','objectId','templateId','web_naver_list','commentBox','discussion'):
    ms=list(re.finditer(re.escape(kw),s))
    print(f'  [{kw}] {len(ms)} matches')
    for m in ms[:4]:
        print('    ', s[max(0,m.start()-200):m.start()+250].replace('\n',' '))
PY
  fi
done < /tmp/pr/urls
echo "=== discussion chunk strings"
f=$(grep -l 'discussion' /tmp/pr/c*.js | head -3)
for x in $f; do echo "-- $x"; grep -oE '"[^"]{3,80}"' "$x" | grep -iE 'cbox|comment|object|ticket|template|pool|discuss|board|post' | sort -u | head -60; done
echo "=== board.naver PC"
curl -s -L -H "$PC" -H 'Accept-Language: ko-KR,ko;q=0.9' -H 'Referer: https://finance.naver.com/item/main.naver?code=005930' "https://finance.naver.com/item/board.naver?code=$C&page=1" -o /tmp/pr/b -w "http %{http_code} bytes %{size_download} url %{url_effective}\n"
iconv -f euc-kr -t utf-8 /tmp/pr/b 2>/dev/null > /tmp/pr/bu || cp /tmp/pr/b /tmp/pr/bu
echo "rows $(grep -c 'board_read' /tmp/pr/bu)"; grep -o '<title>[^<]*' /tmp/pr/bu | head -1
grep -oE 'board_read\.naver\?code=[0-9]+&amp;nid=[0-9]+[^"]*"[^>]*>[^<]*' /tmp/pr/bu | head -3
grep -oE '<td class="tah p10 gray03">[^<]*' /tmp/pr/bu | head -6 | tr '\n' ' '; echo
