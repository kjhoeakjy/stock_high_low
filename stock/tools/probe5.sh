#!/bin/bash
# 토론방 호출 인자: 네이버 페이지가 쓰는 JS에서 그대로 읽기(추측 호출 없음), PC 게시판 페이지 정상 조회 여부
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
C=005930
curl -s -H "$MO" "https://m.stock.naver.com/domestic/stock/$C/discussion" -o /tmp/m
for u in $(grep -oE 'https://ssl.pstatic.net/imgstock/fn/real/_next/static/chunks/[^"]+\.js' /tmp/m | sort -u); do
  curl -s "$u" -o /tmp/c
  if grep -q 'cbox12\|commentBox' /tmp/c; then echo "## $u"; python3 - <<'PY'
import re
s=open('/tmp/c',encoding='utf-8',errors='ignore').read()
for kw in ('cbox12','ticket','objectId','templateId','web_naver_list','commentBox/','pool'):
    for m in list(re.finditer(re.escape(kw),s))[:3]:
        print(f'[{kw}]', s[max(0,m.start()-150):m.start()+200].replace('\n',' '))
PY
  fi
done
echo "=== board.naver GET -L PC"; curl -s -L -H "$PC" -H 'Accept-Language: ko-KR,ko;q=0.9' -H 'Referer: https://finance.naver.com/item/main.naver?code=005930' "https://finance.naver.com/item/board.naver?code=$C&page=1" -o /tmp/b -w "http %{http_code} bytes %{size_download} url %{url_effective}\n"; iconv -f euc-kr -t utf-8 /tmp/b 2>/dev/null > /tmp/bu || cp /tmp/b /tmp/bu; echo "rows $(grep -c 'board_read' /tmp/bu)"; grep -o '<span class="tah p10 gray03">[^<]*' /tmp/bu | head -2; grep -o '<title>[^<]*' /tmp/bu | head -1
