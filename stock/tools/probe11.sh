#!/bin/bash
# 토론방 writtenAt 시간대 확인: 현재 UTC/KST와 최신 글 시각 비교
echo "now UTC $(date -u +%FT%T)  KST $(TZ=Asia/Seoul date +%FT%T)"
for code in 005930 000660 042700; do
  curl -s -A "Mozilla/5.0" -H "Referer: https://m.stock.naver.com/" \
    "https://m.stock.naver.com/front-api/discussion/list?itemCode=$code&discussionType=domesticStock&pageSize=5" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); [print('$code', p.get('writtenAt'), p.get('modifiedAt'), (p.get('title') or '')[:30]) for p in d['result']['posts']]"
done
