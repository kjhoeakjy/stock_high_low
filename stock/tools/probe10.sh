#!/bin/bash
# 분봉 자료 깊이 확인: 네이버 fchart 분봉(몇 일치까지 오는지), 네이버 모바일 분봉 API
PC='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
C=005930
echo "=== fchart minute count=20000"
curl -s -H "$PC" "https://fchart.stock.naver.com/sise.nhn?symbol=$C&timeframe=minute&count=20000&requestType=0" -o /tmp/fm; echo "bytes $(wc -c </tmp/fm)"; grep -o '<item data="[^"]*"' /tmp/fm | head -2; grep -o '<item data="[^"]*"' /tmp/fm | tail -1; echo "items $(grep -c '<item ' /tmp/fm)"
echo "=== m.stock 분봉 API"
for u in "https://api.stock.naver.com/chart/domestic/item/$C/minute?periodType=minute&count=2000" "https://m.stock.naver.com/api/chart/domestic/item/$C/minute?count=2000" "https://api.stock.naver.com/chart/domestic/item/$C/minute?startDateTime=202510010900&endDateTime=202511061530"; do echo "-- $u"; curl -s -H "$PC" -H "Referer: https://m.stock.naver.com/" "$u" -w " [http %{http_code}]\n" | cut -c1-500; done
echo "=== 일별 시가 09:05 근사: 5분봉 count=5000"
curl -s -H "$PC" "https://fchart.stock.naver.com/sise.nhn?symbol=$C&timeframe=minute5&count=5000&requestType=0" -o /tmp/f5; echo "bytes $(wc -c </tmp/f5)"; grep -o '<item data="[^"]*"' /tmp/f5 | head -1; grep -o '<item data="[^"]*"' /tmp/f5 | tail -1; echo "items $(grep -c '<item ' /tmp/f5)"
