#!/bin/bash
# 토론방 목록 API: 응답 필드, 페이지 크기 한도, 과거 이동(offset=orderNo) 범위, 여러 종목 한 번에 조회
MO='User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
B=https://m.stock.naver.com/front-api
H=(-H "$MO" -H "Referer: https://m.stock.naver.com/domestic/stock/005930/discussion")
get(){ curl -s "${H[@]}" "$1" -w "\n[http %{http_code}]\n"; }
echo "=== list 기본"; get "$B/discussion/list?itemCode=005930&discussionType=domesticStock" > /tmp/l1; tail -1 /tmp/l1; python3 - <<'PY'
import json
s=open('/tmp/l1').read().rsplit('\n[http',1)[0]
d=json.loads(s); r=d['result']; print('result keys',list(r.keys())); ps=r.get('posts',[]); print('posts',len(ps))
if ps:
    p=ps[0]; print('post keys',list(p.keys()))
    print({k:(str(v)[:80]) for k,v in p.items() if k not in('writer',)})
    print('first',ps[0].get('writtenAt'),'last',ps[-1].get('writtenAt'),'orderNo',ps[-1].get('orderNo'))
PY
for ps in 50 100 200; do echo "=== pageSize=$ps"; get "$B/discussion/list?itemCode=005930&discussionType=domesticStock&pageSize=$ps" | python3 -c "
import sys,json;t=sys.stdin.read();s,code=t.rsplit('\n[http',1)
try:
    d=json.loads(s);ps=d['result'].get('posts',[]);print('posts',len(ps),'first',ps[0]['writtenAt'] if ps else '-','last',ps[-1]['writtenAt'] if ps else '-')
except Exception as e: print('err',code.strip(),s[:200])"; done
echo "=== filter excludeNews / 소형주 예"; get "$B/discussion/list?itemCode=005930&discussionType=domesticStock&filter=excludeNews" | python3 -c "
import sys,json;t=sys.stdin.read();s,code=t.rsplit('\n[http',1)
try:
    d=json.loads(s);ps=d['result'].get('posts',[]);print('posts',len(ps),[p.get('postType') for p in ps][:10])
except Exception as e: print('err',code.strip(),s[:300])"
echo "=== 과거 이동: 한 종목(005930) offset 반복 20회"
python3 - <<'PY'
import json,subprocess,time
B="https://m.stock.naver.com/front-api"
H=["-H","User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1","-H","Referer: https://m.stock.naver.com/domestic/stock/005930/discussion"]
off=None; n=0
for i in range(20):
    u=f"{B}/discussion/list?itemCode=005930&discussionType=domesticStock&pageSize=100"+(f"&offset={off}" if off else "")
    t0=time.time(); out=subprocess.run(["curl","-s",*H,u],capture_output=True,text=True).stdout
    try: d=json.loads(out); ps=d["result"]["posts"]
    except Exception as e: print("err",out[:200]); break
    if not ps: print("empty"); break
    n+=len(ps); off=ps[-1].get("orderNo"); print(i,len(ps),ps[-1].get("writtenAt"),"orderNo",off,f"{time.time()-t0:.2f}s")
print("total",n)
PY
echo "=== 소형주 한 종목 하루 글 수 예(226950 젬백스)"; get "$B/discussion/list?itemCode=226950&discussionType=domesticStock&pageSize=100" | python3 -c "
import sys,json;t=sys.stdin.read();s,code=t.rsplit('\n[http',1)
try:
    d=json.loads(s);ps=d['result'].get('posts',[]);print('posts',len(ps),'first',ps[0]['writtenAt'] if ps else '-','last',ps[-1]['writtenAt'] if ps else '-')
except Exception as e: print('err',code.strip(),s[:300])"
echo "=== 여러 종목 한 번에"; for q in "itemCodes=005930,000660&discussionType=domesticStock" "itemCodes=005930&itemCodes=000660" "itemCode=005930&itemCode=000660"; do echo "-- $q"; get "$B/discussion/list/itemCodes?$q" | cut -c1-400; done
echo "=== 랭킹 전체"; get "$B/discussion/rankings/itemCodes" | cut -c1-1500
echo "=== 댓글 수"; get "$B/discussion/comment/counts?postIds=431736297" | cut -c1-300
