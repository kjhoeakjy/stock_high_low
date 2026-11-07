#!/bin/bash
# vast.ai 주피터 터미널(tmux)에서 TabPFN 2단계 재순위 실험 실행
# 사용: TABPFN_TOKEN=<토큰> bash gpu-tabpfn.sh [n_ctx=10000] [top=200]
# 결과: ~/stock_high_low/rerank_tabpfn.log 의 "[rerank] {" 줄 (끝나면 이 파일을 대화에 올리면 됨)
set -e
N_CTX=${1:-10000}; TOP=${2:-200}
[ -n "$TABPFN_TOKEN" ] || { echo "TABPFN_TOKEN 환경변수 필요"; exit 1; }
export TABPFN_NO_BROWSER=1 PYTHONUNBUFFERED=1
cd ~
[ -d stock_high_low ] || git clone -q --depth=1 https://github.com/kjhoeakjy/stock_high_low.git
cd stock_high_low && git pull -q
python3 -m pip install -q pandas pyarrow numpy lightgbm scikit-learn tabpfn
python3 -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
git fetch -q --depth=1 origin stock-data
mkdir -p stock/data
git --work-tree=stock/data checkout -q FETCH_HEAD -- oos_y_oc10.parquet prices.parquet
git reset -q
python3 -m stock.rerank --top=$TOP --epochs=1 --refit=3 --backend=tabpfn --n_ctx=$N_CTX 2>&1 | tee rerank_tabpfn.log
echo "== 완료. 결과 요약:"; grep '^\[rerank\] {' rerank_tabpfn.log
