#!/bin/bash
# vast.ai 주피터 터미널(tmux)에서 TabPFN 2단계 재순위 실험 실행
# 사용: bash gpu_tabpfn.sh [n_ctx=10000] [top=200]   (토큰 있으면 TABPFN_TOKEN=<토큰> 앞에 붙임)
# 결과: ~/stock_high_low/rerank_tabpfn.log 의 "[rerank] {" 줄 (끝나면 이 파일을 대화에 올리면 됨)
set -e
N_CTX=${1:-10000}; TOP=${2:-200}
export PYTHONUNBUFFERED=1
# 토큰이 없으면 TabPFN이 터미널에 로그인 주소를 출력 → 브라우저에서 로그인 후 받은 토큰을 터미널에 붙여넣으면 저장됨
[ -n "$TABPFN_TOKEN" ] && export TABPFN_NO_BROWSER=1 || echo "토큰 없음 → 첫 실행 시 터미널 안내에 따라 로그인"
cd ~
[ -d stock_high_low ] || git clone -q --depth=1 https://github.com/kjhoeakjy/stock_high_low.git
cd stock_high_low && git pull -q
python3 -m pip install -q pandas pyarrow numpy lightgbm scikit-learn tabpfn
# 드라이버(CUDA 12.6)에 맞는 torch로 교체(최신 cu13x 휠은 드라이버가 못 받음)
if ! python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python3 -m pip install -q "torch<2.10" --index-url https://download.pytorch.org/whl/cu126
fi
python3 -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
git fetch -q --depth=1 origin stock-data
mkdir -p stock/data
git --work-tree=stock/data checkout -q FETCH_HEAD -- oos_y_oc10.parquet prices.parquet
git reset -q
python3 -m stock.rerank --top=$TOP --epochs=1 --refit=3 --backend=tabpfn --n_ctx=$N_CTX 2>&1 | tee rerank_tabpfn.log
echo "== 완료. 결과 요약:"; grep '^\[rerank\] {' rerank_tabpfn.log
