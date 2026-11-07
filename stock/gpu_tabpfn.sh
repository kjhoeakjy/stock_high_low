#!/bin/bash
# vast.ai 주피터 터미널(tmux)에서 TabPFN 2단계 재순위 실험 실행
# 사용: bash gpu_tabpfn.sh [n_ctx=10000] [top=200] [feats=seq|panel]   (토큰 있으면 TABPFN_TOKEN=<토큰> 앞에 붙임)
#   feats=panel: 1단계와 같은 258변수로 TabPFN과 LightGBM을 같은 후보에 학습해 공정 비교(전체 자료 내려받음, 패널 생성 10분 내외)
# 결과: ~/stock_high_low/rerank_<feats>.log 의 "[rerank] {" 줄
set -e
N_CTX=${1:-10000}; TOP=${2:-200}; FEATS=${3:-seq}
export PYTHONUNBUFFERED=1
[ -n "$TABPFN_TOKEN" ] && export TABPFN_NO_BROWSER=1 || echo "토큰 없음 → 첫 실행 시 터미널 안내에 따라 로그인"
cd ~
[ -d stock_high_low ] || git clone -q --depth=1 https://github.com/kjhoeakjy/stock_high_low.git
cd stock_high_low && git pull -q
python3 -m pip install -q pandas pyarrow numpy lightgbm scikit-learn tabpfn
if ! python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python3 -m pip install -q "torch<2.10" --index-url https://download.pytorch.org/whl/cu126
fi
python3 -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
git fetch -q --depth=1 origin stock-data
mkdir -p stock/data
if [ "$FEATS" = panel ]; then
  git --work-tree=stock/data checkout -q FETCH_HEAD -- .
else
  git --work-tree=stock/data checkout -q FETCH_HEAD -- oos_y_oc10.parquet prices.parquet
fi
git reset -q
LOG=rerank_${FEATS}.log; : > $LOG
if [ "$FEATS" = panel ]; then
  python3 -m stock.rerank --top=$TOP --epochs=1 --refit=3 --backend=lgb --feats=panel 2>&1 | tee -a $LOG
fi
python3 -m stock.rerank --top=$TOP --epochs=1 --refit=3 --backend=tabpfn --n_ctx=$N_CTX --feats=$FEATS 2>&1 | tee -a $LOG
echo "== 완료. 결과 요약:"; grep '^\[rerank\] backend\|^\[rerank\] {' $LOG
