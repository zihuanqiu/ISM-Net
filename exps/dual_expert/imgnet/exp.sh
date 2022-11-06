#!/usr/bin/env zsh
source activate
conda activate zihuanqiu
#conda info -e
MY_PYTHON=python
EXEC=main.py

cd /amax/2020/zihuanqiu/PyCIL

$MY_PYTHON $EXEC --config=./exps/dual_expert/imgnet/B0S10_18+18.json
