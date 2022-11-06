#!/usr/bin/env zsh
source activate
conda activate zihuanqiu
#conda info -e
MY_PYTHON=python
EXEC=main.py

cd /amax/2020/zihuanqiu/PyCIL

#$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S5_32+32.json
#$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S10_32+32.json

#$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S5_18+18.json
#$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S10_18+18.json
#$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S20_18+18.json
$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S5_32+32.json
$MY_PYTHON $EXEC --config=./exps/dual_expert/cifar100/B0S50_32+32.json