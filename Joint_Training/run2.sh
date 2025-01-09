#!/bin/bash

source activate /home/nlp/zjl5282/env/py3.7_RN



nohup python -u main.py --ckp_name 4_$1 --device cuda:4 > 4_$1.log 2>&1 &

nohup python -u main.py --ckp_name 5_$1 --device cuda:5 > 5_$1.log 2>&1 &

nohup python -u main.py --ckp_name 6_$1 --device cuda:6 > 6_$1.log 2>&1 &

nohup python -u main.py --ckp_name 7_$1 --device cuda:7 > 7_$1.log 2>&1 &
