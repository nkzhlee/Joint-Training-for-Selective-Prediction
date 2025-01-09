#!/bin/bash

source activate /home/nlp/zjl5282/env/py3.7_RN

nohup python -u main.py --ckp_name 0_$1 --device cuda:0 > 0_$1.log 2>&1 &

nohup python -u main.py --ckp_name 1_$1 --device cuda:1 > 1_$1.log 2>&1 &

nohup python -u main.py --ckp_name 2_$1 --device cuda:2 > 2_$1.log 2>&1 &

nohup python -u main.py --ckp_name 3_$1 --device cuda:3 > 3_$1.log 2>&1 &


