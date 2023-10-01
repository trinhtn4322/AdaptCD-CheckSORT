#!/usr/bin/env bash

#python ./mmclassification/tools/train.py ./mmclassification/configs/efficientnet/efficientnet-b0_8xb32-01norm_in1k.py --no-validate --work-dir ./mmclassification/work_dirs/b0

CUDA_VISIBLE_DEVICES=0 python ./mmclassification/tools/test.py ./mmclassification/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py ./mmclassification/work_dirs/b2/latest.pth --out ./test/x/ --metrics mAP  #--work-dir ./mmclassification/work_dirs/b2

#python ./mmclassification/tools/test.py ./mmclassification/configs/resnest/resnest50_32xb64_in1k.py --work-dir ./mmclassification/work_dirs/resnest50

#python ./mmclassification/tools/test.py ./mmclassification/configs/resnest/resnest101_32xb64_in1k.py --work-dir ./mmclassification/work_dirs/resnest101
