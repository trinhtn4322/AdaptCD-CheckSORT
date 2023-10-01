#!/usr/bin/env bash

#python ./mmclassification/tools/train.py ./mmclassification/configs/efficientnet/efficientnet-b0_8xb32-01norm_in1k.py --no-validate --work-dir ./mmclassification/work_dirs/b0

#python ./mmclassification/tools/train.py ./mmclassification/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py --no-validate --work-dir ./mmclassification/work_dirs/b2

#python ./mmclassification/tools/train.py ./mmclassification/configs/resnest/resnest50_32xb64_in1k.py --no-validate --work-dir ./mmclassification/work_dirs/resnest50

#python ./mmclassification/tools/train.py ./mmclassification/configs/resnest/resnest101_32xb64_in1k.py --no-validate --work-dir ./mmclassification/work_dirs/resnest101

python ./mmclassification/tools/train.py ./mmclassification/configs/repvgg/repvgg-B2_4xb64-coslr-120e_in1k.py --work-dir ./mmclassification/work_dirs/repvgg

python ./mmclassification/tools/train.py ./mmclassification/configs/swin_transformer/swin-small_16xb64_in1k.py --work-dir ./mmclassification/work_dirs/swins
