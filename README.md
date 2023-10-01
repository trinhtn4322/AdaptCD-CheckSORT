# AdaptCD-CheckSORT

## 1. Environments settings
python 3.7.12
pytorch 1.10.0
torchvision 0.11.2
cuda 10.2
mmcv-full 1.4.3
tensorflow-gpu 1.15.0

1. git clone https://github.com/trinhtn4322/AdaptCD-CheckSORT.git
2. conda create -n checkout python=3.7
3. conda activate checkout
4. pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
5. pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
6. cd ./AdaptCD-CheckSORT
7. pip install -r requirements.txt
8. sh ./tools1/setup.sh 

Download ffmp.exe và thay ffmpeg_path trong tools1/extrasct_frames.py

# 2. Download require file
https://drive.google.com/drive/folders/104_y8Z-1Ha78Fj1G0s3KgUuTs6t1Ta_L?usp=drive_link\

# Run model
## 1. Use the FFmpeg library to extract/count frames.
python tools1/extract_frames.py --out_folder ./frames
## AdaptCD
python tools1/test_network.py --input_folder ./frames --out_file ./results.txt

## CheckSORT

python tools/test_net_23_inframe.py --input_folder ./frames --out_file ./results.txt --detector ./checkpoints/detectors_cascade_rcnn_r50_1x_coco/epoch_5.pth --feature ./checkpoints/b0/epoch_20.pth --b2 ./checkpoints/b2/epoch_20.pth --resnest50 ./checkpoints/resnest50/epoch_20.pth --resnest101 ./checkpoints/resnest101/epoch_20.pth
