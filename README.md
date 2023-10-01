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
