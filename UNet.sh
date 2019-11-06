
rm -rf cache && mkdir -p cache
rm -rf sample_valid && mkdir -p sample_valid
a='Att_UNet'

CUDA_VISIBLE_DEVICES=0,1 python main.py --model_type=$a --batch_size 16
