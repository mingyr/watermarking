nohup python train.py --data_dir=/data/yuming/watermark-data --output_dir=output --gpus=1 &

python test.py --image_seq=123 --checkpoint_dir=output --output_dir=test-output/123 --gpus=1

python psnr.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-psnr/12 --gpus=1
python noop_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-noop/12 --gpus=1

python clip_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-clip/12 --gpus=1
python filt_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-filt-low/12 --gpus=1
python filt_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-filt-high/12 --gpus=1
python noise_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-noise/12 --gpus=1


# python rotated_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-rotated/12 --gpus=1
# python freq_test.py --image_seq=12 --checkpoint_dir=output --output_dir=test-output-freq/12 --gpus=1
