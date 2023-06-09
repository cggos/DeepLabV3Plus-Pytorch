
python main.py --data_root /home/ghc/projects/ml/data/VOC --model deeplabv3plus_mobilenet --enable_vis --vis_port 8097 --gpu_id 0 --year 2007 --crop_val --lr 0.01 --crop_size 300 --batch_size 4 --output_stride 16 --num_classes 22 --total_itrs 1000 --ckpt ../models/best_deeplabv3plus_mobilenet_voc_os16.pth
