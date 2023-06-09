
#python predict_cam.py --dataset voc --model deeplabv3plus_mobilenet --ckpt ../models/best_deeplabv3plus_mobilenet_voc_os16.pth

 python predict_cam.py --dataset voc --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth

#python predict.py --input ../data/VOC/VOCdevkit/VOC2007/JPEGImages/ --dataset voc --model deeplabv3plus_mobilenet --ckpt ../models/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results_to test_results

#python predict.py --input ../data/VOC/VOCdevkit/VOC2007/JPEGImages/000175.jpg --dataset voc --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results_to test_results1 --crop_size 513

#python predict.py --input ../data/VOC/VOCdevkit/VOC2007/JPEGImages --dataset voc --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results_to test_results2
