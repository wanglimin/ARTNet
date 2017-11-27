Model_name=$1
python tools/eval_net.py kinetics 1 c3d_rgb /media/data1/kinetics_val_rgb_img_256_340 models/112_artnet_resnet_18_deploy.prototxt ${Model_name}.caffemodel --caffe_path ../caffe_c3d/ --num_worker 8 --save_scores ${Model_name} --num_frame_per_video 25 --depth 16 --skip 1

