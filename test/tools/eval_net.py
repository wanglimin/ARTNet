import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from pyActionRecog import parse_directory_rgb
from pyActionRecog import parse_directory_rgb1
from pyActionRecog import parse_split_file
from pyActionRecog import parse_directory_flow

from pyActionRecog.utils.video_funcs import default_aggregation_func

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
parser.add_argument('modality', type=str, choices=['rgb', 'flow','c3d_rgb', 'c3d_flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='image_')
parser.add_argument('--c3d_prefix', type=str, help="prefix of RGB frames", default='image_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='y_')
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')
parser.add_argument("--depth", type=int, default=32)
parser.add_argument('--skip', type=int, default=1)
parser.add_argument('--attention_name', type=str, default=None)
args = parser.parse_args()

print args

sys.path.append(os.path.join(args.caffe_path, 'python'))
from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
print args.dataset
split_tp = parse_split_file(args.dataset)

if args.modality == 'rgb' or args.modality =='c3d_rgb':
    if args.dataset == 'hmdb51' or args.dataset == 'ucf101':
        f_info = parse_directory_rgb(args.frame_path, args.rgb_prefix)
    else:
        f_info = parse_directory_rgb1(args.frame_path, args.rgb_prefix)
else:
    f_info = parse_directory_flow(args.frame_path, args.flow_x_prefix, args.flow_y_prefix)

print(len(f_info[0]))
gpu_list = args.gpus

if args.dataset == 'hmdb51' or args.dataset == 'ucf101':
    eval_video_list = split_tp[args.split - 1][1]
else:
    eval_video_list = split_tp

score_name = 'fc-action'
attention_name = args.attention_name

def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])


def eval_video(video):
    global net
    label = video[1]
    vid = video[0]

    video_frame_path = f_info[0][vid]
    if args.modality == 'rgb':
        cnt_indexer = 1
    elif args.modality == 'flow':
        cnt_indexer = 1
    elif args.modality == 'c3d_rgb':
        cnt_indexer = 1
    elif args.modality == 'c3d_flow':
        cnt_indexer = 1
    else:
        raise ValueError(args.modality)
    frame_cnt = f_info[cnt_indexer][vid]

    stack_depth = 0
    if args.modality == 'rgb':
        stack_depth = 1
    elif args.modality == 'flow':
        stack_depth = 5
    elif args.modality == 'c3d_rgb':
        stack_depth = args.depth
    elif args.modality == 'c3d_flow':
        stack_depth  = args.depth
    else:
        raise ValueError('Invalid modality: ' + args.modality)

    step = 1.0 * (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    # step = min(6.0, frame_cnt/10.0)
    if step > 0:
        frame_ticks = np.arange(1, 2 +frame_cnt-stack_depth, step)
    else:
        frame_ticks = [1] * args.num_frame_per_video

    frame_ticks = np.floor(frame_ticks)
    frame_ticks = frame_ticks.astype(int)
#    assert(len(frame_ticks) == args.num_frame_per_video)

    frame_scores = []
    for tick in frame_ticks:
        if args.modality == 'rgb':
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256), attention_name=attention_name)
            frame_scores.append(scores)
        if args.modality == 'flow':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:06d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:06d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
            scores = net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256), attention_name=attention_name)
            frame_scores.append(scores)
        if args.modality == 'c3d_flow':
           frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
           flow_stack = []
           for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
           scores = net.predict_single_c3d_flow_stack(flow_stack, score_name, frame_size=(170, 128))
           frame_scores.append(scores)
        if args.modality == 'c3d_rgb':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]
            c3d_stack = []
	    i = 0;
            skip = args.skip
            for idx in frame_idx:
		i = i+1
		if (i%skip == 0):
			x_name = '{}{:06d}.jpg'.format(args.c3d_prefix, idx)
			img = cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_COLOR)
			c3d_stack.append(img)
            scores = net.predict_single_c3d_rgb_stack(c3d_stack, score_name, frame_size=(170,128))
            frame_scores.append(scores)


    ii = 0
    if attention_name is not None:
        frame_attentions = [x[1] for x in frame_scores]
        frame_scores = [x[0] for x in frame_scores]
        
    if np.argmax(default_aggregation_func(frame_scores)) == label:
       ii = 1
    print 'video {0} pred {1} label {2} same {3} done'.format(vid, np.argmax(default_aggregation_func(frame_scores)), label, ii )
    sys.stdin.flush()
    if attention_name is None:
        return np.array(frame_scores), label
    else:
        return np.array(frame_scores), label, np.array(frame_attentions)

if args.num_worker > 1:
    pool = multiprocessing.Pool(args.num_worker, initializer=build_net)
    video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
    video_scores = map(eval_video, eval_video_list)

video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
video_labels = [x[1] for x in video_scores]
cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit/cls_cnt
print cls_acc

print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

if args.save_scores is not None:
     np.savez(args.save_scores, scores=video_scores, labels=video_labels)
