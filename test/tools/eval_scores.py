import argparse
import sys
import numpy as np
sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy, top_k_acc
from pyActionRecog.utils.video_funcs import top_k_aggregation_func
from pyActionRecog.utils.video_funcs import sliding_window_aggregation_func
parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

# score_aggregation
agg_score_list = []
# tmp_score = [np.mean(x, axis=0) for x in score_list[0]]
for score_vec in score_list:
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]
    # agg_score_vec = [top_k_aggregation_func(x, 25) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))

# score_weights[1] = 0
final_scores = np.zeros_like(agg_score_list[0])
for i, agg_score in enumerate(agg_score_list):
    final_scores += agg_score * score_weights[i]

vid_list = [x.strip().split()[0] for x in open('data/kinetics/val_list1.txt')]
score_dict = dict(zip(vid_list, agg_score_vec))
import cPickle
cPickle.dump(score_dict, open(args.score_files[0] + '.pc', 'wb'), cPickle.HIGHEST_PROTOCOL)

# acc = mean_class_accuracy(final_scores, label_list[0])
top_5 = 0.0
top_1 = 0.0
for i, label in enumerate(label_list[0]):
    idx = np.argsort(final_scores[i])[-5:]
    if label == idx[-1]:
        top_1  = top_1 + 1
       
    if label in idx:
        top_5 = top_5 + 1
top_5 = top_5 / len(label_list[0])
top_1 = top_1 / len(label_list[0])
print 'Final top1 {:02f}%, top5 {:02f}%, AVG: {:02f}%'.format(top_1 * 100, top_5 * 100, (top_1 + top_5)*50)
