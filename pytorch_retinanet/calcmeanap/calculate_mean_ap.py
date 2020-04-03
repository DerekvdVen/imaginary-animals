"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

Will display a plot of precision vs recall curves at 10 distinct IoU thresholds as well as output
summary information regarding the average precision and mAP scores.

NOTE: Requires the files `ground_truth_boxes.json` and `predicted_boxes.json` which can be
downloaded fromt this gist.
"""



from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

from functions import calc_iou_individual, calc_precision_recall, get_single_image_results, get_model_scores_map, get_avg_precision_at_iou, plot_pr_curve

sns.set_style('white')
sns.set_context('poster')
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('-n', default="test1", type=str, help='run name')
args = parser.parse_args()
print(args)

infile_gt = args.n + '_gt.json' # change this to choose rendered or real
infile_pred = args.n + '_pred.json'

print("infile_gt: ", infile_gt)
print("infile_pred: ", infile_pred)

#infile_gt = "ground_truth_boxes_animals_test.json"
#infile_pred = "predicted_boxes_animals_test.json"

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

COLORS = ["#e41a1c",
"#377eb8",
"#4daf4a",
"#984ea3",
"#ff7f00",
"#f781bf",
"#a65628",
"#ffff33",
"#999999"
]

if __name__ == "__main__":

    with open(infile_gt) as infile: #ground_truth_boxes_animals.json
        gt_boxes = json.load(infile)

    with open(infile_pred) as infile: #predicted_boxes_animals.json
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    iou_thr = 0.5
    start_time = time.time()
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    for idx, iou_thr in enumerate(np.linspace(0.1, 0.9, 9)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx], ax=ax,title=args.n)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.savefig("output/"+ args.n + ".png")
    plt.show()

