

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
parser.add_argument('-iou', default=0.2,type=float,help="iou")
args = parser.parse_args()
print(args)

# infile_gt = args.n + '_gt_rendered.json' # change this to choose rendered or real
# infile_pred = args.n + '_pred_rendered.json'

# print("infile_gt: ", infile_gt)
# print("infile_pred: ", infile_pred)

#infile_gt = "ground_truth_boxes_animals_test.json"
#infile_pred = "predicted_boxes_animals_test.json"

COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
COLORS = [
"#8c510a","#bf812d",
"#01665e","#35978f",
"#c51b7d","#de77ae",
"#4d9221","#7fbc41"
]

## Derek ##
if __name__ == "__main__":

    #comparing new annotatioin scheme
    #modelrun_list = ["20304060m_z_full_0.1","23456z_full_0.1","real_image_model_z_full_0.1"]
    
    #new finetuning
    #modelrun_list = ["20304060m_250f_full_0.1","20304060m_500f_full_0.1","20304060m_1000f_full_0.1"]#,"20304060m_2000z_full_0.1","20304060m_4000z_full_0.1"]
    #modelrun_list = ["20304060m_250z_full_0.1","20304060m_250f_full_0.1"]
    #modelrun_list = ["20304060m_500z_full_0.1","20304060m_500f_full_0.1"]
    #modelrun_list = ["20304060m_1000z_full_0.1","20304060m_1000f_full_0.1"]
    

    #comparing mixed and real
    #modelrun_list = ["20304060m_250z_full_0.1","20304060m_250e_full_0.1"]
    #modelrun_list = ["20304060m_500z_full_0.1","20304060m_500e_full_0.1"]
    #modelrun_list = ["20304060m_1000z_full_0.1","20304060m_1000e_full_0.1"]
    #modelrun_list = ["20304060m_2000z_full_0.1","20304060m_2000e_full_0.1"]
    #modelrun_list = ["20304060m_4000z_full_0.1","20304060m_4000e_full_0.1"] 
    #modelrun_list = ["20304060m_250z_full_0.1","20304060m_500z_full_0.1","20304060m_1000z_full_0.1","20304060m_2000z_full_0.1"]#,"20304060m_4000z_full_0.1"]
    modelrun_list = ["20304060m_250e_full_0.1","20304060m_500e_full_0.1","20304060m_1000e_full_0.1","20304060m_2000e_full_0.1"]#,"20304060m_4000e_full_0.1"]
    modelrun_list = ["20304060m_250e_full_0.1","20304060m_250z_full_0.1",
                        "20304060m_500e_full_0.1","20304060m_500z_full_0.1",
                            "20304060m_1000e_full_0.1","20304060m_1000z_full_0.1",
                                "20304060m_2000e_full_0.1","20304060m_2000z_full_0.1"]#,"20304060m_4000e_full_0.1"]
    
    


    # mixed training
    #modelrun_list = ["20304060m_z_full_0.1","20304060m_250z_full_0.1","20304060m_500z_full_0.1","20304060m_1000z_full_0.1","20304060m_2000z_full_0.1","20304060m_4000z_full_0.1","real_image_model_z_full_0.1"]
    
    #overall
    #modelrun_list = ["20304060m_z_full_0.1","real_image_model_z_full_0.1"]
    
    gt_boxes_list = []
    pred_boxes_list = []

    for model in modelrun_list:
        with open(model + '_gt.json') as infile: #ground_truth_boxes_animals.json
            gt_boxes_list.append(json.load(infile))

        with open(model + '_pred.json') as infile: #predicted_boxes_animals.json
            pred_boxes_list.append(json.load(infile))

    # Runs it for each model
    iou_thr = args.iou
    for idx, modelrun in enumerate(modelrun_list):
        start_time = time.time()
        data = get_avg_precision_at_iou(gt_boxes_list[idx], pred_boxes_list[idx], iou_thr=iou_thr)
        end_time = time.time()
        print(modelrun)
        print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
        print('avg precision: {:.4f}'.format(data['avg_prec']))

    avg_precs = []
    iou_thrs = []
    
    # Plot
    
    ax = None
    
    for idx, modelrun in enumerate(modelrun_list):
        data = get_avg_precision_at_iou(gt_boxes_list[idx], pred_boxes_list[idx], iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']

        if idx%2 == 0: 
            ax = plot_pr_curve(
                precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx], ax=ax,title= "iou = " + str(iou_thr))
        else:
            ax = plot_pr_curve(
                precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[(idx)], ax=ax,title= "iou = " + str(iou_thr),line_style=":",alpha = 0.95,thic=1.5)
    for xval in np.linspace(0.0, 1.2, 13):
        plt.vlines(xval, 0.0, 1.2, color='gray', alpha=0.3, linestyles='dashed')
    plt.legend(labels = modelrun_list,loc='upper right', title='Model run', frameon=True,prop={"size":10})
    plt.savefig("output/"+ args.n + "_" + str(iou_thr) +"_comparison.png")
    plt.show()