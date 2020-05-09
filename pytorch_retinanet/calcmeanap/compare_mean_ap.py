

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

from functions import calc_iou_individual, calc_precision_recall, get_single_image_results, get_model_scores_map, get_avg_precision_at_iou, plot_pr_curve, calctps

sns.set_style('white')
sns.set_context('poster')
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('-n', default="test1", type=str, help='run name')
parser.add_argument('-iou', default=0.2,type=float,help="iou")
parser.add_argument('-c', default = "generic",type=str,help='colors for plot')
parser.add_argument('-x', default = "", type=str,help='number of images')

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

COLORS = [#red
    "#fcbba1",
    "#fb6a4a",
    "#cb181d",
    "#67000d"
    ]

if args.c == 'purple':
    COLORS = ["#49006a",
    
    
    "#ae017e",
    
    
    "#f768a1",
    
    
    "#fcc5c0",
    
    '#fde0dd',
    
    "#fff7f3"
    ]

if args.c == "red":
    COLORS = [#red
    "#67000d",
    "#a50f15",
    "#cb181d",
    "#ef3b2c",
    "#fb6a4a",
    "#fc9272",
    "#fcbba1",
    "#fee0d2",
    "#fff5f0"
    ]

if args.c == "bgb":
    COLORS = ["#dd3497","#3690c0","#74c476","#ec7014"]

if args.c == "bgo":
    COLORS = ["#0570b0","#78c679","#fc8d59"]    

if args.c == "green":
    COLORS = [#green
    "#41ab5d",
    "#238b45",
    "#006d2c",
    "#00441b",
    "#74c476",
    "#a1d99b",
    "#c7e9c0",
    "#e5f5e0",
    "#f7fcf5"
    ]
if args.c == "blue":
    COLORS = [#blue
    "#023858",
    "#045a8d",
    "#0570b0",
    "#3690c0",
    "#74a9cf",
    "#a6bddb",
    "#d0d1e6",
    "#ece7f2",
    "#fff7fb",
    ]
if args.c == "lb":
    COLORS = [#lb
    "#084081",
    "#0868ac",
    "#2b8cbe",
    "#4eb3d3",
    "#7bccc4",
    "#a8ddb5",
    "#ccebc5",
    "#e0f3db",
    "#f7fcf0"
    ]
if args.c == "brown":
    COLORS = [#brown
    "#662506",
    "#993404",
    "#cc4c02",
    "#ec7014",
    "#fe9929",
    "#fec44f",
    "#fee391",
    "#fff7bc",
    "#ffffe5"
    ]


THICNESS = [1.5,2,2.5]
MARKERS =['X','o','v','s']

## Derek ##
if __name__ == "__main__":

    #comparing new annotatioin scheme
    modelrun_list = ["20304060m_z_full_0.1","23456z_full_0.1","23456z_end_full_0.1","23456z2_full_0.1","23456z3_full_0.1","real_image_model_z_full_0.1"]
    modelrun_list = ["20304060m_z_full_0.1","23456z3_looksgood_full_0.1","real_image_model_z_full_0.1"]
    
    #rendered
    #modelrun_list = ["23456z3_1250_full_0.1","23456z3_2500_end_good_full_0.1","23456z3_looksgood_full_0.1","23456z3_10000n_full_0.1"]
    
    modelrun_list = ["23456z3_1250_full_0.01","23456z3_2500_end_good_full_0.01","23456z3_looksgood_full_0.01","23456z3_10000n_full_0.01"]
    #title_list = ["250","500","1000","2000"]
    #title_list = ["1250","2500","5000","10000"]
    #modelrun_list = ["23456z3_looksgood_full_0.1","20304060m_2000e_full_0.1"]
    title_list = ["Mixed", "Fine-tuned","Exclusive real"]

    #new finetuning
    if args.x == "250":
        modelrun_list = ["23456z3_250mix_full_0.01","23456z3_250fin_full_0.01","23456z3_250e2_full_0.01"]
    if args.x == "500":
        modelrun_list = ["23456z3_500mix_full_0.01","23456z3_500fin_full_0.01","23456z3_500e2_full_0.01"]
    if args.x == "1000":
        modelrun_list = ["23456z3_1000mix_full_0.01","23456z3_1000fin_full_0.01","23456z3_1000e2_full_0.01"]
    if args.x == "2000":
        modelrun_list = ["23456z3_2000mix_full_0.01","23456z3_2000fin_full_0.01","23456z3_2000e2_full_0.01"]
    
    #title_list = ["Exclusive rendered 5000","Mixed 1000","Fine tuned 1000", "Exclusive real 8000"]

    #comparing mixed and real
    #modelrun_list = ["20304060m_250e_full_0.1","23456z3_250mix_full_0.1","23456z3_250fin_full_0.1",]
    #modelrun_list = ["20304060m_500e_full_0.1","23456z3_500mix_full_0.1","23456z3_500fin_full_0.1"]
    #modelrun_list = ["20304060m_1000e_full_0.1","23456z3_1000mix_full_0.1","23456z3_1000fin_full_0.1"]
    #modelrun_list = ["20304060m_2000e_full_0.1","23456z3_2000mix_full_0.1","23456z3_2000fin_full_0.1"]
    
    #modelrun_list = ["23456z3_1000mix_full_0.1","20304060m_1000e_full_0.1"]
    #modelrun_list = ["23456z3_2000mix_full_0.1","20304060m_2000e_full_0.1"]
    #modelrun_list = ["23456z3__4000mix_full_0.1","20304060m_4000e_full_0.1"] 
    #modelrun_list = ["23456z3_250mix_full_0.1","23456z3_500mix_full_0.1","23456z3_1000mix_full_0.1","23456z3_2000mix_full_0.1"]#,"20304060m_4000z_full_0.1"]
    #modelrun_list = ["23456z3_looksgood_full_0.1","20304060m_250e_full_0.1","20304060m_500e_full_0.1","20304060m_1000e_full_0.1","20304060m_2000e_full_0.1"]#,"20304060m_4000e_full_0.1"]
    
    # mixed training
    
    #overall
    #modelrun_list = ["23456z3_looksgood_full_0.01","23456z3_1000mix_full_0.01","23456z3_1000fin_full_0.01","real_image_model_z_full_0.01"]

    gt_boxes_list = []
    pred_boxes_list = []

    for model in modelrun_list:
        with open(model + '_gt.json') as infile: #ground_truth_boxes_animals.json
            gt_boxes_list.append(json.load(infile))

        with open(model + '_pred.json') as infile: #predicted_boxes_animals.json
            pred_boxes_list.append(json.load(infile))

    # Runs it for each model
    iou_thr = args.iou
    # for idx, modelrun in enumerate(modelrun_list):
    #     start_time = time.time()
    #     data = get_avg_precision_at_iou(gt_boxes_list[idx], pred_boxes_list[idx], iou_thr=iou_thr)
    #     end_time = time.time()
    #     print(modelrun)
    #     print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    #     print('avg precision: {:.4f}'.format(data['avg_prec']))

    avg_precs = []
    iou_thrs = []
    
    # Plot
    
    ax = None
    
    for idx, modelrun in enumerate(modelrun_list):
        print(modelrun)
        #calctps(gt_boxes_list[idx], pred_boxes_list[idx], iou_thr=iou_thr)
        data = get_avg_precision_at_iou(gt_boxes_list[idx], pred_boxes_list[idx], iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

        precisions = data['precisions']
        recalls = data['recalls']
        ax = plot_pr_curve(
            precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx], ax=ax,title = args.x + " images" ,thic=2,marker=MARKERS[idx])
    
    for xval in np.linspace(0.0, 1.2, 13):
        plt.vlines(xval, 0.0, 1.2, color='gray', alpha=0.3, linestyles='dashed')
    plt.legend(labels = title_list,loc='upper right', title='n images', frameon=True,prop={"size":18})
    plt.savefig("output/"+ args.n + "_" + str(iou_thr) +"_comparison.png")
    plt.show()