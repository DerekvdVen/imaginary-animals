#!/bin/bash

python compare_mean_ap.py -n 250t -c bgo -x 250
python compare_mean_ap.py -n 500t -c bgo -x 500
python compare_mean_ap.py -n 1000t -c bgo -x 1000
python compare_mean_ap.py -n 2000t -c bgo -x 2000

# python calculate_mean_ap.py -n 23456z3_looksgood_full_0.01 -c purple
# python calculate_mean_ap.py -n 23456z3_1250_full_0.01 -c purple
# python calculate_mean_ap.py -n 23456z3_2500_end_good_full_0.01 -c purple
# python calculate_mean_ap.py -n 23456z3_10000n_full_0.01 -c purple
# python calculate_mean_ap.py -n 23456z3_7500n_full_0.01 -c red

# python calculate_mean_ap.py -n real_image_model_z_full_0.01 -c brown

# python calculate_mean_ap.py -n 23456z3_250mix_full_0.1 -c blue


# python calculate_mean_ap.py -n 23456z3_250mix_full_0.01 -c blue
# python calculate_mean_ap.py -n 23456z3_500mix_full_0.01 -c blue
# python calculate_mean_ap.py -n 23456z3_1000mix_full_0.01 -c blue
# python calculate_mean_ap.py -n 23456z3_2000mix_full_0.01 -c blue


# python calculate_mean_ap.py -n 23456z3_250fin_full_0.01 -c green
# python calculate_mean_ap.py -n 23456z3_500fin_full_0.01 -c green
# python calculate_mean_ap.py -n 23456z3_1000fin_full_0.01 -c green
# python calculate_mean_ap.py -n 23456z3_2000fin_full_0.01 -c green

python calculate_mean_ap.py -n 23456z3_250e2_full_0.01 -c brown
python calculate_mean_ap.py -n 23456z3_500e2_full_0.01 -c brown
python calculate_mean_ap.py -n 23456z3_1000e2_full_0.01 -c brown
python calculate_mean_ap.py -n 23456z3_2000e2_full_0.01 -c brown
