import matplotlib.pyplot as plt
import json
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit
import statistics

colors = ['lightsteelblue','blue', 'purple', 'red']
title = ['FPR=0.1%', 'FPR=1%', 'z avg']
diversity_label = ['Self-BLEU', 'Entropy']

def fivepl(x, a, b, c, d, g):
    return ( ( (a-d) / ( (1+( (x/c)** b )) **g) ) + d )

def expfunc(x, a, b, c):
    return -a*np.exp(b*x)+1

def plot_kgw_gamma_1(ax, kgw_delta, temp=1.0, dataset="HealthSearchQA", length=100, split='valid', \
                     plot_curve=False, curve_shape='straight', model_name="mistral", key_id=0, gen_seed=42, \
                        cross=False, semantic='STS', diversity=False):
    d_0, d_1 = [], []
    s, diversity_list, ent_list = [], [], []
    z_avg_list = []
    gamma = 0.1

    fpr = json.load(open(f"FPR/KGW/len_{length}/0.1/result.json"))
    z_score_0 = fpr["emp_thres_0.1%"]
    z_score_1 = fpr["emp_thres_1%"]
    # for gamma, delta, z_score_0, z_score_1 in zip(kgw_gamma, kgw_delta, kgw_z_score_0, kgw_z_score_1):
    for delta in kgw_delta:
        if key_id == -1:
            data = []
            for k in range(3):
                file_name = f"{dataset}/gen_seed_{gen_seed}/key{k}/KGW/len_{length}_temp_{temp}/{split}/text/{model_name}_{gamma}_{delta}.json_pp"
                with open(file_name, "r") as f:
                    data.extend([json.loads(x) for x in f.read().strip().split("\n")])
        else:
            file_name = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/KGW/len_{length}_temp_{temp}/{split}/text/{model_name}_{gamma}_{delta}.json_pp"
            with open(file_name, "r") as f:
                data = [json.loads(x) for x in f.read().strip().split("\n")]

        if cross:
            cross_flag = '_cross'
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/KGW/len_{length}_temp_{temp}/{split}/{model_name}_{gamma}_{delta}{cross_flag}.json"
            STS = json.load(open(result_file))[f'STS{cross_flag}']
        else:
            STS = statistics.mean([val['STS'] for val in data])

        if semantic == 'ppl':
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/KGW/len_{length}_temp_{temp}/{split}/{model_name}_{gamma}_{delta}.json"
            STS = json.load(open(result_file))['avg_loss']

        if diversity:
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/KGW/len_{length}_temp_{temp}/{split}/{model_name}_{gamma}_{delta}.json"
            ent = json.load(open(result_file))['avg_entropy']
            diversity = json.load(open(result_file))['avg_diversity']
            diversity_list.append(diversity)
            ent_list.append(ent)

        z_score_list = [val['z_wm'] for val in data]
        thres_0 = sum([1 for val in z_score_list if val > z_score_0]) / len(z_score_list)
        thres_1 = sum([1 for val in z_score_list if val > z_score_1]) / len(z_score_list)
        z_avg = statistics.mean(z_score_list)
        d_0.append(thres_0)
        d_1.append(thres_1)
        s.append(STS)
        z_avg_list.append(z_avg)

    x_list = [s, s, s]
    y_list = [d_0, d_1, z_avg_list]
    if diversity:
        x_list = [diversity_list, ent_list]
        y_list = [d_1, d_1]

    for ax_id in range(len(ax)):
        ax[ax_id].scatter(x_list[ax_id], y_list[ax_id], s=15, color=colors[0], linewidths=0.3)
        if plot_curve:
            if curve_shape == 'curve':
                popt, pcov = curve_fit(fivepl, x_list[ax_id], y_list[ax_id], maxfev=50000)
                x_fit = np.linspace(min(x_list[ax_id]), max(x_list[ax_id]), 100) 
                y_fit = fivepl(x_fit, *popt) 
                ax[ax_id].plot(x_fit, y_fit, color=colors[0], label="KGW")
            elif curve_shape == 'straight':
                ax[ax_id].plot(x_list[ax_id], y_list[ax_id], color=colors[0], label="KGW") 


def plot_synthid_layer_8(ax, params, temp=1.0, dataset="HealthSearchQA", length=100, split='valid', \
                     plot_curve=False, curve_shape='straight', model_name="mistral", key_id=0, gen_seed=42, \
                        cross=False, semantic='STS', diversity=False):
    d_0, d_1 = [], []
    s, diversity_list = [], []
    z_avg_list = []

    for num_leaves, layer in params:
        fpr = json.load(open(f"FPR/SynthID/len_{length}/layer_{layer}/result.json"))
        z_score_0 = fpr["emp_thres_0.1%"]
        z_score_1 = fpr["emp_thres_1%"]
        file_name_post = f"SynthID/len_{length}_temp_{temp}/test/text/num_leaves_{num_leaves}_layer_{layer}_{model_name}.json_pp"
        
        data = []
        for k in range(1):
            file_name = f"{dataset}/gen_seed_{gen_seed}/key{k}/{file_name_post}"
            with open(file_name, "r") as f:
                data.extend([json.loads(x) for x in f.read().strip().split("\n")])

        if cross:
            cross_flag = '_cross'
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/SynthID/len_{length}_temp_{temp}/test/num_leaves_{num_leaves}_layer_{layer}_{model_name}{cross_flag}.json"
            STS = json.load(open(result_file))[f'STS{cross_flag}']
        else:
            STS = statistics.mean([val['STS'] for val in data])

        if semantic == 'ppl':
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/SynthID/len_{length}_temp_{temp}/test/num_leaves_{num_leaves}_layer_{layer}_{model_name}.json"
            STS = json.load(open(result_file))['avg_loss']

        if diversity:
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/SynthID/len_{length}_temp_{temp}/test/num_leaves_{num_leaves}_layer_{layer}_{model_name}.json"
            diversity = json.load(open(result_file))['avg_diversity']
            diversity_list.append(diversity)

        z_score_list = [val['z_wm'] for val in data]
        thres_0 = sum([1 for val in z_score_list if val > z_score_0]) / len(z_score_list)
        thres_1 = sum([1 for val in z_score_list if val > z_score_1]) / len(z_score_list)
        z_avg = statistics.mean(z_score_list)
        d_0.append(thres_0)
        d_1.append(thres_1)
        s.append(STS)
        z_avg_list.append(z_avg)

    x_list = [s, s]
    if diversity:
        x_list = [diversity_list, diversity_list]
    y_list = [d_0, d_1]
    for ax_id in range(len(ax)):
        ax[ax_id].scatter(x_list[ax_id], y_list[ax_id], color='blue', s=15, linewidths=0.3)
        if plot_curve:
            if curve_shape == 'curve':
                popt, pcov = curve_fit(fivepl, x_list[ax_id], y_list[ax_id], maxfev=50000)
                x_fit = np.linspace(min(x_list[ax_id]), max(x_list[ax_id]), 100) 
                y_fit = fivepl(x_fit, *popt) 
                ax[ax_id].plot(x_fit, y_fit, color='blue', label="SynthID")
            elif curve_shape == 'straight':
                ax[ax_id].plot(x_list[ax_id], y_list[ax_id], color='blue', label="SynthID") 


def plot_ts_valid(ax, ckpt_folder, steps, temp=1.0, dataset="HealthSearchQA", length=100, split='valid', \
                  plot_curve=False, curve_shape='straight', model_name="mistral", gen_seed=42):
    d_0, d_1 = [], []
    s, z_avg_list = [], []
    eval_prefix = f"{dataset}/gen_seed_{gen_seed}/key0/MedMark/len_{length}_temp_{temp}/{split}"
    for i in steps: 
        entry = json.load(open(f"{eval_prefix}/{ckpt_folder}/{i}_{model_name}.json"))
        total = entry['z']['total']
        thres_0 = entry['z']['emp_detected_0.1%'] / total
        thres_1 = entry['z']['emp_detected_1%'] / total
        STS = entry['STS']

        d_0.append(thres_0)
        d_1.append(thres_1)
        s.append(STS)
        z_avg_list.append(entry['z']['avg'])

    x_list = [s, s, s]
    y_list = [d_0, d_1, z_avg_list]
    for ax_id in range(len(ax)):
        ax[ax_id].scatter(x_list[ax_id], y_list[ax_id], color=colors[-1], s=3, linewidths=0.3)
        for j in range(len(s)):
            ax[ax_id].text(x_list[ax_id][j], y_list[ax_id][j], str(j), fontsize=8, ha='right', va='bottom')

def plot_ts_good_ckpts(ax, ckpt_lists, temp=1.0, dataset="HealthSearchQA", length=100, split='valid', \
                       plot_curve=False, curve_shape='straight', model_name="mistral", key_id=0, gen_seed=42, \
                        cross=False, semantic='STS', diversity=False):
    d_0, d_1 = [], []
    s, diversity_list, ent_list = [], [], []
    z_avg_list = []
    for params in ckpt_lists:
        initial, lr, step = params.split('/')
        fpr = json.load(open(f"FPR/MedMark/len_{length}/{initial}_{step}/result.json"))
        z_score_0 = fpr["emp_thres_0.1%"]
        z_score_1 = fpr["emp_thres_1%"]
        if split == 'valid':
            file_name_post = f"MedMark/len_{length}_temp_{temp}/valid/{initial}/{lr}/text/{step}_{model_name}.json_pp"
        else:
            file_name_post = f"MedMark/len_{length}_temp_{temp}/test/text/{initial}_{step}_{model_name}.json_pp"
        
        if key_id == -1:
            data = []
            for k in range(3):
                file_name = f"{dataset}/gen_seed_{gen_seed}/key{k}/{file_name_post}"
                with open(file_name, "r") as f:
                    data.extend([json.loads(x) for x in f.read().strip().split("\n")])
        else:
            file_name = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/{file_name_post}"
            with open(file_name, "r") as f:
                data = [json.loads(x) for x in f.read().strip().split("\n")]
        
        if cross:
            cross_flag = '_cross'
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/MedMark/len_{length}_temp_{temp}/{split}/{initial}_{step}_{model_name}{cross_flag}.json"
            STS = json.load(open(result_file))[f'STS{cross_flag}']
        else:
            STS = statistics.mean([val['STS'] for val in data])

        if semantic == 'ppl':
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/MedMark/len_{length}_temp_{temp}/{split}/{initial}_{step}_{model_name}.json"
            STS = json.load(open(result_file))['avg_loss']
        
        if diversity:
            result_file = f"{dataset}/gen_seed_{gen_seed}/key{key_id}/MedMark/len_{length}_temp_{temp}/{split}/{initial}_{step}_{model_name}.json"
            diversity = json.load(open(result_file))['avg_diversity']
            ent = json.load(open(result_file))['avg_entropy']
            diversity_list.append(diversity)
            ent_list.append(ent)

        z_score_list = [val['z_wm'] for val in data]
        thres_0 = sum([1 for val in z_score_list if val > z_score_0]) / len(z_score_list)
        thres_1 = sum([1 for val in z_score_list if val > z_score_1]) / len(z_score_list)
        z_avg = statistics.mean(z_score_list)
        d_0.append(thres_0)
        d_1.append(thres_1)
        s.append(STS)
        z_avg_list.append(z_avg)
    
    x_list = [s, s, s]
    y_list = [d_0, d_1, z_avg_list]
    if diversity:
        x_list = [diversity_list, ent_list]
        y_list = [d_1, d_1]
           
    if diversity:
        ax[1].invert_xaxis()
        # Add vertical line for no-watermark baseline
        no_wm = [0.2543, 9.9604]
        for ax_id in range(len(ax)):
            ax[ax_id].axvline(x=no_wm[ax_id], color='green', linestyle='--', alpha=0.5, label="No Watermark")
            # ax[ax_id].text(no_wm[ax_id], ax[ax_id].get_ylim()[1], 
            #               rotation=90, va='top', ha='right', color='green')

    for ax_id in range(len(ax)):
        ax[ax_id].scatter(x_list[ax_id], y_list[ax_id], color=colors[-1], s=15, linewidths=0.3)
        if plot_curve:
            if curve_shape == 'curve':
                popt, pcov = curve_fit(fivepl, x_list[ax_id], y_list[ax_id], maxfev=40000)
                x_fit = np.linspace(min(x_list[ax_id]), max(x_list[ax_id]), 100) 
                y_fit = fivepl(x_fit, *popt)
                ax[ax_id].plot(x_fit, y_fit, color=colors[-1], label="MedMark") 
            elif curve_shape == 'straight':
                ax[ax_id].plot(x_list[ax_id], y_list[ax_id], color=colors[-1], label="MedMark") 

        if split == 'test':
            if diversity:
                ax[ax_id].set_xlabel(diversity_label[ax_id], fontsize=11)
                ax[ax_id].legend(loc="lower left", fontsize=8) 
            elif semantic == 'STS':
                ax[ax_id].set_xlabel('Semantic Similarity', fontsize=11)
                ax[ax_id].legend(loc="lower left", fontsize=8) 
            elif semantic == 'ppl':
                ax[ax_id].set_xlabel('$\\leftarrow$log(Perplexity)', fontsize=11)
                ax[ax_id].legend(loc="lower right", fontsize=8) 

            ax[ax_id].set_ylabel('TPR', fontsize=11)
            if not diversity:
                ax[ax_id].set_title(title[ax_id], fontsize=11)
            else:
                ax[ax_id].set_title(title[1], fontsize=11)
