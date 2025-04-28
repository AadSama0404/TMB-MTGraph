# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 22:21
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore")


def Violin_Plot(file_path):
    df = pd.read_csv(file_path).iloc[:, :5]

    plt.figure(figsize=(4, 3))
    sns.set(style="white")

    x_labels = [f'Clone {i + 1}' for i in range(5)]
    df.columns = x_labels

    palette = sns.color_palette("Blues_r", n_colors=df.shape[1])

    ## Plot a violin plot
    ax = sns.violinplot(data=df, palette=palette)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=10)

    ## Add mean (only calculate non-zero data)
    means = df.apply(lambda x: x[x != 0].mean() if (x != 0).any() else 0)
    for j, mean in enumerate(means):
        ax.text(j, 1.3, f'{mean:.3f}',
                horizontalalignment='center',
                color='black',
                fontsize=10)

    ax.set_ylabel("Clone Weight", fontsize=10)
    ax.set_title("Clone Weight Distribution", fontsize=12, pad=10)

    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 0.5, 1], fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.show()


def Heatmap_Plot(file_path, subgroup_num):
    def calculate_means(file_path, groups, clones):
        df = pd.read_csv(file_path, header=None)
        results = []
        group_column = df.iloc[:, -1]
        for group in range(1, groups + 1):
            ## Select the data of the current subgroup
            group_df = df[group_column == group]

            ## Calculate the mean of each feature in the current group, ignoring 0 values
            mean_values = group_df.iloc[:, :-1].apply(lambda x: x[x != 0].mean() if (x != 0).any() else 0, axis=0)
            if len(mean_values) < clones:
                mean_values = np.pad(mean_values, (0, clones - len(mean_values)), 'constant', constant_values=0)
            results.append(mean_values)

        return np.array(results)

    results1 = calculate_means(file_path, groups=subgroup_num, clones=8)

    all_results = pd.DataFrame(results1).T
    all_results.index = [f'Clone {i + 1}' for i in range(8)]
    all_results.columns = [f'Experimental {i + 1}' for i in range(subgroup_num)]

    ## Plot a heatmap
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(all_results, cmap='Blues', cbar=True, linewidths=0.9, linecolor='gainsboro', square=False,
                cbar_kws={"shrink": 1}, vmin=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.gca().tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(left=0.15, right=1, bottom=0.3)
    plt.show()


if __name__ == "__main__":
    file_path = '../results/attention_weights.csv'
    Violin_Plot(file_path)
    subgroup_num = torch.load("../data/subgroup_num.pt")
    Heatmap_Plot(file_path, subgroup_num)