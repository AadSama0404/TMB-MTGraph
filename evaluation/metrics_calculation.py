# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 21:55
@Author  : AadSama
@Software: Pycharm
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, average_precision_score
from scipy.stats import mannwhitneyu


def Metrics_Calculation(file_path):
    ## Read the result file
    result_df = pd.read_csv(file_path)

    ## Initialize and store the index of each fold
    auc_list = []
    acc_list = []
    nll_list = []
    auprc_list = []

    ## Get the fold index
    folds = result_df['fold'].unique()

    ## Two-tailed U test
    group_0_scores = result_df[result_df['true_label'] == 0]['score']
    group_1_scores = result_df[result_df['true_label'] == 1]['score']
    u_stat, p_value = mannwhitneyu(group_0_scores, group_1_scores, alternative='two-sided')

    ## Calculate each fold
    for fold in folds:
        ## Filter out the data of the current fold
        fold_df = result_df[result_df['fold'] == fold]

        ## Extract the required columns
        true_labels = fold_df['true_label']
        scores = fold_df['score']
        predicted_labels = fold_df['group']

        ## Calculate area under curve
        auc = roc_auc_score(true_labels, scores)
        auc_list.append(auc)

        ## Calculate accuracy
        acc = accuracy_score(true_labels, predicted_labels)
        acc_list.append(acc)

        ## Calculate negative log-likelihood loss
        nll = log_loss(true_labels, scores)
        nll_list.append(nll)

        ## Calculate area under the precision-recall curve
        auprc = average_precision_score(true_labels, scores)
        auprc_list.append(auprc)

    ## Calculate the average of each item
    avg_auc = sum(auc_list) / len(auc_list)
    avg_acc = sum(acc_list) / len(acc_list)
    avg_nll = sum(nll_list) / len(nll_list)
    avg_auprc = sum(auprc_list) / len(auprc_list)

    ## print results
    print(f"Average Negative Log-Likelihood Loss: {avg_nll:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    print(f"Average AUPRC: {avg_auprc:.4f}")
    print(f"P-value for classification: {p_value:.4e}")


if __name__ == "__main__":
    file_path = '../results/TMB_MTGraphcsv'
    Metrics_Calculation(file_path)