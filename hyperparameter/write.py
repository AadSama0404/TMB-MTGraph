# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 22:03
@Author  : AadSama
@Software: Pycharm
"""
'''
This file is used to assign hyperparameters to ensure fairness in comparative experiments.
'''

import torch
lrs = [[0.001, 0.001, 0.001, 0.001],
       [0.001, 0.001, 0.001, 0.001],
       [0.001, 0.001, 0.001, 0.001]]
torch.save(lrs, 'lrs.pt')