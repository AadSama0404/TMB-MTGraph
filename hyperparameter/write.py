# -*- coding: utf-8 -*-
"""
@Time    : 2025/4/28 22:03
@Author  : AadSama
@Software: Pycharm
"""
import torch
lrs = [[0.001, 0.001, 0.001, 0.001],
       [0.001, 0.001, 0.001, 0.001],
       [0.001, 0.001, 0.001, 0.001]]
torch.save(lrs, 'lrs.pt')