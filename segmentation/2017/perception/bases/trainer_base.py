"""
Copyright (c) 2019. All rights reserved.
Created by Peng Tang on 2019/1/7
"""


class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model, data, config,iter_):
        self.model = model  # 模型
        self.data = data  # 数据
        self.config = config  # 配置
        self.iter_ = iter_
    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError
