# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import math
from torch.autograd import Variable

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_k = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_q = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)
        # print('v的形状：', v.shape)
        # print('k的形状：', k.shape)
        # print('q的形状：', q.shape)
        atted = self.att(v, k, q, mask)
        # print('atted的形状：', atted.shape)



        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['fusion']['mca_HIDDEN_SIZE']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class MHAttForSGA(nn.Module):
    def __init__(self, __C):
        super(MHAttForSGA, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_k = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_q = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)
        # print(n_batches)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)
        # print('v的形状：', v.shape)
        # print('k的形状：', k.shape)
        # print('q的形状：', q.shape)


        atted = self.att(v, k, q, mask)
        # print('atted的形状：', atted.shape)
        atted = atted.contiguous().view(
            n_batches,
            -1,
            self.__C['fusion']['mca_HIDDEN_SIZE']
        )
        # print('变换后atted的形状：', atted.shape)
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C['fusion']['mca_HIDDEN_SIZE'],
            mid_size=__C['fusion']['mca_FF_SIZE'],
            out_size=__C['fusion']['mca_HIDDEN_SIZE'],
            dropout_r=__C['fusion']['mca_DROPOUT_R'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, x_mask=None):
        # print('x的形状：', x.shape)
        flag = self.mhatt(x, x, x, x_mask)
        # print('flag的形状：', flag.shape)
        tmp = self.dropout1(flag)
        # print('tmp的形状：', tmp.shape)

        x = self.norm1(x + tmp)

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAttForSGA(__C)
        self.mhatt2 = MHAttForSGA(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout3 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm3 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, y, x_mask=None, y_mask=None):
        # print('SGA中x的形状：', x.shape)
        flag = self.mhatt1(x, x, x, x_mask)
        # print('flag的形状：', flag.shape)
        tmp = self.dropout1(flag)
        # print('tmp的形状：', tmp.shape)

        x = self.norm1(x + tmp)

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

