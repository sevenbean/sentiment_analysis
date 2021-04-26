'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/4/25_16:50
    @filename:
    @description:
'''
import torch.nn as nn
import torch
class LSTM(nn.Module):
    def __init__(self,emdedding_matrix,opt):
        super(LSTM, self).__init__()
        self.embeding=nn.Embedding.from_pretrained(torch.tensor(emdedding_matrix,dtype=torch.float))
        self.lstm=nn.LSTM(opt.embed_dim,opt.hidden_dim,num_layers=1,batch_first=True)
        self.dense=nn.Linear(opt.hidden_dim,opt.polarities_dim)
    def forward(self,inputs):
        x_embeding=self.embeding(inputs)
        #outputs表示所有隐藏层的输出，h_n表示最后一个单词的隐藏层的输出，c_n_最后一个单词的记忆单元的状态
        outputs,(h_n,c_n)=self.lstm(x_embeding)
        out=self.dense(h_n[0])
        return out

