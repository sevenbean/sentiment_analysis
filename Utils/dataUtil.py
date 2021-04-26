import numpy as np
from gensim.models import KeyedVectors
import pickle
import re
import jieba
import os
import ast
'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/4/25_14:56
    @filename:dataUtil.py
    @description:数据的预处理
'''
class Tokenizer(object):
    def __init__(self,max_seq_len,wordvec_path,embed_dim):
        super(Tokenizer,self).__init__()
        self.word2id={}
        self.id2word={}
        self.idx=1
        self.max_seq_len=max_seq_len
        self.wordvec_path=wordvec_path #已完成训练的词向量文件路径
        self.embed_dim=embed_dim #每个词的维度

    #加载已经训练好的词向量
    def load_wordvector(self):
        embeding_path = "./wordvec/zhihui.bin"
        print("------------load wordvec-----------")
        if (os.path.exists(embeding_path)):
            wordvec = KeyedVectors.load(embeding_path)
        else:
            wordvec = KeyedVectors.load_word2vec_format(self.wordvec_path, binary=False)
            wordvec.save(embeding_path)
        return wordvec

    #文本数据的填充和剪切
    def pad_and_truncat(self,sequence,padding="post", truncating="post", dtype="int64",value=0):
        x = (np.ones(self.max_seq_len) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-self.max_seq_len:]
        else:
            trunc = sequence[:self.max_seq_len]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    #创建该语料的word2idx和idx2word
    def fit_on_text(self,text):
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
        word_list=[word for word in jieba.cut(text)]
        for word in word_list:
            if word not in self.word2id:
                self.word2id[word]=self.idx
                self.id2word[self.idx]=word
                self.idx+=1
    #将文本数据转为对应的数字序列
    def text_to_sequence(self,text,padding='post', truncating='post'):
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        word_list=[word for word in jieba.cut(text)]
        sequences=[self.word2id[word]if word in self.word2id.keys() else 0 for word in word_list]
        if len(sequences)==0:
            sequences=[0]
        return self.pad_and_truncat(sequences,padding=padding, truncating=truncating)

    #将序列转为文本，以验证文本转为数字序列是否正确
    def sequence_to_text(self,text):
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        word_list = [word for word in jieba.cut(text)]
        sequences = [self.word2id[word] for word in word_list]
        text=[self.id2word[idx] for idx in sequences]
        return text

    #构建词向量矩阵
    def build_embeding_matrix(self):
        dat_fname="./wordvec/embedding_matrix.dat"
        if (os.path.exists(dat_fname)):
            print("loading_embedding_matrix:", dat_fname)
            embedding_matrix = pickle.load(open(dat_fname, "rb"))
        else:
            print("loading word vector.......")
            embedding_matrix = np.zeros((len(self.word2id) + 1, self.embed_dim))
            wordvec = self.load_wordvector()
            print(self.id2word[3479])
            for word, index in self.word2id.items():
                if word in wordvec.vocab.keys():
                    embedding_matrix[index] = wordvec.get_vector(word)
            pickle.dump(embedding_matrix, open(dat_fname, "wb"))
        return embedding_matrix


def build_tokenizer(max_seq_len,wordvec_path,embed_dim,dataset_path):
    tokenizer_path= "./wordvec/tokenizer.dat"
    if(os.path.exists(tokenizer_path)):
        tokenizer=pickle.load(open(tokenizer_path,"rb"))
    else:
        text = ""
        for i in dataset_path:
            with open(i,"r",encoding="utf8",errors="ignore") as file:
                lines=file.readlines()
                for line in lines:
                    text+=ast.literal_eval(line)["text"]+" "
        tokenizer=Tokenizer(max_seq_len,wordvec_path,embed_dim)
        tokenizer.fit_on_text(text)
        print(tokenizer.build_embeding_matrix())
        pickle.dump(tokenizer,open(tokenizer_path,"wb"))
    return tokenizer




