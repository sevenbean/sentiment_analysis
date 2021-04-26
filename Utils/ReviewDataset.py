from torch.utils.data import  Dataset
import ast
'''
    -*- coding:utf-8 -*-
    @author:erdou(cwj)
    @time:2021/4/25_14:56
    @filename:ReviewDataset.py
    @description:将所有的文本数据添加到DataSet中，以实现数据的打乱等
'''

class ReviewDataSet(Dataset):
    def __init__(self,tokenizer,dataset_name):
        all_data=[]
        for path in dataset_name:
                with open(path,"r",errors="ignore",encoding="utf8") as file:
                    lines=file.readlines()
                    for line in lines:
                        text=ast.literal_eval(line)["text"]
                        label=ast.literal_eval(line)["label"]
                        sequences=tokenizer.text_to_sequence(text)
                        all_data.append({"text":sequences,"label":label})
        self.data=all_data
    def __getitem__(self, index):
            return self.data[index]
    def __len__(self):
        return len(self.data)