import json
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:  # 指定文件编码为utf-8
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        result = item['result']
        vision = item['vision']
        if(vision=="00"):
            vision_tensor=torch.tensor([0,0])
        if (vision == "01"):
            vision_tensor = torch.tensor([0, 1])
        if (vision == "10"):
            vision_tensor = torch.tensor([1, 0])
        if (vision == "11"):
            vision_tensor = torch.tensor([1, 1])
        return sentence, vision_tensor,result
