import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,device='cuda:0',K=3):
        super().__init__()
        self.device = device
        self.liner = nn.Sequential(
            nn.Linear(770, 770, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(770, 770, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(770, 770, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.outer=nn.Linear(770, 2)

    def forward(self, sentences, visions_tensors):
        visions_tensors=visions_tensors.to('cuda')
        sentences_model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
        embeddings = sentences_model.encode(sentences,convert_to_tensor=True).to('cuda')
        # embeddings=embeddings.unsqueeze(0)
        input=torch.cat([embeddings,visions_tensors],dim=1)
        input=self.liner(input)
        input=self.outer(input)
        input_softmax=F.softmax(input, dim=1)
        return input_softmax;






