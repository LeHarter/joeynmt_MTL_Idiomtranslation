import torch
from torch import Tensor, nn
#from torchcrf import CRF

class IdiomTagger(nn.Module):
    #def __init__(self,emb_size,hidden_size,num_layers,num_classes,dropout):
    def __init__(self,emb_size,hidden_size,num_classes,dropout):
        super().__init__()
        self.num_classes = num_classes
        #self.lstm = nn.LSTM(input_size=emb_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=True)
        #self.linear = nn.Linear(hidden_size*2,num_classes)
        self.linear = nn.Linear(hidden_size,num_classes)
    def forward(self,encoder_output):
        #output, (hn, cn) = self.lstm(encoder_output)
        #return self.linear(output)
        return self.linear(encoder_output)
class IdiomTagger_CRF(nn.Module):
    def __init__(self,emb_size,hidden_size,num_layers,num_classes,dropout):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=emb_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=True)
        
