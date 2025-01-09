import torch
import torch.nn as nn
from transformers import AutoModel
from constants import *

# Define the sequence classification model
class SFRN(torch.nn.Module):
    def __init__(self, DEVICE):
        super(SFRN, self).__init__()
        self.device = DEVICE
        self.bert = AutoModel.from_pretrained(param['model_name'])
        self.dropout = torch.nn.Dropout(param['hidden_dropout_prob'])
        self.classifier = torch.nn.Linear(param['hidden_dim'], param['num_labels'])
        mlp_hidden = param['mlp_dim']
        self.g = nn.Sequential(
            nn.Linear(param['hidden_dim'], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
#             nn.Linear(mlp_hidden, mlp_hidden),
#             nn.ReLU(),
#             nn.Linear(mlp_hidden, mlp_hidden),
#             nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        self.alpha = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.beta = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.f = nn.Sequential(
#             nn.Linear(mlp_hidden, mlp_hidden),
#             nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, param['num_labels']),
        )

    def forward(self, input_ids, seq_pos, attention_mask=None):
        print('input_ids: ', input_ids.size())
        print('attention_mask: ', attention_mask.size())
        outputs = self.bert(input_ids.squeeze(), attention_mask=attention_mask.squeeze())

        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1]
        pooled_output = self.dropout(pooled_output)
        print("pooled_output: {}".format(pooled_output.size()))
        g_t = self.g(pooled_output)

        g_t = self.alpha(g_t) * g_t + self.beta(g_t)
        #print("g_t: {}".format(g_t.size()))
        g = g_t.sum(0)
        #print("g: {}".format(g.size()))
        #g = g_t.sum(1) + g_t.prod(1)
        output = self.f(g.unsqueeze(0))
        #print("f: {}".format(output.size()))
        return output, g
        # return self.classifier(pooled_output)[0].unsqueeze