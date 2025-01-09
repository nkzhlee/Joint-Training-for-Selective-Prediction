import torch
import torch.nn as nn
from transformers import AutoModel
from Constants import *


class SFRNModel(nn.Module):
    def __init__(self):
        super(SFRNModel, self).__init__()
        # Define the pre-trained model and tokenizer
        #self.device = DEVICE
        self.bert = AutoModel.from_pretrained(hyperparameters['model_name'])
        self.dropout = torch.nn.Dropout(hyperparameters['hidden_dropout_prob'])
#         self.bert = transformers.BertModel.from_pretrained(model_name, config=config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Define the number of labels in the classification task
        num_labels = hyperparameters['num_labels']
        # A single layer classifier added on top of BERT to fine tune for binary classification
        mlp_hidden = hyperparameters['mlp_hidden']
        self.g = nn.Sequential(
            nn.Linear(hyperparameters['hidden_dim'], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
            # nn.Linear(mlp_hidden, mlp_hidden),
            # nn.ReLU(),
        )

        self.f = nn.Sequential(
            #nn.Linear(mlp_hidden + hyperparameters['gpt_dim'], hyperparameters['mlp_hidden']),
            nn.Linear(mlp_hidden, hyperparameters['mlp_hidden']),
            nn.ReLU(),
#             nn.Linear(hyperparameters['mlp_hidden'], hyperparameters['mlp_hidden']//2),
#             nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyperparameters['mlp_hidden'], num_labels),
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

    def forward(self, input_ids, emb, token_type_ids=None, attention_mask=None):
        # Forward pass through pre-trained BERT
#         print('input_ids: ', input_ids.size())
#         print('attention_mask: ', attention_mask.size())
#         print('emb: ', emb.size())
        outputs = self.bert(input_ids.squeeze(), attention_mask=attention_mask.squeeze())
        pooled_output = outputs[-1]
        pooled_output = self.dropout(pooled_output)
        #print("pooled_output: {}".format(pooled_output.size()))
        g_t = self.g(pooled_output)
        g_t = self.alpha(g_t) * g_t + self.beta(g_t)
        #print("g_t: {}".format(g_t.size()))
        g = g_t.sum(0)
        #x = torch.cat((g.unsqueeze(0), emb), dim=1) 
        #g = g_t.sum(1) + g_t.prod(1)
        #output = self.f(x)
        output = self.f(g.unsqueeze(0))
        #print("f: {}".format(output.size()))
        logits = torch.softmax(output, dim=1)
        return logits, g
        # return self.classifier(pooled_output)[0].unsqueeze(0)
        
class DeferralClassifier(nn.Module):
    def __init__(self, input_dim=256, gpt_dim=1536, output_dim=2):
        super(DeferralClassifier, self).__init__()
        
        self.defer_layer = AutoModel.from_pretrained(hyperparameters['defer_model_name'])
        self.dropout = torch.nn.Dropout(hyperparameters['hidden_dropout_prob'])
        # Update the input dimension to 2 * input_dim because we are concatenating two inputs
        #self.fc1 = nn.Linear(hyperparameters['mlp_hidden'] , hyperparameters['policy_hidden'])
        #self.fc1 = nn.Linear(hyperparameters['gpt_dim'], hyperparameters['policy_hidden'])
        self.fc1 = nn.Sequential(
            #nn.Linear(hyperparameters['mlp_hidden'] + hyperparameters['gpt_dim'], hyperparameters['policy_hidden']*2),
            #nn.Linear(hyperparameters['gpt_dim'], hyperparameters['policy_hidden']*2),
            nn.Linear(hyperparameters['p_hidden_dim'], hyperparameters['policy_hidden']),
            #nn.Linear(hyperparameters['mlp_hidden']+ hyperparameters['p_hidden_dim'], hyperparameters['policy_hidden']),
            nn.ReLU(),
            nn.Dropout(),
#             nn.Linear(hyperparameters['policy_hidden'], hyperparameters['policy_hidden']),
#             #nn.Linear(hyperparameters['gpt_dim'], hyperparameters['policy_hidden']),
#             nn.ReLU(),
#             nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hyperparameters['policy_hidden'], hyperparameters['policy_hidden']//2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyperparameters['policy_hidden']//2, output_dim)
        )
#         self.fc2 = nn.Linear(hyperparameters['policy_hidden'], output_dim)
        # Sigmoid activation for binary classification
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, attention_mask=None):
#         print('emd: ', x1.size())
#         print('attention_mask: ', attention_mask.size())
        # Concatenate the two input tensors along dimension 1 for GPT emb
        #x = torch.cat((x1, x2), dim=1)  
        #x = x2
        #x = x1.float()
        # for new bert repre
        x = self.defer_layer(x1, attention_mask=attention_mask)
        pooled_output = x[-1]
        pooled_output = self.dropout(pooled_output)
#         print('pooled_output: ', pooled_output.size())
#         print('x2: ', x2.size())
        #pooled_output = torch.cat((pooled_output, x2), dim=1)
        temp = self.fc1(pooled_output)
        #x = torch.relu(self.fc1(x))
        # Forward pass through the linear layer
        out = self.fc2(temp)
        #print('out: ', out.size())
        # Pass the result through the sigmoid activation
        #out = self.sigmoid(out)
        
        return out