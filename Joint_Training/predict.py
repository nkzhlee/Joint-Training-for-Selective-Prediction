import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
from Constants import *
from DataModules import SequenceDataset
from SFRNModel import SFRNModel

'''
 Test acc is 0.873015873015873
110 Test qwk is 0.8164027608607389
111 Test f1 is 0.5741574514002744
112 Real Test:
113 start to test at checkpoint/checkpoint_6_0904_ASAP_set56_test0451_at_epoch9.model
114 Test acc is 0.873015873015873
115 Test f1 is 0.5741574514002744
116 Quadratic Weighted Kappa is 0.8164027608607389
117               precision    recall  f1-score   support
118            0       0.95      0.96      0.95       960
119            1       0.64      0.60      0.62       157
120            2       0.42      0.19      0.26        52
121            3       0.36      0.64      0.46        28
122     accuracy                           0.87      1197
123    macro avg       0.59      0.60      0.57      1197
124 weighted avg       0.87      0.87      0.87      1197
125 [[923  34   1   2]
126  [ 51  94   7   5]
127  [  2  15  10  25]
128  [  0   4   6  18]]

'''


def save_results_to_csv(y_true, y_pred, csv_file_path='results.csv'):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.to_csv(csv_file_path, index=False)

# Usage in your train function



def train(args):
    random.seed(hyperparameters['random_seed'])
    # model
    best_acc, best_f1 = 0, 0
    best_ckp_path = 'checkpoint/checkpoint_6_0904_ASAP_set56_test0451_at_epoch9.model'
    DEVICE = args.device
    print(DEVICE)
    model_name = hyperparameters['model_name']
    print(model_name)
    print(hyperparameters)
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load Train dataset and split it into Train and Validation dataset
    test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    testset_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    model = SFRNModel()
    model.to(DEVICE)
    print('Real Test: \n')
    with torch.no_grad():
        test_correct_total = 0
        print("start to test at {} ".format(best_ckp_path))
        model.load_state_dict(torch.load('./' + best_ckp_path))
        model.eval()
        y_true = list()
        y_pred = list()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for step, batch in enumerate(test_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask=attention_mask)
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            # break
        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        f1 = f1_score(y_true, y_pred, average='macro')
        print("Test acc is {} ".format(acc))
        print("Test f1 is {} ".format(f1))
        print("Quadratic Weighted Kappa is {}".format(qwk))
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        save_results_to_csv(y_true, y_pred, 'set56_results.csv')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt',
                        help='ckp_name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()