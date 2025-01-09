import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import cohen_kappa_score
from models import SFRN
from DataModules import ColSTATDataset
from constants import *
import gc
gc.collect()
torch.cuda.empty_cache()


def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def train(args):
    wandb.init(project="HITL", entity="zhaohuilee", config=config_dictionary)
    random.seed(param['random_seed'])
    print(param)
    print(args.ckp_name)
    # model
    best_acc = 0
    best_ckp_path = ''
    DEVICE = args.device
    print(DEVICE)
    model_name = param['model_name']
    print(model_name)
    # Load the tokenizer config
    config = AutoConfig.from_pretrained(model_name)
    config.max_length = param['max_length']
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    # Load Train dataset and split it into Train and Validation dataset
    train_dataset = ColSTATDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    test_dataset = ColSTATDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    test_dataset.tag2id = train_dataset.tag2id
    trainset_size = len(train_dataset)
    testset_size = len(test_dataset)
    shuffle_dataset = True
    validation_split = param['validation_split']
    indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'],
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'],
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    training_acc_list, validation_acc_list = [], []

    model = SFRN(DEVICE)
    model.to(DEVICE)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    #scheduler = StepLR(optimizer, step_size=param['lr_step'], gamma=param['lr_gamma'])
    num_training_steps = len(train_loader) * param['epochs']
    warmup_steps = int(param['WARMUP_STEPS'] * num_training_steps)  # 10% of total training steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                               num_training_steps=num_training_steps)
    model.train()
    # Training Loop
    for epoch in range(param['epochs']):
        print("Training Epoch: {} LR is {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train_loss, cc_loss = 0, 0
        model.train()
        epoch_loss = 0.0
        train_correct_total = 0
        y_true = list()
        y_pred = list()
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            seq_pos = batch["seq_pos"].to(DEVICE)
            logits, states = model(input_ids, seq_pos, attention_mask=attention_mask)
            #logits, loss = outputs.logits, outputs.loss
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            loss = loss.data.cpu().numpy()
            if type(loss) is list: loss = loss[0]
            train_loss += loss
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            nn.utils.clip_grad_norm_(model.parameters(), param['max_norm'])  # Optional: Gradient clipping
            optimizer.step()
            # this is for linear_scheduler
            #scheduler.step()
        
        train_acc = accuracy_score(y_true, y_pred)
        print('Epoch {} -'.format(epoch))
        # this is for step LR
        scheduler.step()
        # Validation Loop
        val_loss, val_cc_loss, val_ct_loss = 0, 0, 0
        with torch.no_grad():
            model.eval()
            val_y_true = list()
            val_y_pred = list()
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                seq_pos = batch["seq_pos"].to(DEVICE)
                logits, states = model(input_ids, seq_pos, attention_mask=attention_mask)
                _, predicted = torch.max(logits.data, 1)
                val_y_pred += list(predicted.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                vloss = criterion(logits, labels)
                vloss = vloss.data.cpu().numpy()
                if type(vloss) is list: vloss = vloss[0]
                val_loss += vloss
            val_acc = accuracy_score(val_y_true, val_y_pred)
            print('Training Accuracy {} - Validation Accurracy {}'.format(
                train_acc, val_acc))
            print('Training loss {} - Validation Loss {}'.format(
                train_loss, val_loss))
            if val_acc > best_acc:
                best_acc = val_acc
                with open(
                        './checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch)), 'wb'
                ) as f:
                    torch.save(model.state_dict(), f)
                best_ckp_path = './checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch))

        with torch.no_grad():
            test_correct_total = 0
            model.eval() 
            y_true = list()
            y_pred = list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            for step, batch in enumerate(test_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                seq_pos = batch["seq_pos"].to(DEVICE)
                logits, states = model(input_ids, seq_pos, attention_mask=attention_mask)
                pred_idx = torch.max(logits, 1)[1]
                y_true += list(labels.data.cpu().numpy())
                y_pred += list(pred_idx.data.cpu().numpy())
                # break
            acc = accuracy_score(y_true, y_pred)
            print("Test acc is {} ".format(acc))
            wandb.log(
            {"Train loss": train_loss, "Val loss": val_loss, "Train Acc": train_acc, "Val Acc": val_acc, "test Acc": acc})
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
            seq_pos = batch["seq_pos"].to(DEVICE)
            logits, states = model(input_ids, seq_pos, attention_mask=attention_mask)
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        print("Test qwk is {} ".format(qwk))
        print("Test acc is {} ".format(acc))
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        wandb.log({"final_test Acc": acc})
        # output result
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt',
                        help='ckp_name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
    args = parser.parse_args()
    seed_everything(param['random_seed'])
    train(args)

if __name__ == '__main__':
    main()