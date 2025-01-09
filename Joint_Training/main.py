import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
from Constants import *
from DataModules import SequenceDataset
from SFRNModel import SFRNModel, DeferralClassifier  


def train(args):
    wandb.init(project="Your_project", entity="Your_usrname", config=config_dictionary)
    random.seed(hyperparameters['random_seed'])
    # model
    best_acc, best_f1 = 0, 0
    best_ckp_path = ''
    DEVICE = args.device
    print(DEVICE)
    model_name = hyperparameters['model_name']
    print(model_name)
    print(hyperparameters)
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load Train dataset and split it into Train and Validation dataset
    train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    test_dataset.tag2id = train_dataset.tag2id
    trainset_size = len(train_dataset)
    testset_size = len(test_dataset)
    shuffle_dataset = True
    validation_split = hyperparameters['data_split']
    indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    training_acc_list, validation_acc_list = [], []

    model = SFRNModel()
    policy = DeferralClassifier()
    model.to(DEVICE)
    policy.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    optimizer_p = AdamW(policy.parameters(), lr=hyperparameters['p_lr'], weight_decay=hyperparameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    #criterion_p = nn.BCELoss()
    # Assuming class 1 (positive class) is rares
    weights = torch.tensor([0.01, 10.0]).to(DEVICE)
    criterion_p = nn.CrossEntropyLoss(weight=weights)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_training_steps = len(train_loader) * hyperparameters['epochs']
    warmup_steps = int(hyperparameters['WARMUP_STEPS'] * num_training_steps)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    scheduler_p = get_linear_schedule_with_warmup(optimizer_p, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    model.train()
    grad_accumulator_steps = hyperparameters['GRADIENT_ACCUMULATION_STEPS']
    # Training Loop
    for epoch in range(hyperparameters['epochs']):
        if epoch < hyperparameters['pre_step']:
            model.train()
            policy.eval()
        elif epoch < hyperparameters['mid_step']: 
            model.eval()
            policy.train()
        else:
            policy.train()
            model.train()
        train_loss, train_policy_loss, train_cl_loss = 0.0, 0.0, 0.0
        y_true, p_true = list(), list()
        y_pred, p_pred = list(), list()
        y_defer = list()
        tmp_defer = list()
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            emb = batch["embedding"].to(DEVICE)
            emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            # classifier
            logits, hidden = model(input_ids, emb, attention_mask=attention_mask)
            # create a label for deferal policy
            defer_label = [1]
            pred_idx = torch.max(logits, 1)[1]
            if pred_idx[0] == labels[0]: defer_label = [0]
            defer_label = torch.tensor(defer_label).to(DEVICE)
            # policy 
            defer_logits = policy(emb, hidden.unsqueeze(dim=0),emb_attention_mask)
            p_pred_idx = torch.max(defer_logits, 1)[1]
            loss_cl = criterion(logits, labels) #/ GRADIENT_ACCUMULATION_STEPS
            loss_defer = criterion_p(defer_logits, defer_label)
            # Scale loss for gradient accumulation
            loss_cl /= grad_accumulator_steps
            loss_defer /= grad_accumulator_steps
            if p_pred_idx[0] == 1:
                y_defer += list(labels.data.cpu().numpy())
            else:
                y_defer += list(pred_idx.data.cpu().numpy())
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            # policy
            p_true += list(defer_label.data.cpu().numpy())
            p_pred += list(p_pred_idx.data.cpu().numpy())
            tmp_defer += list(p_pred_idx.data.cpu().numpy())
            if epoch < hyperparameters['pre_step']:
                loss = loss_cl
            elif epoch < hyperparameters['mid_step']: 
                loss = loss_defer 
            else:
                loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer
#                 if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0: 
#                     #print(tmp_defer)
#                     tmp_rate = tmp_defer.count(1) / len(tmp_defer)
#                     true_rate = p_true.count(1) / len(p_true)
#                     r = (tmp_rate - true_rate) / (tmp_rate + true_rate + 1)

# #                     r = (tmp_rate - true_rate)
#                     loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer + hyperparameters['gamma'] * tmp_rate
#                 else: 
#                     loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer 
            loss.backward()
            loss = loss.data.cpu().numpy()
            train_loss += loss
            train_cl_loss += loss_cl.data.cpu().numpy()
            train_policy_loss += loss_defer.data.cpu().numpy()
            if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_norm'])
                nn.utils.clip_grad_norm_(policy.parameters(), hyperparameters['max_norm'])
                # renew 
                tmp_defer = list()
                if epoch < hyperparameters['pre_step']:
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler.step()
                elif epoch < hyperparameters['mid_step']: 
                    optimizer_p.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler_p.step()
                else:
                    optimizer.step()
                    optimizer_p.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler.step()
                    scheduler_p.step()
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        train_acc = accuracy_score(y_true, y_pred)
        train_policy_acc = accuracy_score(p_true, p_pred)
        train_policy_f1 = f1_score(p_true, p_pred, average='macro')
        print('Epoch {} - Loss {}'.format(epoch + 1, train_loss))
        #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            model.eval()
            policy.eval()
            val_loss, val_policy_loss, val_cl_loss = 0.0, 0.0, 0.0
            val_y_true, val_p_true = list(), list()
            val_y_pred, val_p_pred = list(), list()
            defer_count = 0
            val_y_defer = list()
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                emb = batch["embedding"].to(DEVICE)
                emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits, hidden = model(input_ids, emb, attention_mask=attention_mask)
                defer_label = [1]
                pred_idx = torch.max(logits, 1)[1]
                if pred_idx[0] == labels[0]: defer_label = [0]
                defer_label = torch.tensor(defer_label).to(DEVICE)
                defer_logits = policy(emb, hidden.unsqueeze(dim=0),attention_mask=emb_attention_mask)
                p_pred_idx = torch.max(defer_logits, 1)[1]
                # loss
                #print(p_pred_idx)
                if p_pred_idx[0] == 1:
                    val_y_defer += list(labels.data.cpu().numpy())
                    defer_count += 1
                else:
                    val_y_defer += list(pred_idx.data.cpu().numpy())
                loss_cl = criterion(logits, labels) #/ GRADIENT_ACCUMULATION_STEPS
                loss_defer = criterion_p(defer_logits, defer_label)
                if epoch > hyperparameters['pre_step']:
                    loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer
                else:
                    loss = loss_cl                
                val_loss += loss.data.cpu().numpy()
                val_cl_loss += loss_cl.data.cpu().numpy()
                val_policy_loss += loss_defer.data.cpu().numpy()
                val_y_pred += list(pred_idx.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                # policy
                val_p_true += list(defer_label.data.cpu().numpy())
                val_p_pred += list(p_pred_idx.data.cpu().numpy())
            val_acc = accuracy_score(val_y_true, val_y_pred)
            val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
            val_qwk = cohen_kappa_score(val_y_true, val_y_pred, weights='quadratic')
            val_policy_acc = accuracy_score(val_p_true, val_p_pred)
            val_policy_f1 = f1_score(val_p_true, val_p_pred, average='macro')
            val_defer_acc = accuracy_score(val_y_true, val_y_defer)
            val_defer_f1 = f1_score(val_y_true, val_y_defer, average='macro')
            val_defer_rate = defer_count/len(val_y_defer)
            print('Training Accuracy {} Deferral Acc {} - Validation Accurracy {}, Deferral Acc {}'.format(
                train_acc, train_policy_acc, val_acc, val_policy_acc))
            print('Training F1 {} Deferral F1 {}- Validation F1 {} Deferral F1 {}'.format(
                train_f1, train_policy_f1, val_f1, val_policy_f1))
            print('The Val Acc/F1 after defer is {}/{}, the deferral rate is {}'.format(val_defer_acc, val_defer_f1, val_defer_rate))
            print('Training Loss {}, Classification loss {}, Polciy loss {} - Validation Loss {}, Classification Loss {}, Polciy loss {}'.format(
                train_loss, train_cl_loss, train_policy_loss, val_loss, val_cl_loss, val_policy_loss))
            if (val_acc > best_acc) and (val_f1 > best_f1):
                best_acc = val_acc
                best_f1 = val_f1
                with open(
                        'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch)), 'wb'
                ) as f:
                    torch.save(model.state_dict(), f)
                best_ckp_path = 'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch))

        with torch.no_grad():
            model.eval() 
            policy.eval()
            test_y_true, test_p_true = list(), list()
            test_y_pred, test_p_pred = list(), list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            for step, batch in enumerate(test_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                emb = batch["embedding"].to(DEVICE)
                emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits, hidden = model(input_ids, emb, attention_mask=attention_mask)
                defer_label = [1]
                pred_idx = torch.max(logits, 1)[1]
                if pred_idx[0] == labels[0]: defer_label = [0]
                defer_label = torch.tensor(defer_label, dtype=torch.float32).to(DEVICE)
                defer_logits = policy(emb, hidden.unsqueeze(dim=0),attention_mask=emb_attention_mask)
                p_pred_idx = torch.max(defer_logits, 1)[1]
                test_y_pred += list(pred_idx.data.cpu().numpy())
                test_y_true += list(labels.data.cpu().numpy())
                # policy
                test_p_true += list(defer_label.data.cpu().numpy())
                test_p_pred += list(p_pred_idx.data.cpu().numpy())
            acc = accuracy_score(test_y_true, test_y_pred)
            qwk = cohen_kappa_score(test_y_true, test_y_pred, weights='quadratic')
            f1 = f1_score(test_y_true, test_y_pred, average='macro')
            policy_acc = accuracy_score(test_p_true, test_p_pred)
            policy_f1 = f1_score(test_p_true, test_p_pred, average='macro')
            print("Test acc is {} ".format(acc))
            print("Test qwk is {} ".format(qwk))
            print("Test f1 is {} ".format(f1))
            print("Test defer acc is {} ".format(policy_acc))
            print("Test defer f1 is {} ".format(policy_f1))
            wandb.log({"test Acc": acc,"test F1":f1, "test qwk":qwk, 
                       "test policy Acc": policy_acc,"test policy F1":policy_f1,
                       "Val der Acc": val_defer_acc, "Val defer f1": val_defer_f1,
                       "val deferral rate": val_defer_rate,
                       "Train loss": train_loss, "Val loss": val_loss, 
                       "Train Classification loss": train_cl_loss, "Val Classification loss": val_cl_loss, 
                       "Train Policy loss": train_policy_loss, "Val Policy loss": val_policy_loss,
                       "Train f1": train_f1, "Val f1": val_f1,
                       "Train polciy f1": train_policy_f1, "Val policy f1": val_policy_f1,
                       "Train Acc": train_acc, "Val Acc": val_acc, 
                       "Train polciy Acc": train_policy_acc, "Val policy Acc": val_policy_acc,
                       "Train QWK": train_qwk, "Val QWK": val_qwk})
#     print('Real Test: \n')
#     with torch.no_grad():
#         test_correct_total = 0
#         print("start to test at {} ".format(best_ckp_path))
#         model.load_state_dict(torch.load('./' + best_ckp_path))
#         model.eval()
#         policy.eval()
#         test_y_true, test_p_true = list(), list()
#         test_y_pred, test_p_pred = list(), list()
#         test_iterator = tqdm(test_loader, desc="Test Iteration")
#         for step, batch in enumerate(test_iterator):
#             input_ids = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             emb = batch["embedding"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)
#             logits, hidden = model(input_ids, emb, attention_mask=attention_mask)
#             defer_label = [1]
#             pred_idx = torch.max(logits, 1)[1]
#             if pred_idx[0] == labels[0]: defer_label = [0]
#             defer_label = torch.tensor(defer_label, dtype=torch.float32).to(DEVICE)
#             defer_logits = policy(emb, hidden.unsqueeze(dim=0))
#             p_pred_idx = torch.max(defer_logits, 1)[1]
#             test_y_pred += list(predicted.data.cpu().numpy())
#             test_y_true += list(labels.data.cpu().numpy())
#             # policy
#             test_p_true += list(defer_label.data.cpu().numpy())
#             test_p_pred += list(p_pred_idx.data.cpu().numpy())
#         acc = accuracy_score(test_y_true, test_y_pred)
#         qwk = cohen_kappa_score(test_y_true, test_y_pred, weights='quadratic')
#         f1 = f1_score(test_y_true, test_y_pred, average='macro')
#         policy_acc = accuracy_score(test_p_true, test_p_pred)
#         policy_f1 = f1_score(test_p_true, test_p_pred, average='macro')
#         print("Final Test acc is {} ".format(acc))
#         print("Final Test qwk is {} ".format(qwk))
#         print("Final Test f1 is {} ".format(f1))
#         print("Final Test defer acc is {} ".format(policy_acc))
#         print("Final Test defer f1 is {} ".format(policy_f1))
#         print(classification_report(y_true, y_pred))
#         print(confusion_matrix(y_true, y_pred))
#         wandb.log({"final_test Acc": acc})
#         wandb.log({"final_test QWK": qwk})
#         wandb.log({"final_test f1": f1})
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