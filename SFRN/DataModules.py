import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from constants import *

class ColSTATDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, device):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.device = device
        self.lable_set = set()
        file_data = []
        for file in dataset_file_path:
            with open(file, encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                file_header = next(csv_reader)
                for row in csv_reader:
                    file_data.append(row)
        for row in file_data:
            data_list = []
            a_id = row[0]
            cat = row[-1]
            q_id = row[2]
            context_text = q_context_dict[q_id]
            data_list.append(context_text)
            q_text = q_text_dict[q_id]
            ans_text = row[1]
            #ref_list = correct_ref_dict[q_id][0:4] + part_ref_dict[q_id][0:4] + in_ref_dict[q_id][0:4]
            ref_list = correct_ref_dict[q_id][0:4]
            data_list.append(q_text)
            for t in ref_list[0:]:
                # add data
                data_list.append(t)
            data_list.append(ans_text)
            data = []
            self.lable_set.add(cat)
            data.append(cat)
            data.append(data_list)
            self.data_dict.append(data)
        self.tokenizer = tokenizer
        #self.tag2id = self.set2id(self.lable_set)
        self.tag2id = {'0': 0, '1': 1, '2': 2}
        print(self.tag2id)
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        label, lines = self.data_dict[index]
        label = self.tag2id[label]
        input_ids, attention_masks, seq_positions = [], [], []
        # organize the lines
        ct = lines[0] # context
        q = lines[1]
        anws = lines[2:]
        # Tokenize context and question
        #tokens_ct = tokenizer(line[0], return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True)
        tokens_ct = self.tokenizer(ct, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=128)
        tokens_q = self.tokenizer(q, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=64)
#         print('tokens_ct: ', tokens_ct)
#         print('tokens_q: ', tokens_q)
        # Concatenate tokenized context and question
        cq_input_ids = torch.cat([tokens_ct['input_ids'], tokens_q['input_ids'][:, 1:]], dim=-1)
        cq_attention_mask = torch.cat([tokens_ct['attention_mask'], tokens_q['attention_mask'][:, 1:]], dim=-1)
#         print('cq_input_ids: ', cq_input_ids)
#         print('cq_attention_mask: ', cq_attention_mask)
        # Tokenize and concatenate each answer
        for answer in anws:
            tokens_ans = self.tokenizer(answer, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=param['max_length'])
            answ_input_ids = torch.cat([cq_input_ids, tokens_ans['input_ids'][:, 1:]], dim=-1).squeeze(0)
            answ_attention_mask = torch.cat([cq_attention_mask, tokens_ans['attention_mask'][:, 1:]], dim=-1).squeeze(0)
            #print('answ_input_ids: ', answ_input_ids)
            sep_p = (answ_input_ids == self.tokenizer.sep_token_id).nonzero().squeeze()
            input_ids.append(answ_input_ids)
            attention_masks.append(answ_attention_mask)
            seq_positions.append(sep_p)
        return {
            "input_ids":  torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "seq_pos": torch.stack(seq_positions, dim=0),
            "label": label,
        }


    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id