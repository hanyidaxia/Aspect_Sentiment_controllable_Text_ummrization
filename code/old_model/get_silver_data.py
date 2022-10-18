import argparse
import os
import random
import json
import numpy as np
import torch
import sys
import itertools
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from mil import MIL
import spacy
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import time
sys.path.append(os.path.realpath('..'))
from data_processor.readers import AspectDetectionDataset, aspect_detection_collate




sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
all_stopwords |= {"Very", 'The', 'That', 'They', 'I'}
nltk_stop_words = set(stopwords.words('english'))


def soft_margin(a, b):
    a = np.maximum(0, a)
    b = np.maximum(0, b)
    return np.log(1 + np.exp(-a * b)).sum()


def soft_margin_list(a_list, b):
    ret = 0
    for a in a_list:
        ret += soft_margin(a, b)
    return ret

def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key


def prepare_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model = MIL(args)
    model.cuda()
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    return model, tokenizer


def create_aspect_test_data(args,num_keywords=10):
    for name in ['dev_fin', 'test_fin']:
        man_dic = []
        with open (args.file_dir + args.dataset + '/split/' + name + '.jsonl',  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        gen_pos = []
        gen_neu = []
        gen_neg = []

        asp = {}
        for inst in man_dic:
            # print(type(inst))
            tmp_count = len([1 for j in inst['switch'][:7] if j > 0])
            if tmp_count > 1:
                if inst['switch'][-3] > 0:
                    gen_pos.append(inst)
                elif inst['switch'][-2] > 0:
                    gen_neu.append(inst)
                elif inst['switch'][-1] > 0:
                    gen_neg.append(inst)
            elif tmp_count == 0:
                if inst['switch'][-3] > 0:
                    gen_pos.append(inst)
                elif inst['switch'][-2] > 0:
                    gen_neu.append(inst)
                elif inst['switch'][-1] > 0:
                    gen_neg.append(inst)
            else:
                idx_list = [i for i, k in enumerate(inst['switch'][:7]) if k > 0]
                # print(idx_list)
                if inst['switch'][-3] > 0:
                    if idx_list[0] and str(idx_list[0]) + 'Pos' not in asp:
                        asp[str(idx_list[0]) + 'Pos'] = [inst]
                    elif idx_list[0] and str(idx_list[0]) in asp:
                        asp[str(idx_list[0]) + 'Pos'].append(inst)
                if inst['switch'][-2] > 0:
                    if idx_list[0] and str(idx_list[0]) + 'Neu' not in asp:
                        asp[str(idx_list[0]) + 'Neu'] = [inst]
                    elif idx_list[0] and str(idx_list[0]) + 'Neu' in asp:
                        asp[str(idx_list[0]) + 'Neu'].append(inst)
                if inst['switch'][-1] > 0:
                    if idx_list[0] and str(idx_list[0]) + 'Neg' not in asp:
                        asp[str(idx_list[0]) + 'Neg'] = [inst]
                    elif idx_list[0] and str(idx_list[0]) + 'Neg' in asp:
                        asp[str(idx_list[0]) + 'Neg'].append(inst)
        gen = {'Pos':gen_pos, 'Neu':gen_neu, 'Neg':gen_neg}

        for idx, data in gen.items():
            with open (args.file_dir + args.dataset + '/split/' + name  + '_' + str(idx) + '_' + args.gen_data_name + '.jsonl' , 'w') as f:
                for i in data:
                    f.write(json.dumps(i) + '\n')
                print('Done')

        for idx, data in asp.items():
            with open (args.file_dir + args.dataset + '/split/' + name + '_' +  str(idx) + '_' + args.asp_data_name + '.jsonl' , 'w') as f:
                for i in data:
                    f.write(json.dumps(i) + '\n')
                print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='eval-aspect', type=str)
    parser.add_argument('-file_dir', default='/home/jade/untextsum/data/', type=str)
    parser.add_argument('-gen_data_name', default='gen', type=str)
    parser.add_argument('-asp_data_name', default='asp', type=str)
    # parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/S_7_5_finishline10000/naivestep.10000 average_loss.0.44 sentence_f1.93.15 document_f1.90.97', type=str)
    parser.add_argument('-dataset', default='finishline', type=str)
    # parser.add_argument('-data_source', default='mil_7_man.json', type=str)
    # parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)


    args = parser.parse_args()

    if args.mode == 'eval-aspect':
        create_aspect_test_data(args)
    elif args.mode == 'eval-general':
        create_general_test_data(args)
