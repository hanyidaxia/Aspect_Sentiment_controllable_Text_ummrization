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


def prepare_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model = MIL(args)
    model.cuda()
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    return model, tokenizer


def create_aspect_test_data(args,
                     num_keywords=10):
  # get model
    model, tokenizer = prepare_model(args)
    model.cuda()
    model.eval()

    # prepare switch map
    switch_map = {}
    with open (args.attribute_lexicon_name, 'r') as f:
        keyword_dic = json.load(f)
    keyword_dic['Pos'] = ''
    keyword_dic['Neu'] = ''
    keyword_dic['Neg'] = ''
    # keyword_dirs = sorted(keyword_dirs)
    for i, key in enumerate(dict(itertools.islice(keyword_dic.items(), 7))):
        for sentiment in ['Pos', 'Neu', 'Neg']:
            keyword = key + '_' + sentiment
            switch = [0] * len(keyword_dic)
            switch[i] = 1
            if sentiment == 'Pos':
                switch[-3] = 1
            elif sentiment == 'Neu':
                switch[-2] = 1
            if sentiment == 'Neg':
                switch[-1] = 1
            switch_map[keyword] = switch
            # print(switch_map)

    for split in ['dev', 'test']:
    # obtain data
        data = []

        f = open(args.data_dir + args.dataset + '/' + 'split' + '/' + split + '_fin' + '.jsonl', 'r')
        for line in tqdm(f):
            inst = json.loads(line.strip())
            data.append(inst)

        f.close()

        f = open(args.data_dir + args.dataset + '/' + 'split' + '/' + split + '.sum.aspect.jsonl', 'w')
        # reviews = []
        for inst in tqdm(data):
            if any(key_word in inst['keywords']
            reviews = inst['reviews']

      # tokenize reviews
            tok_reviews = []
            # for review in reviews:
            tok_reviews.append([tokenizer.encode(sentence.lower()) for sentence in reviews])
            sentence_switches = []
            word_switches = {}

      # run model
            print('running model...')
            for j in range(0, len(tok_reviews), 2):
                tok_reviews_batch = tok_reviews[j:j+2]

                inp_batch = [(review, -1) for review in tok_reviews_batch]
                inp_batch, _ = aspect_detection_collate(inp_batch)
                inp_batch = inp_batch.cuda()

                with torch.no_grad():
                    preds = model(inp_batch)
                sentence_pred = preds['sentence'].cpu().detach().numpy()
                word_pred = preds['word'].cpu().detach().numpy()

                sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
                word_weight = preds['word_weight'].cpu().detach().numpy()

                for k, sentences in enumerate(tok_reviews_batch):
                    for l, sentence in enumerate(sentences):
                        tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]

                        sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                        pred = sentence_pred[k,l]
                        weight = sentence_weight[k,l]
                        sentence_switch = pred * weight
                        sentence_switches.append((sentence, sentence_switch))

                        word = ""
                        pred = []
                        weight = []
                        for m, token in enumerate(tokens):
                            if token[0] == '\u0120':
                                # start of a new token; reset values
                                word = word.replace('<s>', '')
                                token = token[1:]
                                pred = np.max(pred, axis=0)
                                weight = np.max(weight, axis=0)
                                word_switch = pred * weight
                                if word not in word_switches:
                                    word_switches[word] = 0
                                word_switches[word] += word_switch

                                word = ""
                                pred = []
                                weight = []

                            word += token
                            pred.append(word_pred[k,l,m])
                            weight.append(word_weight[k,l,m])

            word_switches = [(word, word_switches[word]) for word in word_switches]

            document_switch = switch_map[inst['keywords'][0]]

            random.shuffle(word_switches)
            random.shuffle(sentence_switches)

            # get keywords
            word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches]
            word_switches = [
            (word_switch, word_score)
            for word_switch, word_score in zip(word_switches, word_scores)
            if word_score != 1e9
            ]
            word_switches = sorted(word_switches, key=lambda a: a[-1])[:2*num_keywords]
            keywords = [word_switch[0][0] for word_switch in word_switches if word_switch[0][0].lower() not in all_stopwords and word_switch[0][0].lower() not in nltk_stop_words]

            # get sentences
            sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
            sentence_switches = [
            (sentence_switch, sentence_score)
            for sentence_switch, sentence_score in zip(sentence_switches, sentence_scores)
            #if sentence_score != 1e9
            ]
            sentence_switches = sorted(sentence_switches, key=lambda a: a[-1])

            input_length = 0
            idx = 0
            new_reviews = []
            for idx in range(len(sentence_switches)):
                if input_length > 600:
                    break
                try:
                    sentence = sentence_switches[idx][0][0]
                except:
                    continue
                input_length += len(sentence.split())
                new_reviews.append(sentence)

            pair = {}
            pair['summary'] = inst['summary']
            pair['reviews'] = new_reviews
            pair['keywords'] = keywords
            pair['switch'] = document_switch
            f.write(json.dumps(pair) + '\n')

        f.close()


def create_general_test_data(args,
                             num_keywords=10):
  # get model
    model, tokenizer = prepare_model(args)
    model.cuda()
    model.eval()

    for split in ['dev', 'test']:
    # obtain data
        data = []

        f = open('data/' + args.dataset + '/general_' + split + '.json', 'r')
        data = json.load(f)
        f.close()

        f = open('data/' + args.dataset + '/' + split + '.sum.general.jsonl', 'w')

        for inst in tqdm(data):
            reviews = inst['reviews']

      # tokenize reviews
            tok_reviews = []
            for review in reviews:
                sentences = [' '.join(word_tokenize(sentence.lower())) for sentence in review['sentences']]
                tok_reviews.append([tokenizer.encode(sentence) for sentence in sentences])

            sentence_switches = []
            word_switches = {}

      # run model
            print('running model...')
            for j in range(0, len(tok_reviews), 2):
                tok_reviews_batch = tok_reviews[j:j+2]

                inp_batch = [(review, -1) for review in tok_reviews_batch]
                inp_batch, _ = aspect_detection_collate(inp_batch)
                inp_batch = inp_batch.cuda()

                with torch.no_grad():
                    preds = model(inp_batch)
                sentence_pred = preds['sentence'].cpu().detach().numpy()
                word_pred = preds['word'].cpu().detach().numpy()

                sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
                word_weight = preds['word_weight'].cpu().detach().numpy()

                for k, sentences in enumerate(tok_reviews_batch):
                    for l, sentence in enumerate(sentences):
                        tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]

                        sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                        pred = sentence_pred[k,l]
                        weight = sentence_weight[k,l]
                        sentence_switch = pred * weight
                        sentence_switches.append((sentence, sentence_switch))

                        word = ""
                        pred = []
                        weight = []
                        for m, token in enumerate(tokens):
                            if token[0] == '\u0120':
                # start of a new token; reset values
                                word = word.replace('<s>', '')
                                token = token[1:]
                                pred = np.max(pred, axis=0)
                                weight = np.max(weight, axis=0)
                                word_switch = pred * weight
                                if word not in word_switches:
                                    word_switches[word] = 0
                                word_switches[word] += word_switch

                                word = ""
                                pred = []
                                weight = []

                            word += token
                            pred.append(word_pred[k,l,m])
                            weight.append(word_weight[k,l,m])

            word_switches = [(word, word_switches[word]) for word in word_switches]

            document_switch = [1] * args.num_aspects

            random.shuffle(word_switches)
            random.shuffle(sentence_switches)

            # get keywords
            word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches]
            word_switches = [
            (word_switch, word_score)
            for word_switch, word_score in zip(word_switches, word_scores)
            if word_score != 1e9
            ]
            word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
            keywords = [word_switch[0][0] for word_switch in word_switches]

      # get sentences
            sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
            sentence_switches = [
            (sentence_switch, sentence_score)
            for sentence_switch, sentence_score in zip(sentence_switches, sentence_scores)
            if sentence_score != 1e9
            ]
            sentence_switches = sorted(sentence_switches, key=lambda a: a[-1])

            input_length = 0
            idx = 0
            new_reviews = []
            for idx in range(len(sentence_switches)):
                if sentence_switches[idx][1] == 1e9:
                    break
                if input_length > 600:
                    break
                try:
                    sentence = sentence_switches[idx][0][0]
                except:
                    continue
                input_length += len(sentence.split())
                new_reviews.append(sentence)

            pair = {}
            pair['summary'] = [x.lower() for x in inst['summaries']['general']]
            pair['reviews'] = new_reviews
            pair['keywords'] = keywords
            pair['switch'] = document_switch

            f.write(json.dumps(pair) + '\n')

        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='eval-aspect', type=str)
    parser.add_argument('-data_dir', default='/home/jade/untextsum/data/', type=str)
    parser.add_argument('-num_heads', default=12, type=int)
    parser.add_argument('-dataset', default='finishline', type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    parser.add_argument('-model_name', default='naive', type=str)
    parser.add_argument('-model_dim', default=768, type=int)
    parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/S_7_5_finishline10000/naivestep.10000 average_loss.0.44 sentence_f1.93.15 document_f1.90.97', type=str)
    parser.add_argument('-model_type', default='distilroberta-base', type=str)
    parser.add_argument('-data_source', default='mil_7_man.json', type=str)
    parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)


    args = parser.parse_args()

    if args.mode == 'eval-aspect':
        create_aspect_test_data(args)
    elif args.mode == 'eval-general':
        create_general_test_data(args)
