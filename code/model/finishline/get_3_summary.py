import argparse
import os
import random
import json
import numpy as np
import torch
import sys
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
all_stopwords |= {"Very", 'The', 'That', 'They', 'I', 'I,m', 'comfortable,', 'They\'re', 'I\'m',}
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

def create_train_data(args,
                      min_reviews=10,
                      min_summary_tokens=20, #30, 60
                      max_summary_tokens=100, #60, 100
                      max_tokens=512,
                      num_keywords=10):
  # obtain data
    data = []
    f = open('/home/jade/untextsum/data/' + args.dataset + '/' + args.data_source, 'r')
    for line in tqdm(f):
        inst = json.loads(line.strip())
        # print(type(inst['review']))
        data.append(inst)
    f.close()
    random.shuffle(data)
    # get model
    model, tokenizer = prepare_model(args)
    model.cuda()
    model.eval()

    os.makedirs('/home/jade/untextsum/data/' + args.dataset + '/' + 'Sentiment_filtered' + str(args.num_aspects) + "_" + args.load_model.split('/')[-1] + '_' + args.load_model.split('/')[-2:][0], exist_ok=True)
    dataset_file = '/home/jade/untextsum/data/' + args.dataset + '/' + 'Sentiment_filtered' + str(args.num_aspects) + "_" + args.load_model.split('/')[-1] + '_' + args.load_model.split('/')[-2:][0] + '/train_sum.jsonl'
    summary_set = set()
    if os.path.exists(dataset_file):
        f = open(dataset_file, 'r')
        for line in f:
            try:
                inst = json.loads(line.strip())
            except:
                continue
            summary_set.add(inst['summary'])
        f.close()

    # count = 0
    total_reviews = [inst['review'] for inst in tqdm(data)]
    for i in range(0, len(total_reviews), 50):
        reviews = total_reviews[i:i+50]
        reviews = [review for review in reviews if len(review) != 0]
        # remove instances with few reviews
        if len(reviews) < min_reviews:
            continue
  # tokenize reviews
        tok_reviews = []
        # print(len(reviews))
        for review in reviews:
            tok_reviews.append([tokenizer.encode(sentence) for sentence in review])
        sentence_switches_list = []
        word_switches_list = []
        document_switches = []
  # run model
        print('running model...')
        for j in range(0, len(tok_reviews), 2):
            tok_reviews_batch = tok_reviews[j:j+2]
            # print(tok_reviews_batch)
            inp_batch = [(review, -1) for review in tok_reviews_batch]
            inp_batch, _ = aspect_detection_collate(inp_batch)
            inp_batch = inp_batch.cuda()
            with torch.no_grad():
                preds = model(inp_batch)
            document_pred = preds['document'].tolist()
            # print(document_pred)
            # for pp in document_pred:
            #     doc_contains_aspect = np.any([x>0 for x in pp])
            #     if doc_contains_aspect:
            #         print('True')
            sentence_pred = preds['sentence'].cpu().detach().numpy()
            # print(sentence_pred)
            word_pred = preds['word'].cpu().detach().numpy()
            # print(word_pred)
            sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
            word_weight = preds['word_weight'].cpu().detach().numpy()
            document_switches += document_pred
            for k, sentences in enumerate(tok_reviews_batch):
                sentence_switches = []
                word_switches = []
                for l, sentence in enumerate(sentences):
                    tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]
                    # print(tokens[0][0])
                    sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                    pred = sentence_pred[k,l]
                    weight = sentence_weight[k,l]
                    sentence_switch = pred * weight
                    sentence_switches.append((sentence, sentence_switch))
                    word_switches_of_sentence = {}
                    # print(type(word_switches_of_sentence))
                    word = ""
                    pred = []
                    weight = []
                    for m, token in enumerate(tokens):
                        if token[0] == '\u0120':
                        # start of a new token; reset values
                            word = word.replace('<s>', '')
                            token = token[1:]
                            pred = np.mean(pred, axis=0)
                            weight = np.mean(weight, axis=0)
                            word_switch = np.maximum(0, pred * weight)
                            # print(type(word_switches_of_sentence))
                            if word not in word_switches_of_sentence:
                                word_switches_of_sentence[word] = 0
                            word_switches_of_sentence[word] += word_switch
                            word = ""
                            pred = []
                            weight = []
                        word += token
                        pred.append(word_pred[k,l,m])
                        weight.append(word_weight[k,l,m])
                    word_switches_of_sentence = [(word, word_switches_of_sentence[word]) for word in word_switches_of_sentence] # list of switches
                    word_switches.append(word_switches_of_sentence) # list of list of switches
                sentence_switches_list.append(sentence_switches)
                word_switches_list.append(word_switches) # list of list of list of switches
  # sample summary and its reviews, keywords
        print('creating data...')
        for s_id, summary in enumerate(reviews):
            sentence_list = []
            for j, sentence in enumerate(summary):
                # print(sentence)
                sentence_switch = sentence_switches_list[s_id][j][1]
                # print(sentence_switch)
                tmp_sentiment_count = [1 for x in sentence_switch[-3:] if x>0]
                contains_sentiment = np.any(len(tmp_sentiment_count)>0)
                if not contains_sentiment:
                    # print('some problem here')
                    continue
                tmp_aspect_count = [1 for x in sentence_switch[:7] if x>0]
                contains_aspect = np.any((len(tmp_sentiment_count) + len(tmp_aspect_count)) == 1)
                if not contains_aspect:
                    # print('damn no aspect')
                    continue
                sentence_list.append(sentence)
                if s_id % 500 == 0:
                    print('Aspect and sentiment related sentence detected')
                # print(len(sentence_list))
            summary = ' '.join(sentence_list)
            if summary in summary_set:
                continue
            # check min max length
            if len(summary.split()) < min_summary_tokens or len(summary.split()) > max_summary_tokens:
                continue
            document_switch = document_switches[s_id]
            # remove sentences of this review
            sentence_switches = []
            #word_switches = word_switches_list[s_id]
            word_switches = [] # list of list of switches
            for j in range(len(sentence_switches_list)):
                if j != s_id:
                    sentence_switches += sentence_switches_list[j]
                    word_switches += word_switches_list[j]
            # get sentences
            sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
            sentence_ids = np.argsort(sentence_scores)
            input_length = 0
            new_reviews = []
            for j, idx in enumerate(sentence_ids):
                if sentence_scores[idx] == 1e9:
                    break
                if input_length > 600:
                    break
                try:
                    sentence = sentence_switches[idx][0]
                except:
                    continue
                input_length += len(sentence.split())
                new_reviews.append(sentence)
            sentence_ids = sentence_ids[:j]
            if len(sentence_ids) == 0: # no related sentences
                continue
                # combine word switches
            word_switches_dict = {}
            for idx in range(len(word_switches)):
                if idx not in sentence_ids:
                    continue
                for word, switch in word_switches[idx]:
                    if word not in word_switches_dict:
                        word_switches_dict[word] = np.zeros(args.num_aspects)
                    word_switches_dict[word] += np.maximum(0, switch)
            word_switches_final = [(word, word_switches_dict[word]) for word in word_switches_dict]
            word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches_final]
            word_switches = [(word_switch, word_score)
              for word_switch, word_score in zip(word_switches_final, word_scores)
              if word_score != 1e9
            ]
            word_switches = sorted(word_switches, key=lambda a: a[-1])[:3*num_keywords]
            # print(word_switches)
            keywords = []
            for w_switch in word_switches:
                if w_switch[0][0].lower() not in keywords and w_switch[0][0].lower() not in all_stopwords and w_switch[0][0].lower() not in nltk_stop_words:
                    keywords.append(w_switch[0][0])
            pair = {}
            pair['summary'] = summary
            pair['reviews'] = new_reviews
            pair['keywords'] = keywords
            pair['switch'] = document_switch
            # print(dataset_file)
            f = open(dataset_file, 'a')
            # pi = {'test':2}
            # f.write(json.dump(pi) + '\n')
            f.write(json.dumps(pair) + '\n')
            print('write something')
            f.close()
            # count += 1
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-num_heads', default=12, type=int)
    parser.add_argument('-dataset', default='finishline', type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    parser.add_argument('-model_name', default='naive', type=str)
    parser.add_argument('-model_dim', default=768, type=int)
    parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/S_7_5_finishline_neg10000/naivestep.10000 average_loss.0.44 sentence_f1.93.31 document_f1.93.34', type=str)
    parser.add_argument('-model_type', default='distilroberta-base', type=str)
    parser.add_argument('-data_source', default='S_7_5_finishline_neg.json', type=str)
    parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)


    args = parser.parse_args()

    if args.mode == 'train':
        create_train_data(args)
    elif args.mode == 'eval-aspect':
        create_aspect_test_data(args)
    elif args.mode == 'eval-general':
        create_general_test_data(args)
