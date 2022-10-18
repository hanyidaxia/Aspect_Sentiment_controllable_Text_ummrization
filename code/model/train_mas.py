import argparse
import copy
import math
import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from mas import MAS
import random
sys.path.append(os.path.realpath('..'))
from data_processor.readers import AspectDetectionDataset, aspect_detection_collate
"""
entering the training file, this model aim to get the multiple instance result dataset which could entended to further training
"""


def _update_counts(gold, pred, counts):
    if gold * pred > 0:
        counts[0] += 1
    elif gold == 0 and pred == 0:
        counts[0] += 1
    else:
        counts[1] += 1
    # elif gold < 0 and pred > 0:
    #     counts[1] += 1
    # elif gold > 0 and pred < 0:
    #     counts[1] += 1


def evaluate(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.dev_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
    model = MAS(args)
    model.cuda()
    with open (args.attribute_lexicon_name, 'r') as f:
        lexicon = json.load(f)
    aspects = list(lexicon.keys())
    assert args.load_model is not None
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    aspect_word_list = {aspect:[] for aspect in aspects}
    aspect_sentence_list = {aspect:[] for aspect in aspects}

    for inp_batch, _ in dataloader:
        model.eval()
        inp_batch = inp_batch.cuda() # b, s, t
        preds = model(inp_batch)
        document_pred = preds['document']
        sentence_pred = preds['sentence']
        word_pred = preds['word']
        sentence_weight = preds['sentence_weight']
        word_weight = preds['word_weight']
        inp_batch = inp_batch.tolist()
        for i, sentences in enumerate(inp_batch):
            for j, sentence in enumerate(sentences):
                tokens = tokenizer.convert_ids_to_tokens(sentence)
                sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                pred = sentence_pred[i,j].tolist()
                weight = sentence_weight[i,j].tolist()
                for asp_id in range(len(pred)):
                    aspect = aspects[asp_id]
                    aspect_sentence_list[aspect].append((sentence, pred[asp_id]*weight))
                word = ""
                pred = []
                weight = []
                for k, token in enumerate(tokens):
                    if token[0] == '\u0120':
            # start of a new token; reset values
                        word = word.replace('<s>', '')
                        token = token[1:]
                        pred = np.max(pred, axis=0)
                        weight = np.max(weight, axis=0)
                        for asp_id in range(len(pred)):
                            aspect = aspects[asp_id]
                            aspect_word_list[aspect].append((word, pred[asp_id]*weight))
                        pred = [aspects[idx] for idx in range(len(pred)) if pred[idx] > 0]
                        if len(pred) == 0:
                            pred = 'none'
                        else:
                            pred = ' '.join(pred)

            # print(word + '\t' + pred)
                        word = ""
                        pred = []
                        weight = []

                    word += token
                    pred.append(word_pred[i,j,k].tolist())
                    weight.append(word_weight[i,j,k])
        # print()

    print("-------------SENTENCES-------------")
    for aspect in aspect_sentence_list:
        sentence_list = aspect_sentence_list[aspect]
        print(aspect)
        print('------------------------------')
        sentence_list.sort(key=lambda a: -a[-1])
        print_count = 20
        for sentence in sentence_list:
            if print_count == 0:
                break
            print(sentence)
            print_count -= 1
        print('------------------------------')
        print('/n')

    print("--------------WORDS---------------")
    for aspect in aspect_word_list:
        word_list = aspect_word_list[aspect]

        combined_word_list = {}
        for word in word_list:
            if word[0] not in combined_word_list:
                combined_word_list[word[0]] = 0
            combined_word_list[word[0]] += max(0, word[1])

        word_list = [(word, combined_word_list[word]) for word in combined_word_list]

        print(aspect)
        print('------------------------------')
        word_list.sort(key=lambda a: -a[-1])
        print_count = 50
        word_set = set()
        for word in word_list:
            if print_count == 0:
                break
            if word[0] in word_set:
                continue
            word_set.add(word[0])
            print(word)
            print_count -= 1
        print('-----------------------------')
        print('/n')


def train(args):
    print('train the mas model you have following parameters:', args)
    print('Preparing data...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
    print(args.data_dir + '/' + args.dataset + '/' + args.train_file)
    dev_dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.dev_file, tokenizer, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
    print('Preparing model...')
    model = MAS(args)
    model.cuda()
  #optimizer = torch.optim.Adam(model.parameters())
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.no_warmup_steps, args.no_train_steps)
    step = 0
    if args.load_model is not None:
        print('Loading model...')
        best_point = torch.load(args.load_model)
        model.load_state_dict(best_point['model'])
        optimizer.load_state_dict(best_point['optimizer'])
        scheduler.load_state_dict(best_point['scheduler'])
        step = best_point['step']

    print('Start training...')
    while step < args.no_train_steps:
        losses = []
        for inp_batch, out_batch in tqdm(dataloader):
            model.train()
            inp_batch = inp_batch.cuda()
            out_batch = out_batch.cuda().float()
            preds = model(inp_batch, out_batch, step=step)
            document_pred = preds['document']
            sentence_pred = preds['sentence']
            loss = preds['loss']
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.check_every == 0:
                print('Step SSSTTTTTTEEEEEEEEEEEEEEEEEEEPPPPPPPPPPPPPPPPPPPPPPPPP %d Train LossSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS %.4f' % (step, np.mean(losses)))
                doc_counts = [[0] * 2] * args.num_aspects
                sent_counts = [[0] * 2] * args.num_aspects
                dev_loss = []
                for inp_batch, out_batch in tqdm(dev_dataloader):
                    model.eval()
                    inp_batch = inp_batch.cuda()
                    out_batch = out_batch.cuda().float()
                    # print(out_batch)
                    preds = model(inp_batch, out_batch)
                    document_pred = preds['document']
                    if step % 100 == 0:
                        for pp in document_pred.tolist():
                            contains_aspect = np.any(len([1 for x in pp if x>0])>1)
                            # print(pp)
                            if not contains_aspect:
                                print('no aspects detected')
                    sentence_pred = preds['sentence']
                    for bid in range(len(out_batch)):
                        for aid in range(args.num_aspects):
                            _update_counts(out_batch[bid][aid], document_pred[bid][aid], doc_counts[aid])
                            for sid in range(len(sentence_pred[bid])):
                                _update_counts(out_batch[bid][aid], sentence_pred[bid][sid][aid], sent_counts[aid])
                    loss = preds['loss']
                    dev_loss.append(loss.item())
                print('Dev Loss %.4f' % np.mean(dev_loss))
                doc_f1 = []
                sent_f1 = []
                for aid in range(args.num_aspects):
                    doc_f1.append(2*doc_counts[aid][0] / float(2*doc_counts[aid][0] + doc_counts[aid][1]))
                    sent_f1.append(2*sent_counts[aid][0] / float(2*sent_counts[aid][0] + sent_counts[aid][1]))
                doc_f1 = np.mean(doc_f1) * 100
                sent_f1 = np.mean(sent_f1) * 100
                print('Document F1 %.4f and Sentence F1 %.4f' % (doc_f1, sent_f1))
                rand_idx = random.randint(0, args.batch_size-44)
                inp = inp_batch[rand_idx]
                print('Document prediction is %s and the Gold label was %s'% (str(document_pred[rand_idx].tolist()), str(out_batch[rand_idx].tolist())))
                for sid, sentence in enumerate(inp):
                    sentence = tokenizer.decode(sentence, skip_special_tokens=True)
                    if len(sentence.strip()) == 0:
                        continue
                    print('Sentence', sid, ':', sentence)
                    print(sentence_pred[0][sid].tolist())


            if step % args.ckpt_every == 0:
                print('Saving...')
                os.makedirs(args.model_dir + '/' + args.dataset + '/' + args.train_file.split('.')[0] + args.model_name + str(args.no_train_steps), exist_ok=True)
                torch.save({
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'step': step,
                  'loss': np.mean(dev_loss)
                }, args.model_dir + '/' + args.dataset + '/' + args.train_file.split('.')[0] + args.model_name + str(args.no_train_steps) + '/' \
                + args.model_name + 'step.%d average_loss.%.2f sentence_f1.%.2f document_f1.%.2f' % (step, np.mean(losses), doc_f1, sent_f1))
                losses = []

            if step == args.no_train_steps:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-dataset', default='all', type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    # parser.add_argument('-model_name', default='SoftmarginLoss', type=str)
    parser.add_argument('-model_name', default='BCE_Sum_weight', type=str)
    parser.add_argument('-load_model', default=None, type=str)
    parser.add_argument('-train_file', default='best_data.jsonl', type=str)
    parser.add_argument('-dev_file', default='dev_best.jsonl', type=str)
    parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
    parser.add_argument('-model_dir', default='/home/jade/untextsum/model', type=str)
    parser.add_argument('-model_type', default='distilroberta-base', type=str)
    parser.add_argument('-model_dim', default=768, type=int)
    parser.add_argument('-num_heads', default=12, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-learning_rate', default=1e-6, type=float)
    parser.add_argument('-no_train_steps', default=16000, type=int)
    parser.add_argument('-no_warmup_steps', default=8000, type=int)
    parser.add_argument('-check_every', default=100, type=int)
    parser.add_argument('-ckpt_every', default=1000, type=int)
    parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)
