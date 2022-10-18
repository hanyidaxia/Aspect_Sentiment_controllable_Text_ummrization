from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
# from mil import MIL
import argparse
import os
import math
import numpy as np
import spacy
import torch
import sys
# sys.path.append(os.path.realpath('..'))
# from data_processor.readers import AspectDetectionDataset, aspect_detection_collate
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import random
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
# from mil import MIL
from mil import MIL
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
# from calculate_rouge import calculate
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score
tmp_score = scorer('fuck you abby', 'fuck you too, joshua')['rougeL'][-1]
print(tmp_score)

sys.path.append(os.path.realpath('..'))
from data_processor.readers import AspectDetectionDataset, aspect_detection_collate, SummarizationDataset


def calculate(list_a, list_b):
    rouge_score = []
    for i in range(len(list_a)):
        rouge_score.append(scorer(list_a[i], list_b[i])['rougeL'][-1])
    return rouge_score


def train(args):
    print(args)
    print('Preparing data...')
    if args.train_file is None:
        train_file = args.data_dir + '/' + args.dataset + '/' + args.model_dir + '/train_sum.jsonl'
    else:
        train_file = args.train_file
    dataset = SummarizationDataset(train_file, use_keywords=args.use_keywords, use_switch=args.use_switch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print('load train file done')

    if args.asp_dev_file is None:
        asp_dev_file = args.data_dir + '/' + args.dataset + '/dev.sum.aspect.jsonl' % args.seed_type
    else:
        asp_dev_file = args.asp_dev_file

    print('Reading the aspect summary dataset...')
    asp_dev_dataset = SummarizationDataset(asp_dev_file, use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
    asp_dev_dataloader = DataLoader(asp_dev_dataset, batch_size=args.batch_size)
    f = open(asp_dev_file, 'r')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    f.close()
    asp_gold_sums = [inst['summary'].lower() for inst in data]
    print('Read done')

    if args.gen_dev_file is None:
        gen_dev_file = args.data_dir + '/' + args.dataset + '/dev.sum.general.jsonl' % args.seed_type
        print(gen_dev_file)
    else:
        gen_dev_file = args.gen_dev_file

    gen_dev_dataset = SummarizationDataset(gen_dev_file, use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
    gen_dev_dataloader = DataLoader(gen_dev_dataset, batch_size=args.batch_size)
    f = open(gen_dev_file, 'r')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    f.close()
    gen_gold_sums = [inst['summary'].lower() for inst in data] ##############
    print('load general validation file done')
    print('Initializing model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
    if args.use_switch != 'none':
        special_tokens.append('<Pos>')
        special_tokens.append('<Neu>')
        special_tokens.append('<Neg>')
        for i in range(args.num_aspects - 3):
            special_tokens.append('<asp_%d>' % i)
    print('current special token set is:', special_tokens)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    pad_id = tokenizer.pad_token_id
    model = Model.from_pretrained(args.model_type, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.no_warmup_steps, args.no_train_steps)
    step = 0
    best_rouge = 0
    rng = np.random.default_rng()
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
        for _, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
            # print(inp_batch, 'the out atch is isisiisisisiisisiisiisisAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', out_batch, switch_batch)
            model.train()
            batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch), tgt_texts=list(out_batch), max_length=args.max_length, max_target_length=args.max_target_length,
            padding=True, truncation=True, return_tensors='pt')
            inp_ids = batch_encoding['input_ids'].cuda()
            inp_mask = batch_encoding['attention_mask'].cuda()
            out_ids = batch_encoding['labels'].cuda()
            out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
            out_ids[out_ids==0] = -100
            dec_inp_ids = model._shift_right(out_ids)
            model_outputs = model(input_ids=inp_ids, attention_mask=inp_mask, decoder_input_ids=dec_inp_ids, labels=out_ids, output_hidden_states=True)
            loss = model_outputs.loss
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.check_every == 0:
                print('Step %d Loss %.4f' % (step, np.mean(losses)))
                model.eval()
                rouge_scores = []

        # generate general summaries
                gen_pred_sums = []
                for _, (inp_batch, out_batch, _) in enumerate(tqdm(gen_dev_dataloader)):
                    batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch), tgt_texts=list(out_batch), max_length=args.max_length,
                    max_target_length=args.max_target_length, padding=True, truncation=True, return_tensors='pt')
                    inp_ids = batch_encoding['input_ids'].cuda()

                    preds = model.generate(inp_ids, min_length=60, max_length=args.max_target_length*2, num_beams=2, no_repeat_ngram_size=2, decoder_start_token_id=0,
                    repetition_penalty=1, length_penalty=1)

                    for pred in preds:
                        gen_pred_sums.append(tokenizer.decode(pred))
                # print('the general gold sums was GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG', gen_gold_sums, len(gen_gold_sums))
                # print('the general gold sums got from model was GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG', gen_pred_sums, len(gen_pred_sums))

                gen_scores = calculate(gen_gold_sums, gen_pred_sums)
                rouge_scores += list(gen_scores)

                asp_pred_sums = []
                for _, (inp_batch, out_batch, _) in enumerate(tqdm(asp_dev_dataloader)):
                    model.eval()
                    batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch), tgt_texts=list(out_batch), max_length=args.max_length,
                    max_target_length=args.max_target_length, padding=True, truncation=True, return_tensors='pt')

                    inp_ids = batch_encoding['input_ids'].cuda()
                    preds = model.generate(inp_ids, min_length=10, max_length=args.max_target_length*2, num_beams=2, no_repeat_ngram_size=2, decoder_start_token_id=0,
                    repetition_penalty=1, length_penalty=1)

                    for pred in preds:
                        asp_pred_sums.append(tokenizer.decode(pred))

                asp_scores = calculate(asp_gold_sums, asp_pred_sums)
                rouge_scores += list(asp_scores)
                print(rouge_scores)
                rouge = np.power(np.product(rouge_scores), 1.0/len(rouge_scores))

                print("ROUGE: %.4f" % rouge)
                if args.dataset == 'finishline':
                    print("General Gold:", gen_gold_sums[0])
                    print("General Pred:", gen_pred_sums[0])
                print("Aspect Gold:", asp_gold_sums[0])
                print("Aspect Pred:", asp_pred_sums[0])

                if rouge > best_rouge:
                    print('Saving...')
                    best_rouge = rouge
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step, 'loss': np.mean(losses)
                    }, args.model_dir + '/' + args.dataset + '/' + args.model_name + '.best.%d.%.3f' % (step, rouge))

            if step % args.ckpt_every == 0:
                print('Saving...')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step, 'loss': np.mean(losses)
                }, args.model_dir + '/' + args.dataset + '/' + args.model_name + '.%d.%.2f' % (step, np.mean(losses)))
                losses = []

            if step == args.no_train_steps:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-dataset', default='finishline', type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    parser.add_argument('-model_name', default='naive', type=str)
    parser.add_argument('-load_model', default=None, type=str)
    parser.add_argument('-seed_type', default='my', type=str)
    parser.add_argument('-train_file', default=None, type=str)
    parser.add_argument('-asp_dev_file', default='/home/jade/untextsum/data/finishline/split/overall/dev_fin_asp.jsonl', type=str)
    parser.add_argument('-gen_dev_file', default='/home/jade/untextsum/data/finishline/split/overall/dev_fin_gen.jsonl', type=str)
    parser.add_argument('-asp_test_file', default='/home/jade/untextsum/data/finishline/split/overall/test_fin_asp.jsonl', type=str)
    parser.add_argument('-gen_test_file', default='/home/jade/untextsum/data/finishline/split/overall/test_fin_gen.jsonl', type=str)
    parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
    parser.add_argument('-model_dir', default='Sentiment_filtered10_naivestep.10000 average_loss.0.44 sentence_f1.93.31 document_f1.93.34_S_7_5_finishline_neg10000', type=str)
    parser.add_argument('-model_type', default='t5-small', type=str)
    parser.add_argument('-model_dim', default=512, type=int)
    parser.add_argument('-use_keywords', default='input', type=str) # none, input, output
    parser.add_argument('-use_switch', default='input', type=str) # none, input, output
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-learning_rate', default=1e-6, type=float)
    parser.add_argument('-no_train_steps', default=1000, type=int)
    parser.add_argument('-no_warmup_steps', default=500, type=int)
    parser.add_argument('-check_every', default=100, type=int)
    parser.add_argument('-ckpt_every', default=1000, type=int)
    parser.add_argument('-max_length', default=512, type=int)
    parser.add_argument('-min_target_length', default=15, type=int)
    parser.add_argument('-max_target_length', default=128, type=int)
    parser.add_argument('-num_beams', default=2, type=int)
    parser.add_argument('-no_repeat_ngram_size', default=3, type=int)
    parser.add_argument('-repetition_penalty', default=1, type=float)
    parser.add_argument('-length_penalty', default=1, type=float)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval-general':
        evaluate(args, 'general')
    elif args.mode == 'eval-aspect':
        evaluate(args, 'aspect')
    elif args.mode == 'eval-multi':
        evaluate(args, 'multi')
    elif args.mode == 'eval-double':
        evaluate(args, 'double')













































































# m = nn.Sigmoid()
# input = torch.randn(2,3)
# print(input)
# output = m(input)
# print(output)
#
#
# #
# m = nn.Sigmoid()
# loss = nn.BCELoss(reduction = 'sum')
# input = torch.randn(3, 5, requires_grad=True)
# print('size of the input and actual input is', input.size(), input)
# target = torch.empty(3, 5).random_(2)
# print('size of the target and actual target is', target.size(), target)
# output = loss(m(input), target)
# print(output)
# output.backward()











# Test the sentiment infused model
# def _update_counts(gold, pred, counts):
#     if gold * pred > 0:
#         counts[0] += 1
#     else:
#         counts[1] += 1
#
#
# def train(args):
#     print('train the mil model you have following parameters:', args)
#     print('Preparing data...')
#
#     tokenizer = AutoTokenizer.from_pretrained(args.model_type)
#     dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.train_file, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
#     print(args.data_dir + '/' + args.dataset + '/' + args.train_file)
#     print(type(dataloader))
#     dev_dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.dev_file, tokenizer, shuffle=False)
#     dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
#     print('Initializing model...')
#
#     model = MIL(args)
#     model.cuda()
#
#   #optimizer = torch.optim.Adam(model.parameters())
#     optimizer = AdamW(model.parameters(), lr=args.learning_rate)
#     scheduler = get_cosine_schedule_with_warmup(optimizer, args.no_warmup_steps, args.no_train_steps)
#
#     step = 0
#     # rng = np.random.default_rng()
#     if args.load_model is not None:
#         print('Loading model...')
#         best_point = torch.load(args.load_model)
#         model.load_state_dict(best_point['model'])
#         optimizer.load_state_dict(best_point['optimizer'])
#         scheduler.load_state_dict(best_point['scheduler'])
#         step = best_point['step']
#
#     print('Start training...')
#     while step < args.no_train_steps:
#         losses = []
#         for _, (inp_batch, out_batch) in enumerate(tqdm(list(dataloader))):
#             model.train()
#             inp_batch = inp_batch.cuda()
#             out_batch = out_batch.cuda().float()
#             # print(out_batch)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#             preds = model(inp_batch, out_batch, step=step)
#             document_pred = preds['document']
#             sentence_pred = preds['sentence']
#             loss = preds['loss']
#             losses.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             step += 1
#             if step % args.check_every == 0:
#                 print('Step SSSTTTTTTEEEEEEEEEEEEEEEEEEEPPPPPPPPPPPPPPPPPPPPPPPPP %d Train LossSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS %.4f' % (step, np.mean(losses)))
#
#                 doc_counts = [[0] * 2] * args.num_aspects
#                 sent_counts = [[0] * 2] * args.num_aspects
#
#                 dev_loss = []
#                 for _, (inp_batch, out_batch) in enumerate(tqdm(dev_dataloader)):
#                     model.eval()
#
#                     inp_batch = inp_batch.cuda()
#                     out_batch = out_batch.cuda().float()
#                     # print(out_batch)
#
#                     preds = model(inp_batch, out_batch)
#                     document_pred = preds['document']
#                     sentence_pred = preds['sentence']
#
#                     for bid in range(len(out_batch)):
#                         for aid in range(args.num_aspects):
#                             _update_counts(out_batch[bid][aid], document_pred[bid][aid], doc_counts[aid])
#
#                             for sid in range(len(sentence_pred[bid])):
#                                 _update_counts(out_batch[bid][aid], sentence_pred[bid][sid][aid], sent_counts[aid])
#
#
#                     loss = preds['loss']
#                     dev_loss.append(loss.item())
#
#                 print('Dev Loss %.4f' % np.mean(dev_loss))
#
#                 doc_f1 = []
#                 sent_f1 = []
#                 for aid in range(args.num_aspects):
#                     doc_f1.append(2*doc_counts[aid][0] / float(2*doc_counts[aid][0] + doc_counts[aid][1]))
#                     sent_f1.append(2*sent_counts[aid][0] / float(2*sent_counts[aid][0] + sent_counts[aid][1]))
#                 doc_f1 = np.mean(doc_f1) * 100
#                 sent_f1 = np.mean(sent_f1) * 100
#
#                 print('Document F1 %.4f' % doc_f1)
#                 print('Sentence F1 %.4f' % sent_f1)
#
#                 inp = inp_batch[0]
#                 print('Document prediction', document_pred[0].tolist())
#                 print('Gold', out_batch[0].tolist())
#                 print()
#                 for sid, sentence in enumerate(inp):
#                     sentence = tokenizer.decode(sentence, skip_special_tokens=True)
#                     if len(sentence.strip()) == 0:
#                         continue
#                     print('Sentence', sid, ':', sentence)
#                     print(sentence_pred[0][sid].tolist())
#                 print('\n')
#
#
#             if step % args.ckpt_every == 0:
#                 print('Saving...')
#                 os.makedirs(args.model_dir + '/' + args.dataset + '/' + args.train_file.split('.')[0] + str(args.no_train_steps), exist_ok=True)
#                 torch.save({
#                   'model': model.state_dict(),
#                   'optimizer': optimizer.state_dict(),
#                   'scheduler': scheduler.state_dict(),
#                   'step': step,
#                   'loss': np.mean(dev_loss)
#                 }, args.model_dir + '/' + args.dataset + '/' + args.train_file.split('.')[0] + str(args.no_train_steps) + '/' + args.model_name + 'JUST_TEST**************' + '.%d.%.2f.%.2f.%.2f' % (step, np.mean(losses), doc_f1, sent_f1))
#                 losses = []
#
#             if step == args.no_train_steps:
#                 break
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-mode', default='train', type=str)
#     parser.add_argument('-dataset', default='finishline', type=str)
#     parser.add_argument('-num_aspects', default=10, type=int)
#     parser.add_argument('-model_name', default='naive', type=str)
#     parser.add_argument('-load_model', default=None, type=str)
#     parser.add_argument('-train_file', default='S_7_5_finishline.json', type=str)
#     parser.add_argument('-dev_file', default='dev_S_7_5_finishline.json', type=str)
#     parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
#     parser.add_argument('-model_dir', default='/home/jade/untextsum/model', type=str)
#     parser.add_argument('-model_type', default='distilroberta-base', type=str)
#     parser.add_argument('-model_dim', default=768, type=int)
#     parser.add_argument('-num_heads', default=12, type=int)
#     # parser.add_argument('-vocab_size', default=50265, type=int)
#     parser.add_argument('-batch_size', default=64, type=int)
#     parser.add_argument('-learning_rate', default=1e-4, type=float)
#     parser.add_argument('-no_train_steps', default=20, type=int)
#     parser.add_argument('-no_warmup_steps', default=5, type=int)
#     parser.add_argument('-check_every', default=100, type=int)
#     parser.add_argument('-ckpt_every', default=1000, type=int)
#     parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)
#
#     args = parser.parse_args()
#     if args.mode == 'train':
#         train(args)
#     else:
#         evaluate(args)







# keyword_dirs = os.listdir('/home/jade/untextsum/model/finishline/')
# print(keyword_dirs)

# Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, 3, requires_grad=True)
# print('The input of the test is', input.size(), input)
# print(input.max(dim=-2))
# print(input.max(dim=-2)[0])
#
# print(input.max(dim=-1))
# print(input.max(dim=-1)[0])
#
#
#
# target = torch.empty(3, 3, dtype=torch.long).random_(5)
# print('The target of the test is', target.size(), target)
# output = loss(input, target)
# output.backward()
#
#
#
#
#
#
#
#
# print('..........................................................................................')
# # Example of target with class probabilities
# input = torch.randn(3, 5, 3, requires_grad=True)
# print('The prob input of the test is', input.size(), input)
# target = torch.randn(3, 3).softmax(dim=1)
# print('The prob target of the test is', target.size(), target)
# output = loss(input, target)
# output.backward()


# f = None
# for i in range (5):
#     with open ('/home/jade/untextsum/model/finishline/man.json',  'w') as f:
#         if i > 2:
#             break
# print(f.closed)


# def loss_test():
#     print(np.log(1 + math.exp(-1)))
#     print(np.log(1 + math.exp(1)))
#     output = Variable(torch.FloatTensor([0,0,0,1])).view(1, -1)
#     print(output)
#     target = Variable(torch.LongTensor([3]))
#     print(target)
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
#     print(loss)
#
#
#
#
# #












# sp = spacy.load('en_core_web_sm')
# all_stopwords = sp.Defaults.stop_words
#
# def soft_margin(a, b):
#     a = np.maximum(0, a)
#     b = np.maximum(0, b)
#     return np.log(1 + np.exp(-a * b)).sum()
#
#
# def soft_margin_list(a_list, b):
#     ret = 0
#     for a in a_list:
#         ret += soft_margin(a, b)
#     return ret
#
#
# def prepare_model(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_type)
#     model = MIL(args)
#     model.cuda()
#     best_point = torch.load(args.load_model)
#     model.load_state_dict(best_point['model'])
#     return model, tokenizer
# # #
# #
# # def create_aspect_test_data(args,
# #                      num_keywords=10):
# #   # get model
# #     model, tokenizer = prepare_model(args)
# #     model.cuda()
# #     model.eval()
# #
# #     # prepare switch map
# #     switch_map = {}
# #     with open (args.attribute_lexicon_name, 'r') as f:
# #         keyword_dic = json.load(f)
# #     # keyword_dirs = sorted(keyword_dirs)
# #     for i, key in enumerate(keyword_dic):
# #         # f = open('data/' + args.dataset + '/keywords/' + file, 'r')
# #         keyword = key
# #         # f.close()
# #         switch = [0] * len(keyword_dic)
# #         switch[i] = 1
# #         switch_map[keyword] = switch
# #     print(switch_map)
# #     print(switch_map)
# #     for split in ['dev', 'test']:
# #     # obtain data
# #         data = []
# #
# #         f = open('data/' + args.dataset + '/' + split + '.jsonl', 'r')
# #         for line in tqdm(f):
# #             inst = json.loads(line.strip())
# #             data.append(inst)
# #
# #         f.close()
# #
# #         f = open('data/' + args.dataset + '/' + split + '.sum.aspect.jsonl', 'w')
# #
# #         for inst in tqdm(data):
# #             reviews = inst['reviews']
# #
# #       # tokenize reviews
# #             tok_reviews = []
# #             for review in reviews:
# #                 tok_reviews.append([tokenizer.encode(sentence.lower()) for sentence in review['sentences']])
# #
# #             sentence_switches = []
# #             word_switches = {}
# #
# #       # run model
# #             print('running model...')
# #             for j in range(0, len(tok_reviews), 2):
# #                 tok_reviews_batch = tok_reviews[j:j+2]
# #
# #                 inp_batch = [(review, -1) for review in tok_reviews_batch]
# #                 inp_batch, _ = aspect_detection_collate(inp_batch)
# #                 inp_batch = inp_batch.cuda()
# #
# #                 with torch.no_grad():
# #                     preds = model(inp_batch)
# #                 sentence_pred = preds['sentence'].cpu().detach().numpy()
# #                 word_pred = preds['word'].cpu().detach().numpy()
# #
# #                 sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
# #                 word_weight = preds['word_weight'].cpu().detach().numpy()
# #
# #                 for k, sentences in enumerate(tok_reviews_batch):
# #                     for l, sentence in enumerate(sentences):
# #                         tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]
# #
# #                         sentence = tokenizer.decode(sentence, skip_special_tokens=True)
# #                         pred = sentence_pred[k,l]
# #                         weight = sentence_weight[k,l]
# #                         sentence_switch = pred * weight
# #                         sentence_switches.append((sentence, sentence_switch))
# #
# #                         word = ""
# #                         pred = []
# #                         weight = []
# #                         for m, token in enumerate(tokens):
# #                             if token[0] == '\u0120':
# #                                 # start of a new token; reset values
# #                                 word = word.replace('<s>', '')
# #                                 token = token[1:]
# #                                 pred = np.max(pred, axis=0)
# #                                 weight = np.max(weight, axis=0)
# #                                 word_switch = pred * weight
# #                                 if word not in word_switches:
# #                                     word_switches[word] = 0
# #                                 word_switches[word] += word_switch
# #
# #                                 word = ""
# #                                 pred = []
# #                                 weight = []
# #
# #                             word += token
# #                             pred.append(word_pred[k,l,m])
# #                             weight.append(word_weight[k,l,m])
# #
# #             word_switches = [(word, word_switches[word]) for word in word_switches]
# #
# #             document_switch = switch_map[inst['keywords'][0]]
# #
# #             random.shuffle(word_switches)
# #             random.shuffle(sentence_switches)
# #
# #             # get keywords
# #             word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches]
# #             word_switches = [
# #             (word_switch, word_score)
# #             for word_switch, word_score in zip(word_switches, word_scores)
# #             if word_score != 1e9
# #             ]
# #             word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
# #             keywords = [word_switch[0][0] for word_switch in word_switches]
# #
# #             # get sentences
# #             sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
# #             sentence_switches = [
# #             (sentence_switch, sentence_score)
# #             for sentence_switch, sentence_score in zip(sentence_switches, sentence_scores)
# #             #if sentence_score != 1e9
# #             ]
# #             sentence_switches = sorted(sentence_switches, key=lambda a: a[-1])
# #
# #             input_length = 0
# #             idx = 0
# #             new_reviews = []
# #             for idx in range(len(sentence_switches)):
# #                 if input_length > 600:
# #                     break
# #                 try:
# #                     sentence = sentence_switches[idx][0][0]
# #                 except:
# #                     continue
# #                 input_length += len(sentence.split())
# #                 new_reviews.append(sentence)
# #
# #             pair = {}
# #             pair['summary'] = inst['summary']
# #             pair['reviews'] = new_reviews
# #             pair['keywords'] = keywords
# #             pair['switch'] = document_switch
# #             f.write(json.dumps(pair) + '\n')
# #
# #         f.close()
# # #
# # #
# def create_train_data(args,
#                       min_token_frequency=5,
#                       min_reviews=3,
#                       min_summary_tokens=3, #30, 60
#                       max_summary_tokens=100, #60, 100
#                       max_tokens=512,
#                       num_keywords=10):
#   # obtain data
#     data = []
#     f = open('/home/jade/untextsum/data/' + args.dataset + '/' + args.data_source, 'r')
#     for line in tqdm(f):
#         inst = json.loads(line.strip())
#         # print(type(inst['review']))
#         data.append(inst)
#     f.close()
#     random.shuffle(data)
#     # get model
#     model, tokenizer = prepare_model(args)
#     model.cuda()
#     model.eval()
#
#     os.makedirs('/home/jade/untextsum/data/' + args.dataset + '/' + str(args.num_aspects) + args.load_model.split('/')[-1] + '_' + args.load_model.split('/')[-2:][0], exist_ok=True)
#     dataset_file = '/home/jade/untextsum/data/' + args.dataset + '/' + str(args.num_aspects) + args.load_model.split('/')[-1] + '_' + args.load_model.split('/')[-2:][0] + '/train_sum.jsonl'
#
#
#     # count = 0
#     total_reviews = [inst['review'] for inst in tqdm(data)]
#     count = 1
#     for i in range(0, len(total_reviews), 3):
#         if count <=2:
#             print('training started :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;')
#             reviews = total_reviews[i:i+3]
#             reviews = [review for review in reviews if len(review) != 0]
#             print(reviews)
#             # remove instances with few reviews
#             if len(reviews) < min_reviews:
#                 continue
#       # tokenize reviews
#             tok_reviews = []
#             # print(len(reviews))
#             for review in reviews:
#                 tok_reviews.append([tokenizer.encode(sentence) for sentence in review])
#             print(tok_reviews)
#             sentence_switches_list = []
#             word_switches_list = []
#             document_switches = []
#       # run model
#             # print('running model...')
#             count_2 = 1
#             for j in range(0, len(tok_reviews), 2):
#
#                 tok_reviews_batch = tok_reviews[j:j+2]
#                 if count_2 < 2:
#
#                     print('token_reviews_batch has a shape %s' % str(np.shape(tok_reviews_batch)), 'which is', tok_reviews_batch)
#                 # print(tok_reviews_batch)
#                 inp_batch = [(review, -1) for review in tok_reviews_batch]
#                 inp_batch, _ = aspect_detection_collate(inp_batch)
#                 if count_2 < 2:
#                     print('inp_batch has a shape %s' % str(np.shape(inp_batch)), 'which is', inp_batch)
#                 inp_batch = inp_batch.cuda()
#                 with torch.no_grad():
#                     preds = model(inp_batch)
#                 document_pred = preds['document'].tolist()
#                 if count_2 < 2:
#                     print('document level prediction has a shape %s' % str(np.shape(document_pred)), 'which is', document_pred)
#
#
#
#                 # print(document_pred)
#                 # for pp in document_pred:
#                 #     doc_contains_aspect = np.any([x>0 for x in pp])
#                 #     if doc_contains_aspect:
#                 #         print('True')
#                 sentence_pred = preds['sentence'].cpu().detach().numpy()
#                 if count_2 < 2:
#                     print('sentence level prediction has a shape %s' % str(np.shape(sentence_pred)), 'which is', sentence_pred)
#                 # print(sentence_pred)
#                 word_pred = preds['word'].cpu().detach().numpy()
#                 if count_2 < 2:
#                     print('token level prediction has a shape %s' % str(np.shape(word_pred)))
#                     # , 'which is', word_pred)
#                 # print(word_pred)
#                 sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
#                 if count_2 < 2:
#                     print('sentence level weight has a shape %s' % str(np.shape(sentence_weight)), 'which is', sentence_weight)
#                 word_weight = preds['word_weight'].cpu().detach().numpy()
#                 if count_2 < 2:
#                     print('word level weight has a shape %s' % str(np.shape(word_weight)), 'which is', word_weight)
#                 document_switches += document_pred
#                 if count_2 < 2:
#                     print('document switch has a shape %s' % str(np.shape(document_switches)), 'this switches is:', document_switches )
#                 count_2 += 1
#                 count_8 = 1
#                 for k, sentences in enumerate(tok_reviews_batch):
#                     if count_8 < 2:
#                         print('enumerating the tokenized review batches >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                     sentence_switches = [] # store the sentence pred score times it's weight
#                     word_switches = []
#                     count_4 = 1
#
#                     for l, sentence in enumerate(sentences):
#                         if count_4 < 2:
#                             print('enumerating sentences >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                         tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]
#                         if count_4 < 2:
#                             print('tokens has a shape %s' % str(np.shape(tokens)), 'which is', tokens)
#                         # print(tokens[0][0])
#                         sentence = tokenizer.decode(sentence, skip_special_tokens=True)
#                         if count_4 < 2:
#                             print('sentence has a shape %s' % str(sentence), 'which is', sentence)
#                         pred = sentence_pred[k,l]
#                         if count_4 < 2:
#                             print('current sentence pred has a shape %s' % str(np.shape(pred)), 'which is', pred)
#                         weight = sentence_weight[k,l]
#                         if count_4 < 2:
#                             print('sentence_weight has a shape %s' % str(weight), 'which is', weight)
#                         sentence_switch = pred * weight
#                         if count_4 < 2:
#                             print('sentence_switch has a shape %s' % str(np.shape(sentence_switch)), 'which is', sentence_switch)
#                         sentence_switches.append((sentence, sentence_switch))
#                         if count_4 < 2:
#                             print('sentence_switches has a shape %s' % str(np.shape(sentence_switches)), 'which is', sentence_switches)
#                         word_switches_of_sentence = {}
#                         # count_4 += 1
#                         # print(type(word_switches_of_sentence))
#                         word = ""
#                         pred = []
#                         weight = []
#                         count_3 = 0
#                         print('\n Token level start')
#                         print('what are tokens will be enumerating', tokens)
#                         for m, token in enumerate(tokens):
#                             if token[0] == '\u0120':
#                                 if count_3 < 4:
#                                     print('encounter that big GGGG when enumerating the tokens tokenstokenstokenstokenstokenstokenstokenstokenstokens' )
#                             # start of a new token; reset values
#                                 word = word.replace('<s>', '')
#                                 token = token[1:]
#                                 if count_3 < 4:
#                                     print('token a shape %s' % str(np.shape(token)), 'which is', token)
#                                 pred = np.mean(pred, axis=0)
#                                 if count_3 < 4:
#                                     print('word level pred has a shape %s' % str(pred), 'which is', pred)
#                                 weight = np.mean(weight, axis=0)
#                                 if count_3 < 4:
#                                     print('word level weight has a shape %s' % str(weight), 'which is', weight)
#                                 word_switch = np.maximum(0, pred * weight)
#                                 if count_3 < 4:
#                                     print('word_switch prediction has a shape %s' % str(word_switch), 'which is', word_switch)
#                                 # print(type(word_switches_of_sentence))
#                                 if word not in word_switches_of_sentence:
#                                     word_switches_of_sentence[word] = 0
#                                 word_switches_of_sentence[word] += word_switch
#                                 word = ""
#                                 pred = []
#                                 weight = []
#                                 count_3 += 1
#
#                             word += token
#                             pred.append(word_pred[k,l,m])
#                             if count_3 < 2:
#                                 print('This is the first code excuted in token level _1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1 word_pred prediction has a shape %s' % str(np.shape(pred)), 'which is', pred)
#                             weight.append(word_weight[k,l,m])
#                             if count_3 < 2:
#                                 print('word_weight prediction has a shape %s' % str(np.shape(weight)), 'which is', weight )
#                             count_3 += 1
#                         word_switches_of_sentence = [(word, word_switches_of_sentence[word]) for word in word_switches_of_sentence] # list of switches
#                         if count_4 < 2:
#                             print('word_switches_of_sentence prediction has a shape %s' % str(np.shape(word_switches_of_sentence)), 'which is', word_switches_of_sentence)
#                         word_switches.append(word_switches_of_sentence) # list of list of switches
#                         if count_4 < 2:
#                             print('word_switches prediction has a shape %s' % str(np.shape(word_switches)), 'which one of the element is ', word_switches[0])
#                         count_4 += 1
#                     sentence_switches_list.append(sentence_switches)
#                     if count_8 < 2:
#                         print('sentence_switches_list prediction has a shape %s' % str(np.shape(sentence_switches_list)), 'which one of the element is', sentence_switches_list[0])
#                     word_switches_list.append(word_switches) # list of list of list of switches
#                     if count_8 < 2:
#                         print('word_switches_list prediction has a shape %s' % str(np.shape(word_switches_list)),'which one of the element inside is', word_switches_list[0])
#                     count_8 += 1
#       # sample summary and its reviews, keywords
#             if count < 2:
#                 # print('sentence_switch prediction has a shape %s' % str(sentence_switch), 'which is', sentence_switch)
#                 print('creating data.......................................................')
#                 print('\n')
#             count_5 = 1
#             for s_id, summary in enumerate(reviews):
#                 print('creating data and start enumerating reviews, current reviews is<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', reviews)
#                 sentence_list = []
#                 count_6 = 1
#                 for j, sentence in enumerate(summary):
#                     # print(sentence)
#                     sentence_switch = sentence_switches_list[s_id][j][1]
#                     if count_6 < 2:
#                         print('sentence_switch prediction has a shape %s' % str(np.shape(sentence_switch)), 'which is', sentence_switch)
#                     # print(sentence_switch)
#                     contains_aspect = np.any([x>0 for x in sentence_switch])
#                     if not contains_aspect:
#                         continue
#                     sentence_list.append(sentence) # append related sentence into a list
#                     if count_6 < 2:
#                         print('sentence_list has a shape %s' % str(np.shape(sentence_list)), 'which is', sentence_list)
#                     count_6 += 1
#                     # print(len(sentence_list))
#                 summary = ' '.join(sentence_list)
#                 if count_5 < 2:
#                     print('summary prediction has a shape %s' % str(summary), 'which is', summary)
#                 # if summary in summary_set:
#                 #     continue
#                 # check min max length
#                 if len(summary.split()) < min_summary_tokens or len(summary.split()) > max_summary_tokens:
#                     continue
#                 document_switch = document_switches[s_id]
#                 if count_5 < 2:
#                     print('document_switch prediction has a shape %s' % str(np.shape(document_switch)), 'which is', document_switch)
#                 # remove sentences of this review
#                 sentence_switches = []
#                 #word_switches = word_switches_list[s_id]
#                 word_switches = [] # list of list of switches
#                 for j in range(len(sentence_switches_list)):
#                     if j != s_id:
#                         sentence_switches += sentence_switches_list[j] #list of list of switches
#                         print('The list of list of switches is ', sentence_switches )
#                         word_switches += word_switches_list[j] #list of list of list of switches
#                         print('The list of listos list of switches is word_switches', word_switches )
#                 # get sentences
#                 sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
#                 if count_5 < 2:
#                     print('sentence_scores prediction has a shape %s' % str(np.shape(sentence_scores)), 'which is', sentence_scores)
#                 sentence_ids = np.argsort(sentence_scores)
#                 if count_5 < 2:
#                     print('sentence_ids sorted result has a shape %s' % str(np.shape(sentence_ids)), 'which is', sentence_ids)
#
#                 input_length = 0
#                 new_reviews = []
#                 for j, idx in enumerate(sentence_ids):
#                     if sentence_scores[idx] == 1e9:
#                         break
#                     if input_length > 600:
#                         break
#                     try:
#                         sentence = sentence_switches[idx][0]
#                     except:
#                         continue
#                     input_length += len(sentence.split())
#                     new_reviews.append(sentence)
#                 sentence_ids = sentence_ids[:j]
#                 if count_5 < 2:
#                     print('sentence_id prediction has a shape %s' % str(np.shape(sentence_ids)), 'which is', sentence_ids)
#                 if len(sentence_ids) == 0: # no related sentences
#                     continue
#                     # combine word switches
#                 word_switches_dict = {}
#                 count_7 = 0
#                 for idx in range(len(word_switches)):
#                     if idx not in sentence_ids:
#                         continue
#                     for word, switch in word_switches[idx]:
#                         if word not in word_switches_dict:
#                             word_switches_dict[word] = np.zeros(args.num_aspects)
#                             # count +=1
#                             if count_7 < 2:
#                                 print('word_switches_dict prediction has a shape %s' % str(np.shape(word_switches_dict)), 'which is', word_switches_dict)
#                             count_7 += 1
#                         word_switches_dict[word] += np.maximum(0, switch)
#                 word_switches_final = [(word, word_switches_dict[word]) for word in word_switches_dict]
#                 if count_5 < 2:
#                     print('word_switches_final prediction has a shape %s' % str(np.shape(word_switches_final)), 'which is', word_switches_final)
#                 word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches_final]
#                 if count_5 < 2:
#                     print('word_scores prediction has a shape %s' % str(np.shape(word_scores)), 'which is', word_scores)
#                 word_switches = [(word_switch, word_score)
#                   for word_switch, word_score in zip(word_switches_final, word_scores)
#                   if word_score != 1e9
#                 ]
#                 if count_5 < 2:
#                     print('word_switches prediction has a shape %s' % str(np.shape(word_switches)), 'which is', word_switches)
#                 count_5 += 1
#                 word_switches = sorted(word_switches, key=lambda a: a[-1])[:2*num_keywords]
#                 keywords = [word_switch[0][0] for word_switch in word_switches if word_switch[0][0] not in all_stopwords]
#                 pair = {}
#                 pair['summary'] = summary
#                 pair['reviews'] = new_reviews
#                 pair['keywords'] = keywords
#                 print('length of the keywords is %d' % len(keywords), keywords)
#                 pair['switch'] = document_switch
#
#                 print('finished one loop of the whole function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#
#                 # print(dataset_file)
#                 # f = open(dataset_file, 'a')
#                 # pi = {'test':2}
#                 # f.write(json.dump(pi) + '\n')
#                 # f.write(json.dumps(pair) + '\n')
#                 # f.close()
#                 # count += 1
#             # print('Done')
#         count += 1
# #
# #
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-mode', default='train', type=str)
#     parser.add_argument('-num_heads', default=12, type=int)
#     parser.add_argument('-dataset', default='finishline', type=str)
#     parser.add_argument('-num_aspects', default=7, type=int)
#     parser.add_argument('-model_name', default='naive', type=str)
#     parser.add_argument('-model_dim', default=768, type=int)
#     parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/mil_7_5_words_man5000/naive.5000.3.11.92.89.89.88', type=str)
#     parser.add_argument('-model_type', default='distilroberta-base', type=str)
#     parser.add_argument('-data_source', default='mil_7_5_words_woman.json', type=str)
#     parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)
#
#     args = parser.parse_args()
#
#     if args.mode == 'train':
#         create_train_data(args)
#     elif args.mode == 'eval-aspect':
#         create_aspect_test_data(args)
#     elif args.mode == 'eval-general':
#         create_general_test_data(args)
#     elif args.mode == 'loss_test':
#         loss_test()











#
# pred = []
#
# pred = np.max(pred, axis=0)
# dataset_file = '/home/jade/untextsum/data/' + 'finishline' + '/train_sum.jsonl'
# f = open(dataset_file, 'a')
# dic = {'jiba':2}
# f.write(json.dumps(dic) + '\n')
# f.close()


"""Jiangpilianzi you youdianbukaixin"""

#
# a = False
#
# if a:
#     print('dui')
#
# if not a:
#     print('cuo')

# m_data = []
# with open ('/home/jade/untextsum/data/finishline/mil_man.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         m_data.append(data)
#
# print(len(m_data))
# # print(m_data[0])
#
# aspect_count = {}
# for ii, pp in enumerate(m_data):
#     for ppp in pp['aspects'].values():
#         if 'yes' in ppp:
#             aspect_count[str(ii)] = 1
#         else:
#             continue
#
# print(len(aspect_count))


#
# def soft_margin(a, b):
#     a = np.maximum(0, a)
#     b = np.maximum(0, b)
#     return np.log(1 + np.exp(-a * b)).sum()
#
#
# def soft_margin_list(a_list, b):
#     ret = 0
#     for a in a_list:
#         ret += soft_margin(a, b)
#     return ret
#
#
# def get_model(args):
#
#     model = MIL(args)
#     model.cuda()
#
#     tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
#     best_point = torch.load(args.load_model)
#     model.load_state_dict(best_point['model'])
#
#     return model, tokenizer
#
#
# def create_data(args, min_token_frequency=5, min_reviews=10, min_summary_tokens=10, #30, 60
#                 max_summary_tokens=100, #60, 100
#                 max_tokens=512, num_keywords=10):
#     data = []
#     f = open('/home/jade/untextsum/data/' + args.dataset + '/mil_man.json', 'r')
#     for line in tqdm(f):
#         inst = json.loads(line.strip())
#         # print(type(inst['review']))
#         data.append(inst)
#     f.close()
#     random.shuffle(data)
#
#     dataset_file = '/home/jade/untextsum/data/' + args.dataset + '/train_sum.jsonl'
#     summary_set = set()
#
#     model, tokenizer = get_model(args)
#     model.eval()
#
#     total_reviews = [inst['review'] for inst in tqdm(data)]
#     for i in range(0, len(total_reviews), 50):
#         reviews = total_reviews[i:i+50]
#         reviews = [review for review in reviews if len(review) != 0]
#         # remove instances with few reviews
#         if len(reviews) < min_reviews:
#             continue
#   # tokenize reviews
#         tok_reviews = []
#         # print(len(reviews))
#         for review in reviews:
#             # print(review)
#             tok_reviews.append([tokenizer.encode(sentence) for sentence in review])
#         # print('-------------------------------------------', np.shape(tok_reviews[1]))
#         # print(len(tok_reviews))
#         sentence_switches_list = []
#         word_switches_list = []
#         document_switches = []
#   # run model
#         print('running model...')
#         for j in range(0, len(tok_reviews), 2):
#             tok_reviews_batch = tok_reviews[j:j+2]
#             # print(tok_reviews_batch)
#             inp_batch = [(review, -1) for review in tok_reviews_batch]
#             inp_batch, _ = aspect_detection_collate(inp_batch)
#             # print('------------------------------------------------', np.shape(inp_batch))
#             inp_batch = inp_batch.cuda()
#             # print(np.shape(inp_batch))
#             with torch.no_grad():
#                 preds = model(inp_batch)
#             document_pred = preds['document'].tolist()
#             # print(document_pred)
#             # for pp in document_pred:
#             #     doc_contains_aspect = np.any([x>0 for x in pp])
#             #     if doc_contains_aspect:
#             #         print('-----------------True')
#             sentence_pred = preds['sentence'].cpu().detach().numpy()
#             # print(sentence_pred)
#             word_pred = preds['word'].cpu().detach().numpy()
#             # print(word_pred)
#             sentence_weight = preds['sentence_weight'].cpu().detach().numpy()
#             word_weight = preds['word_weight'].cpu().detach().numpy()
#             document_switches += document_pred
#             for k, sentences in enumerate(tok_reviews_batch):
#                 sentence_switches = []
#                 word_switches = []
#                 for l, sentence in enumerate(sentences):
#                     tokens = tokenizer.convert_ids_to_tokens(sentence)[:100]
#                     # print(tokens[0][0])
#                     sentence = tokenizer.decode(sentence, skip_special_tokens=True)
#                     pred = sentence_pred[k,l]
#                     weight = sentence_weight[k,l]
#                     sentence_switch = pred * weight
#                     sentence_switches.append((sentence, sentence_switch))
#                     word_switches_of_sentence = {}
#                     # print(type(word_switches_of_sentence))
#                     word = ""
#                     pred = []
#                     weight = []
#                     for m, token in enumerate(tokens):
#                         if token[0] == '\u0120':
#                         # start of a new token; reset values
#                             word = word.replace('<s>', '')
#                             token = token[1:]
#                             pred = np.max(pred, axis=0)
#                             weight = np.max(weight, axis=0)
#                             word_switch = np.maximum(0, pred * weight)
#                             # print(type(word_switches_of_sentence))
#                             if word not in word_switches_of_sentence:
#                                 word_switches_of_sentence[word] = 0
#                             word_switches_of_sentence[word] += word_switch
#                             word = ""
#                             pred = []
#                             weight = []
#                         word += token
#                         pred.append(word_pred[k,l,m])
#                         weight.append(word_weight[k,l,m])
#                     word_switches_of_sentence = [(word, word_switches_of_sentence[word]) for word in word_switches_of_sentence] # list of switches
#                     word_switches.append(word_switches_of_sentence) # list of list of switches
#                 sentence_switches_list.append(sentence_switches)
#                 word_switches_list.append(word_switches) # list of list of list of switches
#   # sample summary and its reviews, keywords
#         print('creating data...')
#         for s_id, summary in enumerate(reviews):
#             sentence_list = []
#             for j, sentence in enumerate(summary):
#                 # print(sentence)
#                 sentence_switch = sentence_switches_list[s_id][j][1]
#                 # print(sentence_switch)
#                 contains_aspect = np.any([x>0 for x in sentence_switch])
#                 if not contains_aspect:
#                     continue
#                 sentence_list.append(sentence)
#                 print(len(sentence_list))
#             summary = ' '.join(sentence_list)
#             if summary in summary_set:
#                 continue
#             # check min max length
#             if len(summary.split()) < min_summary_tokens or len(summary.split()) > max_summary_tokens:
#                 continue
#             document_switch = document_switches[s_id]
#             # remove sentences of this review
#             sentence_switches = []
#             #word_switches = word_switches_list[s_id]
#             word_switches = [] # list of list of switches
#             for j in range(len(sentence_switches_list)):
#                 if j != s_id:
#                     sentence_switches += sentence_switches_list[j]
#                     word_switches += word_switches_list[j]
#             # get sentences
#             sentence_scores = [soft_margin(sentence_switch[-1], document_switch) for sentence_switch in sentence_switches]
#             sentence_ids = np.argsort(sentence_scores)
#             input_length = 0
#             new_reviews = []
#             for j, idx in enumerate(sentence_ids):
#                 if sentence_scores[idx] == 1e9:
#                     break
#                 if input_length > 600:
#                     break
#                 try:
#                     sentence = sentence_switches[idx][0]
#                 except:
#                     continue
#                 input_length += len(sentence.split())
#                 new_reviews.append(sentence)
#             sentence_ids = sentence_ids[:j]
#             if len(sentence_ids) == 0: # no related sentences
#
#                 continue
#                 # combine word switches
#             word_switches_dict = {}
#             for idx in range(len(word_switches)):
#                 if idx not in sentence_ids:
#                     continue
#                 for word, switch in word_switches[idx]:
#                     if word not in word_switches_dict:
#                         word_switches_dict[word] = np.zeros(args.num_aspects)
#                     word_switches_dict[word] += np.maximum(0, switch)
#             word_switches_final = [(word, word_switches_dict[word]) for word in word_switches_dict]
#             word_scores = [soft_margin(word_switch[-1], document_switch) for word_switch in word_switches_final]
#             word_switches = [(word_switch, word_score)
#               for word_switch, word_score in zip(word_switches_final, word_scores)
#               if word_score != 1e9
#             ]
#             word_switches = sorted(word_switches, key=lambda a: a[-1])[:num_keywords]
#             keywords = [word_switch[0][0] for word_switch in word_switches]
#             pair = {}
#             pair['summary'] = summary
#             pair['reviews'] = new_reviews
#             pair['keywords'] = keywords
#             pair['switch'] = document_switch
#             print(dataset_file)
#             f = open(dataset_file, 'a')
#             # pi = {'test':2}
#             # f.write(json.dump(pi) + '\n')
#             f.write(json.dumps(pair) + '\n')
#             f.close()
#             # count += 1
#         print('Done')
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-mode', default='train', type=str)
#     parser.add_argument('-num_heads', default=12, type=int)
#     parser.add_argument('-dataset', default='finishline', type=str)
#     parser.add_argument('-num_aspects', default=22, type=int)
#     parser.add_argument('-model_name', default='naive', type=str)
#     parser.add_argument('-model_dim', default=768, type=int)
#     parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/naive.1000.14.20.87.07.84.42', type=str)
#     parser.add_argument('-model_type', default='distilroberta-base', type=str)
#
#     args = parser.parse_args()
#
#     if args.mode == 'train':
#         create_data(args)

#
#
# a = 1
#
# assert a == 1

# import torch
#
# a = torch.randint(1, 2, (1,))
# print(a)
# b = torch.randint(1, 2, (1,))
# print(b)
#
#
# print(torch.exp(-a*b))
# print(torch.exp(a*b))
# import math
#
#
# class MyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, start, end):
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:  # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else:  # in a worker process
#             # split workload
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#         return iter(range(iter_start, iter_end))
# # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
# ds = MyIterableDataset(start=3, end=7)
#
# # Single-process loading
# print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
#
# # Mult-process loading with two worker processes
# # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
# print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
#
# # With even more workers
# print(list(torch.utils.data.DataLoader(ds, num_workers=20)))

from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import numpy as np

# class AspectDetectionDataset(IterableDataset):
#
#     def __init__(self, file, tokenizer, shuffle=True):
#         self.file = file
#         self.shuffle = shuffle
#         self.tokenizer = tokenizer
#         self.buffer_size = 4096
#
#     def process(self, inst):
#         if type(inst) is str:
#             inst = json.loads(inst)
#         inp = inst['review'] # list of sentences
# #         print('current encoding review is %s' % str(inp))
#         inp = [self.tokenizer.encode(sentence) for sentence in inp]
#         out = [] # list of classes
#         aspects = list(inst['aspects'].keys())
# #         aspects.sort() TOTALLY no need to perform this, this guy is just stupid
#         for aspect in aspects:
#             if inst['aspects'][aspect] == 'yes':
#                 out.append(1)
#             else:
#                 out.append(-1)
#         # print(out)
# #         print('In the AspectDetection Dataset, Process output the inp is: %s' % str(inp), 'In the AspectDetection Dataset, Process output the inp shape is: %s' % str(numpy.shape(inp)))
# #         print('In the AspectDetection Dataset, Process output the out shape is: %s' %str(out), 'In the AspectDetection Dataset, Process output the out shape is: %s' % str(numpy.shape(out)))
#         return inp, out # return the tokens and aspects len list with 1 and -1
#
#     def __iter__(self):
#         count = 0
#         if self.shuffle:
#             shufbuf = []
#             try:
#                 dataset_iter = open(self.file, 'r')
#                 print('what is the dataset_iter type %s' % type(dataset_iter))
#                 for i in range(self.buffer_size):
#                     shufbuf.append(next(dataset_iter))
#                     # print('shufbuf is what %s, what is its size %d, what is it shape %s' % (str(type(shufbuf)), len(shufbuf), str(np.shape(shufbuf))))
#             except:
#                 print('does this code ever excuted?')
#                 self.buffer_size = len(shufbuf)
#             print('whats in shufbuf %s, its length is %d' % (str(shufbuf[0]), len(shufbuf)))
#             try:
#                 while True:
#                     # print('a loop has been excuted here')
#                     try:
# #                         dataset_iter = open(self.file, 'r')
#                         item = next(dataset_iter)
#                         # print('what is this item %s, this item is %s' % (type(item), item))
#                         evict_idx = random.randint(0, self.buffer_size-1)
#                         # print('what is this evict_idx %s' % str(evict_idx))
#                         # print(shufbuf[evict_idx])
#
#                         yield self.process(shufbuf[evict_idx])
#                         count += 1
#                         shufbuf[evict_idx] = item
#                     except StopIteration:
#                         break
#                 while len(shufbuf) > 0:
#                     # print('does this code ever excuted?')
#                     # print(shufbuf.pop())
#                     print(count)
#                     yield self.process(shufbuf.pop())
#                     count += 1
#             except GeneratorExit:
#                 print('does this code ever excuted?')
#                 pass
#
#         else:
#             f = open(self.file, 'r')
#             for line in f:
#                 yield self.process(line)
#             f.close()
#
#
#
#
# def aspect_detection_collate(batch, mask_id=0):
#     text = [inst[0] for inst in batch] # B, S, T
#     # print('what is in the batch %s' % str(batch[0]))
#
# #     print('what is the BST ? %s' % str(type(text)), text[:2])
#     max_sentence_len = max([len(sentences) for sentences in text])
# #     print('what is the max sentence length ? %s' % str(max_sentence_len))
#     max_token_len = min(100, max([max([len(tokens) for tokens in sentences]) for sentences in text]))
# #     print('what is the max token length ? %s' % str(max_token_len))
#     padded_text = []
#     for sentences in text:
#         padded_sentences = []
#         for tokens in sentences:
#             if len(tokens) < max_token_len:
#                 tokens = tokens + [mask_id] * (max_token_len - len(tokens))
#             tokens = tokens[:max_token_len]
#             assert len(tokens) == max_token_len
#             padded_sentences.append(tokens)
#
#         if len(padded_sentences) < max_sentence_len:
#             zeros = [mask_id] * max_token_len
#             padded_sentences = padded_sentences + [zeros] * (max_sentence_len - len(padded_sentences))
#         assert len(padded_sentences) == max_sentence_len
#         padded_text.append(padded_sentences)
#
#     padded_text = torch.tensor(padded_text)
#     label = [inst[1] for inst in batch]
# #     print('what is inst? inst size is %s' % str(numpy.shape(batch[0])), 'what is inst? label size is %s' % str(numpy.shape(label)), label)
#     label = torch.tensor(label)
#
# #     print('In the aspect_detection_collate, Process output the padded_text shape is: the three dimension is batch size, the number of sentences that a review
# # contains most sentences, the longest sentences that contain token number,
# # Thus B, S, T. the second was actually sentence number instead of sentence length %s' % str(numpy.shape(padded_text)))
# #     print('In the aspect_detection_collate, Process output the label shape is: %s' % str(numpy.shape(label)))
#     return padded_text, label
#
#
# def see_iter (args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_type)
#     print('dataset is the file %s' % args.data_dir + '/' + args.dataset + '/' + args.train_file)
#     dataset = AspectDetectionDataset(args.data_dir + '/' + args.dataset + '/' + args.train_file, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=aspect_detection_collate)
#     # print('lenlenlenlenlnelnelnlenlenlenlenlenlenelnelenlenlenelnelenlen', len(list(dataloader)))
#     for _, (inp_batch, out_batch) in enumerate(tqdm(dataloader)):
#         # model.train()
#         inp_batch = inp_batch.cuda()
#         # print(inp_batch)
#         out_batch = out_batch.cuda().float()
#         # print(out_batch)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-mode', default='train', type=str)
#     parser.add_argument('-dataset', default='finishline', type=str)
#     # parser.add_argument('-num_aspects', default=7, type=int)
#     parser.add_argument('-model_name', default='naive', type=str)
#     parser.add_argument('-load_model', default=None, type=str)
#     parser.add_argument('-train_file', default='mil_7_5_words_man.json', type=str)
#     # parser.add_argument('-dev_file', default='dev_mil_7_5_words_man.json', type=str)
#     parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
#     parser.add_argument('-model_dir', default='/home/jade/untextsum/model', type=str)
#     parser.add_argument('-model_type', default='distilroberta-base', type=str)
#     # parser.add_argument('-model_dim', default=768, type=int)
#     # parser.add_argument('-num_heads', default=12, type=int)
#     # parser.add_argument('-vocab_size', default=50265, type=int)
#     parser.add_argument('-batch_size', default=64, type=int)
#     parser.add_argument('-learning_rate', default=1e-4, type=float)
#     # parser.add_argument('-no_train_steps', default=5000, type=int)
#     # parser.add_argument('-no_warmup_steps', default=500, type=int)
#     # parser.add_argument('-check_every', default=100, type=int)
#     # parser.add_argument('-ckpt_every', default=1000, type=int)
#     # parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)
#
#     args = parser.parse_args()
#     if args.mode == 'train':
#         see_iter(args)
