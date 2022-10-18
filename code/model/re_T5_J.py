import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as Model
from transformers import T5Tokenizer
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
from rouge_score import rouge_scorer
import copy
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score
from tqdm import tqdm
import datetime
from datetime import date
# from calculate_rouge import calculate
sys.path.append(os.path.realpath('..'))
from data_processor.re_readers import AspectDetectionDataset, aspect_detection_collate, SummarizationDataset,\
re_SummarizationDataset, ada_SummarizationDataset


def calculate(list_a, list_b):
    rouge_score = []
    for i in range(len(list_a)):
        rouge_score.append(scorer(list_a[i], list_b[i])['rougeL'][-1])
    return rouge_score


def train(args):
    print(args)
    print('Preparing data...')
    if args.train_file is None:
        train_file = args.data_dir + '/' + args.dataset + '/' + args.mil_data_dir + '/retrain_sum.jsonl'
    else:
        train_file = args.train_file
    dataset = SummarizationDataset(train_file, use_keywords=args.use_keywords, use_switch=args.use_switch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print('load train file done')
    asp_dev_file = args.asp_dev_file
    print('Reading the aspect summary dataset...')
    asp_dev_dataset = SummarizationDataset(asp_dev_file, use_keywords=args.use_keywords,
    use_switch=args.use_switch, shuffle=False)
    asp_dev_dataloader = DataLoader(asp_dev_dataset, batch_size=args.batch_size)
    f = open(asp_dev_file, 'r')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    f.close()
    asp_gold_sums = [inst['summary'].lower() for inst in data]
    print('Read done')
    print('Initializing model...')
    tokenizer = T5Tokenizer.from_pretrained(args.model_type)
    special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
    if args.use_switch != 'none':
        special_tokens.append('<Pos>')
        special_tokens.append('<Neu>')
        special_tokens.append('<Neg>')
        for i in range(args.num_aspects-3):
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
    count = 0
    while step < args.no_train_steps:
        losses = []
        for _, (inp_batch, out_batch, switch_batch) in enumerate(tqdm(dataloader)):
            model.train()
            batch_encoding = tokenizer(text=list(inp_batch),
            max_length=args.max_length,
            padding='longest', truncation=True, return_tensors='pt')
            inp_ids = batch_encoding.input_ids.cuda()
            inp_mask = batch_encoding.attention_mask.cuda()
            target_encoding = tokenizer(text=list(out_batch), padding='longest',
            max_length=args.max_target_length, truncation=True, return_tensors='pt')
            out_ids = target_encoding.input_ids.cuda()
            out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
            out_ids[out_ids==0] = -100
            dec_inp_ids = model._shift_right(out_ids)
            model_outputs = model(input_ids=inp_ids, attention_mask=inp_mask,
            decoder_input_ids=dec_inp_ids, labels=out_ids, output_hidden_states=True)
            loss = model_outputs.loss
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            count += 1
            if step % args.check_every == 0:
                print('Step %d Loss %.4f' % (step, np.mean(losses)))
                model.eval()
                rouge_scores = []
                asp_pred_sums = []
                for _, (inp_batch, out_batch, _) in enumerate(tqdm(asp_dev_dataloader)):
                    model.eval()
                    batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch),
                    tgt_texts=list(out_batch), max_length=args.max_length,
                    max_target_length=args.max_target_length, padding=True, truncation=True, return_tensors='pt')
                    inp_ids = batch_encoding['input_ids'].cuda()
                    preds = model.generate(inp_ids, min_length=10, max_length=args.max_target_length,
                    num_beams=2, no_repeat_ngram_size=2, decoder_start_token_id=0,
                    repetition_penalty=1, length_penalty=1,)
                    for pred in preds:
                        asp_pred_sums.append(tokenizer.decode(pred))
                asp_scores = calculate(asp_gold_sums, asp_pred_sums)
                rouge_scores += list(asp_scores)
                print('The rouge score list now is:', np.shape(rouge_scores))
                rouge = np.mean(rouge_scores)
                print("ROUGE: %.4f" % rouge)
                print("Aspect Gold:", asp_gold_sums[0])
                print("Aspect Pred:", asp_pred_sums[0])
                e = datetime.datetime.now()
                if rouge > best_rouge:
                    os.makedirs(args.model_dir + '/' + args.model_name + 'aspect' + str(args.num_aspects)
                    + 'rep_pen' + str(args.repetition_penalty) + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                    + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps) + str(e.month)
                    + str(e.day) + str(e.hour), exist_ok=True)
                    print('Saving...')
                    best_rouge = rouge
                    torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step, 'loss': np.mean(losses)
                    }, args.model_dir + '/' + args.model_name + 'aspect' + str(args.num_aspects)
                    + 'rep_pen' + str(args.repetition_penalty) + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                    + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps) + str(e.month)
                    + str(e.day) + str(e.hour)
                    + '/' + 'Rouge' +
                    '.best_step.%d_rouge.%.3f' % (step, rouge))

            if step % args.ckpt_every == 0:
                os.makedirs(args.model_dir + '/'+ args.model_name + 'aspect' + str(args.num_aspects)
                + 'rep_pen' + str(args.repetition_penalty)+ '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps) + str(e.month)
                + str(e.day) + str(e.hour), exist_ok=True)
                print('Saving...')
                torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step, 'loss': np.mean(losses),
                }, args.model_dir + '/' + args.model_name + 'aspect' + str(args.num_aspects)
                + 'rep_pen' + str(args.repetition_penalty) + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps)  + str(e.month)
                + str(e.day) + str(e.hour)
                + '/' + 'Step' +
                'step.%d_loss.%.2f' % (step, np.mean(losses)))
                losses = []
            if step == args.no_train_steps:
                break


def evaluate(args, test_type='aspect'):
    print(args)
    print('Preparing data...')
    if test_type == 'aspect':
        test_file = args.asp_test_file
    else:
        test_file = args.gen_test_file
    dataset = SummarizationDataset(test_file, use_keywords=args.use_keywords, use_switch=args.use_switch,
    shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print('Initializing model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
    if args.use_switch != 'none':
        special_tokens.append('<Pos>')
        special_tokens.append('<Neu>')
        special_tokens.append('<Neg>')
        for i in range(args.num_aspects):
            special_tokens.append('<asp_%d>' % i)
    print('current special token set is:', special_tokens)

    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    pad_id = tokenizer.pad_token_id

    model = Model.from_pretrained(args.model_type, return_dict=True)
    model.resize_token_embeddings(len(tokenizer))
    # model = nn.DataParallel(model)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.no_warmup_steps, args.no_train_steps)

    assert args.load_model is not None
    best_point = torch.load(args.load_model)
    model.load_state_dict(best_point['model'])
    optimizer.load_state_dict(best_point['optimizer'])
    scheduler.load_state_dict(best_point['scheduler'])

    rng = np.random.default_rng()

    os.makedirs('/home/jade/untextsum/output/' + args.dataset, exist_ok=True)
    f = open('/home/jade/untextsum/output/' + args.dataset + '/' + args.load_model.split('/')[-1] +
    '.out.' + test_type + '_2.jsonl', 'w')
    for _, (inp_batch, out_batch, _) in enumerate(tqdm(dataloader)):
        # print(inp_batch, out_batch)
        model.eval()
        batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch),
        tgt_texts=list(out_batch),max_length=args.max_length,max_target_length=args.max_target_length,
          padding=True,truncation=True,return_tensors='pt')
        inp_ids = batch_encoding['input_ids'].cuda()
        inp_mask = batch_encoding['attention_mask'].cuda()
        preds = model.generate(inp_ids,decoder_start_token_id=1,min_length=args.min_target_length,
        max_length=args.max_target_length,num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size, repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty)
        for pred in preds:
            f.write(tokenizer.decode(pred, skip_special_tokens=True) + '\n')
    f.close()



def ada_evaluate(args, test_type='aspect'):
    print(args)
    print('Preparing data...')
    if test_type == 'aspect':
        test_file = args.asp_test_file
    # textset = [('pos', 'gene'), ('pos', 'exterior')]
    textset = [('neg', 'gene'), ('neg', 'exterior')]
    for tu in textset:
        dataset = ada_SummarizationDataset(test_file, use_keywords=args.use_keywords, use_switch=args.use_switch,
        senti=tu[0], asp = tu[1], shuffle=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        print('Initializing model...')
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        special_tokens = ['<rev>', '<key>', '<sum>', '<switch>']
        if args.use_switch != 'none':
            special_tokens.append('<Pos>')
            special_tokens.append('<Neu>')
            special_tokens.append('<Neg>')
            for i in range(args.num_aspects):
                special_tokens.append('<asp_%d>' % i)
        print('current special token set is:', special_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        pad_id = tokenizer.pad_token_id
        model = Model.from_pretrained(args.model_type, return_dict=True)
        model.resize_token_embeddings(len(tokenizer))
        model.cuda()
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.no_warmup_steps, args.no_train_steps)
        assert args.load_model is not None
        best_point = torch.load(args.load_model)
        model.load_state_dict(best_point['model'])
        optimizer.load_state_dict(best_point['optimizer'])
        scheduler.load_state_dict(best_point['scheduler'])
        rng = np.random.default_rng()
        os.makedirs('/home/jade/ACOS/output/' + args.asp_test_file.split('/')[-1],
        exist_ok=True)
        f = open('/home/jade/ACOS/output/' + args.asp_test_file.split('/')[-1] +'/'\
        + args.load_model.split('/')[-1] +
        '.out.' + test_type + tu[0] + tu[1] + '_2.jsonl', 'w')
        for _, (inp_batch, out_batch, _) in enumerate(tqdm(dataloader)):
            model.eval()
            batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch),
            tgt_texts=list(out_batch),max_length=args.max_length,max_target_length=args.max_target_length,
              padding=True,truncation=True,return_tensors='pt')
            inp_ids = batch_encoding['input_ids'].cuda()
            inp_mask = batch_encoding['attention_mask'].cuda()
            preds = model.generate(inp_ids,decoder_start_token_id=0,min_length=args.min_target_length,
            max_length=args.max_target_length,num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size, repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty)
            for pred in preds:
                tmp_dic = {}
                tmp_dic['Pred_summary'] = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                f.write(json.dumps(tmp_dic) + '\n')
        f.close()


def chunker(seq, size):
    pp = []
    for pos in range(0, len(seq), size):
        pp.append(seq[pos:pos + size])
    return pp




def ada_sum(file, result):
    with open (file, 'r') as f:
        dic = json.load(f)
    all = []
    for k, v in dic.items():
        tmp_dic = {}
        tmp_dic['summary'] = ''
        tmp_dic['reviews'] = []
        tmp_dic['keywords'] = []
        tmp_dic['switch'] = []
        tmp_dic['product_name'] = k
        if len(v) <= 15:
            for inst in v:
                tmp_dic['reviews'].append(inst['text'])
            all.append(tmp_dic)
        else:
            for group in chunker(v, 15):
                tmp_tmp_dic = copy.deepcopy(tmp_dic)
                tmp_tmp_dic['reviews'] = [inst['text'] for inst in group]
                all.append(tmp_tmp_dic)
    with open(result, 'w') as f:
        for i in all:
            f.write(json.dumps(i) + '\n')
    print('write done')









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='eval-aspect', type=str)
    parser.add_argument('-dataset', default='allCampusmin60max120summary_source_num5review_\
source_num60keywords5',
    type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    parser.add_argument('-model_name', default='T5_B4', type=str)
    # parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/T5/T5dev_fin_5Pos_asp_10000/Rouge.best_step.7900_rouge.0.159', type=str)
    # parser.add_argument('-load_model', default='/media/jade/yi_Data/Data/MASmodel/T5_B4/Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight1600050000/Rouge.best_step.19400_rouge.0.147', type=str)
    # parser.add_argument('-load_model', default=None, type=str)
    parser.add_argument('-load_model', default='/home/jade/untextsum/model/T5_campus/T5_B4aspect10rep_pen5/\
Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 \
document_f1.81.44_best_dataBCE_Sum_weight160005000072822/Rouge.best_step.26100_rouge.0.144', type=str)
    parser.add_argument('-train_file', default=None, type=str)
    parser.add_argument('-asp_dev_file', default='/home/jade/untextsum/data/\
allCampusmin60max120summary_source_num5review_source_num60keywords5\
/Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 \
sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000/redev_fin.jsonl', type=str)
#     parser.add_argument('-asp_test_file', default='/home/jade/untextsum/data/\
# allCampusmin60max120summary_source_num5review_source_num60keywords5/\
# Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 \
# sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000/retest_fin.jsonl', type=str)
    parser.add_argument('-asp_test_file', default='/home/jade/ACOS/data/\
ada/bad_ada_sum.jsonl', type=str)
    parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
    parser.add_argument('-mil_data_dir', default='Sentiment_filtered10_BCE_Sum_weightstep.10000 \
average_loss.613.18 sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000', type=str)
    parser.add_argument('-model_dir', default='/home/jade/untextsum/model/T5_campus', type=str)
    # parser.add_argument('-model_type', default='google/t5-v1_1-small', type=str)
    parser.add_argument('-model_type', default='t5-small', type=str)
    parser.add_argument('-use_keywords', default='input', type=str) # none, input, output
    parser.add_argument('-use_switch', default='input', type=str) # none, input, output
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-learning_rate', default=1e-6, type=float)
    parser.add_argument('-no_train_steps', default=50000, type=int)
    parser.add_argument('-no_warmup_steps', default=25000, type=int)
    parser.add_argument('-check_every', default=100, type=int)
    parser.add_argument('-ckpt_every', default=1000, type=int)
    parser.add_argument('-max_length', default=500, type=int)
    parser.add_argument('-min_target_length', default=50, type=int)
    parser.add_argument('-max_target_length', default=250, type=int)
    parser.add_argument('-num_beams', default=2, type=int)
    parser.add_argument('-no_repeat_ngram_size', default=2, type=int)
    parser.add_argument('-repetition_penalty', default=5.0, type=float)
    parser.add_argument('-length_penalty', default=3, type=float)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval-aspect':
        # ada_sum('/home/jade/Downloads/Text_data729/Text_data/ada/man.json',
        #         '/home/jade/ACOS/data/ada/ada_sum.jsonl')
            #
            # parser.add_argument('-senti', default=str(tu[0]), type=str)
            # parser.add_argument('-asp', default=str(tu[1]), type=str)
        # args = parser.parse_args()
        ada_evaluate(args, 'aspect')
    elif args.mode == 'eval-multi':
        evaluate(args, 'multi')
    elif args.mode == 'eval-double':
        evaluate(args, 'double')
