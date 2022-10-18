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
# from transformers import AdamW
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True).score
from tqdm import tqdm
# from calculate_rouge import calculate
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
        train_file = args.data_dir + '/' + args.dataset + '/' + args.mil_data_dir + '/train_sum.jsonl'
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
    # model = nn.DataParallel(model)
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
            # if count <= 2:
            #     print(inp_batch,'the inp_batch, out batch, and switch_batch is isisiisisisiisisiisiisisAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
            #             out_batch, switch_batch)
            model.train()
            batch_encoding = tokenizer.prepare_seq2seq_batch(src_texts=list(inp_batch), tgt_texts=list(out_batch),
            max_length=args.max_length,max_target_length=args.max_target_length,
            padding=True, truncation=True, return_tensors='pt')
            inp_ids = batch_encoding['input_ids'].cuda()
            # if count <= 1:
            #     print('inp_ids has a shape which is %s' % str(inp_ids.size()), inp_ids)
            inp_mask = batch_encoding['attention_mask'].cuda()
            # if count <= 1:
            #     print('attention mask has a shape which is %s' % str(inp_mask.size()), inp_mask)
            out_ids = batch_encoding['labels'].cuda()
            # if count <= 1:
            #     print('out_ids has a shape which is %s' % str(out_ids.size()), out_ids)
            out_mask = torch.where(out_ids==0, 0, 1).unsqueeze(-1) # batch_size, out_len
            # if count <= 1:
            #     print('out_mask has a shape which is %s' % str(out_mask.size()), out_mask)
            out_ids[out_ids==0] = -100
            # if count <= 1:
            #     print('out_ids all 0 elements been transfered to -100 which is %s' % str(out_ids.size()), out_ids)
            dec_inp_ids = model._shift_right(out_ids)
            # if count <= 1:
            #     print('dec_inp_ids has a shape which is %s' % str(out_ids.size()), dec_inp_ids)
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

                if rouge > best_rouge:
                    os.makedirs(args.model_dir + '/' + args.model_name  + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                    + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps), exist_ok=True)
                    print('Saving...')
                    best_rouge = rouge
                    torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step, 'loss': np.mean(losses)
                    }, args.model_dir + '/' + args.model_name + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                    + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps) + '/' + 'Rouge' +
                    '.best_step.%d_rouge.%.3f' % (step, rouge))

            if step % args.ckpt_every == 0:
                os.makedirs(args.model_dir + '/'+ args.model_name + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps), exist_ok=True)
                print('Saving...')
                torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step, 'loss': np.mean(losses),
                }, args.model_dir + '/' + args.model_name + '/' #+ args.gen_dev_file.split('dev_fin_')[-1].split('_gen')[0]
                + args.asp_dev_file.split('/')[-2] + str(args.no_train_steps) + '/' + 'Step' +
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
    dataset = SummarizationDataset(test_file, use_keywords=args.use_keywords, use_switch=args.use_switch, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
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
    model = nn.DataParallel(model)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=1e-5)
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
        no_repeat_ngram_size=args.no_repeat_ngram_size,repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty)
        for pred in preds:
            f.write(tokenizer.decode(pred) + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-dataset', default='allCampusmin60max120summary_source_num5review_source_num60', type=str)
    parser.add_argument('-num_aspects', default=10, type=int)
    parser.add_argument('-model_name', default='T5_B4', type=str)
    # parser.add_argument('-load_model', default='/home/jade/untextsum/model/finishline/T5/T5dev_fin_5Pos_asp_10000/Rouge.best_step.7900_rouge.0.159', type=str)
    # parser.add_argument('-load_model', default='/media/jade/yi_Data/Data/MASmodel/T5_B4/Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight1600050000/Rouge.best_step.19400_rouge.0.147', type=str)
    parser.add_argument('-load_model', default=None, type=str)
    parser.add_argument('-train_file', default=None, type=str)
    parser.add_argument('-asp_dev_file', default='/home/jade/untextsum/data/\
allCampusmin60max120summary_source_num5review_source_num60\
/Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 \
document_f1.81.44_best_dataBCE_Sum_weight16000/dev_fin.jsonl', type=str)
    parser.add_argument('-asp_test_file', default='/home/jade/untextsum/data/\
allCampusmin60max120summary_source_num5review_source_num60/\
Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 \
document_f1.81.44_best_dataBCE_Sum_weight16000/test_fin.jsonl', type=str)
    parser.add_argument('-data_dir', default='/home/jade/untextsum/data', type=str)
    parser.add_argument('-mil_data_dir', default='Sentiment_filtered10_BCE_Sum_weightstep.10000 \
average_loss.613.18 sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000', type=str)
    parser.add_argument('-model_dir', default='/home/jade/untextsum/model/T5_campus', type=str)
    # parser.add_argument('-model_type', default='google/t5-v1_1-small', type=str)
    parser.add_argument('-model_type', default='t5-small', type=str)
    parser.add_argument('-use_keywords', default='output', type=str) # none, input, output
    parser.add_argument('-use_switch', default='output', type=str) # none, input, output
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
    parser.add_argument('-no_train_steps', default=10000, type=int)
    parser.add_argument('-no_warmup_steps', default=5000, type=int)
    parser.add_argument('-check_every', default=100, type=int)
    parser.add_argument('-ckpt_every', default=1000, type=int)
    parser.add_argument('-max_length', default=512, type=int)
    parser.add_argument('-min_target_length', default=50, type=int)
    parser.add_argument('-max_target_length', default=150, type=int)
    parser.add_argument('-num_beams', default=2, type=int)
    parser.add_argument('-no_repeat_ngram_size', default=2, type=int)
    parser.add_argument('-repetition_penalty', default=1, type=float)
    parser.add_argument('-length_penalty', default=1, type=float)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval-aspect':
        evaluate(args, 'aspect')
    elif args.mode == 'eval-multi':
        evaluate(args, 'multi')
    elif args.mode == 'eval-double':
        evaluate(args, 'double')
