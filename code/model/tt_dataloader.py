import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.realpath('..'))
from data_processor.re_readers import AspectDetectionDataset, aspect_detection_collate, SummarizationDataset
import argparse
import tqdm
from transformers import AutoTokenizer
import datetime

e = datetime.datetime.now()

print(e.month, type(e.month),
        e.day, type(e.day),
        e.hour, type(e.hour))






# dataset = SummarizationDataset('/home/jade/untextsum/data/all/\
# Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 \
# sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000/dev_fin.jsonl',
# use_keywords='input', use_switch='input')
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
dataset = AspectDetectionDataset('/home/jade/untextsum/data/all/re_best.jsonl', tokenizer=tokenizer,
shuffle=True)

dataloader = DataLoader(dataset, batch_size=12, collate_fn=aspect_detection_collate)
print(type(dataloader))
for inp_batch, out_batch in enumerate(list(dataloader)):
    print(inp_batch, out_batch)
    break
print('what is the curren input and output')
