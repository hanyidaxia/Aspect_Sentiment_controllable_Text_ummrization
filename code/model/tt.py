from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
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
new_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
# import torch. as sigmoid
from transformers import get_cosine_schedule_with_warmup
# from mil import MIL
from mas import MAS
from torch.autograd import Variable
import time
sys.path.append(os.path.realpath('..'))
import random
from data_processor.readers import AspectDetectionDataset, aspect_detection_collate

s = nn.Sigmoid()





m = list(range(20))
mm = np.argsort(m)
print(mm)

print(m)
x = range(0, 20, 3)
for i in x:
    print(i)
for n in x:
  print(m[n+3])



























# target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
# output = torch.full([10, 64], 1.5)  # A prediction (logit)
# pos_weight = torch.ones([64])  # All weights are equal to 1
# print(output, target)
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# aa= criterion(output, target)  # -log(sigmoid(1.5))
# print(aa)

# loss = nn.BCEWithLogitsLoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# print(input, target)
# output = loss(input, target)
# print(output)
# output.backward()

tmp_sentiment_count = [1 for x in [135,354,345] if x>0]
a = np.any(len(tmp_sentiment_count)>0)
print(a)













doc_counts = [[0] * 2] * 10
print('doc_counts is', doc_counts)

print(random.randint(0, 64))

y1 = torch.randn(8, 5)
print(y1)
y11 = s(y1).float()
y111 = 1 - y11
print('y11 is', y11)

y3 = y1.max(dim=-2)[0]
y4 = torch.topk(y1, 3)
print(y3)
print('y4 is', y4)

y2 = torch.randint(-1, 1, (8, 5))
print('y2 is', y2)
y2 = torch.where(y2 == 0, -1, 1).float()

print('y1, y2 is ', y1, y2)

nloss = new_loss(y1, y2)
print('new_loss is', nloss)
loss = ((y2*torch.log(s(y1)))+((1 - y2)*torch.log(1-s(y1))))
print(loss)


loss_BxC = torch.log(1 + torch.exp(-y1 * y2))
# print('y_BxC size is %s' % str(y_BxC.size()))
# loss_BxC = new_loss((y_BxC), y_true_BxC)
loss = loss_BxC.sum(dim=-1).mean()
print('the loss is ', loss)






a = torch.randint(10, (2,))
print(a, a.size())
print(a.max(0), a.max(-1))
b = torch.randint(10, (2, 3, 4))
print(b, b.size())
lin = nn.Linear(4, 2)
b = b.view(6, 4).float()
print(b)
c = lin(b)
print(c, c.size())
b = torch.where(b != 0, 1, 0)
print(b)
c = c.view(2, 3, 2, -1)
print(c, c.size())
print(b.max(dim = -2)[0])

d = nn.Parameter(torch.Tensor(16))
print(d, d.size())






# meet = []
#
# target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
# output = torch.full([10, 64], 1.5)  # A prediction (logit)
# pos_weight = torch.ones([64])  # All weights are equal to 1
# print(target.size(), output.size(), target, output)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# a= criterion(output, target)  # -log(sigmoid(1.5))
# print(a)
#
#
# loss = nn.BCEWithLogitsLoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(input, target)
# output.backward()








# with open ('/home/jade/untextsum/data/finishline/S_7_5_finishline.json', 'r') as f:
#     for line in f:
#         inst = json.loads(line.strip())
#         meet.append(inst)
#
# meet2 = []
# neg = {}
# for ele in meet:
#     if ele['aspects']['Neg'] == 'yes':
#         meet2.append(ele)
#
#
# f = open('/home/jade/untextsum/data/finishline/S_7_5_finishline_neg.json', 'w')
# for i in meet2:
#     # print(i)
#     f.write(json.dumps(i) + '\n')
# f.close()




# a = []
# b = np.random.randint(2, 5, (3,))
# print(np.product(b))
# print(b)
# b = np.append(b, [0])
# print(b)
# c = np.random.randint(5, 10, (3,))
# b[b==0] = -100
# print(b, c)
#
# a += list(b)
# a += list(c)
# print(a)
#
#
#
#
#
#
#
#
# a = np.random.randint(-5, -2, (5,))
#
# print(a, a[-3:], a[:3])
#
# print(a, np.maximum(0, a))



# print('-------------------')
#
# print(np.maximum(0, a.cpu().detach().numpy()))
#
# m_l = [a.cpu().detach().numpy(), b.cpu().detach().numpy()]
#
# print(np.max(m_l, axis=0))
#
# # pp = torch.stack(m_l, -2)
#
# print(pp, pp.size())
#
# print(pp.max(dim=0))
