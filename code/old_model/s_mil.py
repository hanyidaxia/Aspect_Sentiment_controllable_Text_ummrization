import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel as robert_encoder


class MIL(nn.Module):

    def __init__(self, args model_type, model_dim, num_aspects, num_heads, no_warmup_steps):
        super(MIL, self).__init__()
        self.word_enc = robert_encoder.from_pretrained(model_type, return_dict=True)
        for p in self.word_enc.parameters():
            p.requires_grad = False
        self.word_linear = nn.Linear(model_dim, num_aspects, bias=False)



        self.word_key = nn.Parameter(torch.Tensor(model_dim))



        self.word_transform = nn.Linear(model_dim, model_dim)

        self.sent_key = nn.Parameter(torch.Tensor(model_dim))
        self.dropout = nn.Dropout(0.5)
        nn.init.normal_(self.word_key)
        nn.init.normal_(self.sent_key)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.no_warmup_steps = no_warmup_steps




    def forward(self, x_BxSxT, y_true_BxC=None, p_true_BxSxTxC=None, step=None):
        B, S, T = x_BxSxT.size()
        print('current data dimension is %s' % str((B, S, T)))
        H = self.num_heads
        D = self.model_dim
        E = D // H
        eps = -1e9
        # get word encodings
        x_BSxT = x_BxSxT.view(B*S, T)
        print('first dimension transfer got %s' % str(x_BSxT.size()))
        x_mask_BSxT = torch.where(x_BSxT != 0, 1, 0)
        print('mask dimension transfer got %s' % str(x_mask_BSxT.size()))
        x_BSxTxD = self.word_enc(x_BSxT, x_mask_BSxT, output_hidden_states=True).last_hidden_state
        print('last word_vec dimension transfer got %s' % str(x_BSxTxD.size()))
        if self.training:
            assert step is not None
            drop_rate = max(0.2 * (step - self.no_warmup_steps) / float(self.no_warmup_steps), 0)
            drop_rate = min(drop_rate, 0.2)
            drop_BSxTx1 = torch.rand(B*S, T, 1).cuda()
            drop_BSxTx1 = torch.where(drop_BSxTx1 > drop_rate, 1, 0)
            x_BSxTxD = x_BSxTxD * drop_BSxTx1
            print('after drop step dimension transfer got %s' % str(x_BSxTxD.size()))

        x_BSxTxD = self.dropout(x_BSxTxD)

        # word-level predictions
        p_BSxTxC = torch.tanh(self.word_linear(x_BSxTxD)) #in the paper, this was called Zt
        print('this step got the word level predictions for each sentence ----------------------------------------------------------------------')
        print('word dimension tanh and linear transfer got, This is Zt %s' % str(p_BSxTxC.size()))
        p_BxSxTxC = p_BSxTxC.view(B, S, T, -1)
        print('word dimension transfer back to B, S, T size got, Also another format of Zt %s' % str(p_BxSxTxC.size()))

        # word-level representation/value
        z_BSxTxD = torch.tanh(self.word_transform(x_BSxTxD))
        print('word-level dimension tanh transfer to value got, this step is getting the key for later calculation %s' % str(z_BSxTxD.size()))
        z_list_BSxTxE = z_BSxTxD.chunk(H, -1) # thie is the key in the paper
        print('first dimension chunk transfer got 12 heads of chunks, this is the key in paper %s' % str(np.shape(z_list_BSxTxE)))
        z_key_list_E = self.word_key.chunk(H, -1) # this is the query in the paper
        print('dimension word_key chunk transfer got, function is nn.Parameter.chunk this is the query in paper, this vector has nothing to do with the data %s' %               str(np.shape(z_key_list_E)))
        print('word level encoding done ------------------------------------------------------------------------------------------------------------')
        print('\n')


        q_list_BxSxC = []
        h_list_BxSxE = []
        p_wt_list_BxSxT = []
        mini_step = 0
        for z_BSxTxE, z_key_E in zip(z_list_BSxTxE, z_key_list_E):
            a_BSxT = torch.matmul(z_BSxTxE, z_key_E) # multiple predictions Zh for each attention head in the paper
            if mini_step <= 2:
                print('\n')
                print('in the previous key list and query list, each query element size is: %s' % str(z_BSxTxE.size()))
                print('in the previous key list and query list, each key element size is: %s' % str(z_key_E.size()))
                print('in the previous attention head, each dot product of ah size is: %s' % str(a_BSxT.size()))
            a_BSxT = a_BSxT.masked_fill(x_mask_BSxT == 0, eps)
#             if mini_step <= 2:
#                 print('\n')
#                 print('in the previous key list and query list, each query element size is: %s' % str(z_BSxTxE.size()))
#                 print('in the previous key list and query list, each key element size is: %s' % str(z_key_E.size()))
#                 print('in the previous attention head, each dot product of ah size is: %s' % str(a_BSxT.size()))
            a_BSxT = F.softmax(a_BSxT, -1) # in the paper, this is the attention ah
            if mini_step <= 2:
                print('\n')
                print('after softmax each attention ah size is: %s' % str(a_BSxT.size()))
                print('ah transform to a size is: %s' % str(a_BSxT.view(B, S, T).size()))

            p_wt_list_BxSxT.append(a_BSxT.view(B, S, T))


              # sentence-level predictions
            q_BSxC = torch.sum(p_BSxTxC * a_BSxT.unsqueeze(-1), 1)
            if mini_step <= 2:
                print('start the sentence prediction ------------------------------------------------------------------------------------------------')
                print('\n')
                print('maxpooling for sentence level prediction P_BSxTxC which is each token level prediction times each head\
                      prediction, this step is the sumup step in the paper: %s' % str(a_BSxT.unsqueeze(-1).size()))
                print(p_BSxTxC.size())
                print('after maxpooling the prediction of each attention Zs, its product of tensor (B, S, T, label number) and\
                      (BxS, S)size is: %s' % str(q_BSxC.size()))
                print('append each sentence prediction to a list, each sentence prediction is stored as a size %s' % str(q_BSxC.size()))
            q_list_BxSxC.append(q_BSxC.view(B, S, -1))

              # sentence-level encodings
            h_BSxE = torch.sum(z_BSxTxE * a_BSxT.unsqueeze(-1), 1)
            if mini_step <= 2:
                print('\n')
                print('this step perform a token-level likely encoding got h_BSxE size is: %s' % str(h_BSxE.size()))

            h_list_BxSxE.append(h_BSxE.view(B, S, -1))
            mini_step += 1


        print('after the attention transfer, each head prediction was calculated: %s' % str(np.shape(h_list_BxSxE)))



        q_BxSxHxC = torch.stack(q_list_BxSxC, -2)
        q_BxSxC = q_BxSxHxC.max(dim=-2)[0]

        p_wt_BxSxT = torch.stack(p_wt_list_BxSxT, -2).max(-2)[0]

        h_BxSxHxE = torch.stack(h_list_BxSxE, -2)
        h_BxSxD = h_BxSxHxE.view(B, S, D)

        # sentence-level attention weights
        x_mask_BxS = x_mask_BSxT.view(B, S, T).sum(dim=-1)
        x_mask_BxS = torch.where(x_mask_BxS != 0, 1, 0)
        h_list_BxSxE = h_BxSxD.chunk(H, -1)
        h_key_list_E = self.sent_key.chunk(H, -1)

        y_list_BxC = []
        q_wt_list_BxS = []
        for h_BxSxE, h_key_E in zip(h_list_BxSxE, h_key_list_E):
            b_BxS = torch.matmul(h_BxSxE, h_key_E)
            b_BxS = b_BxS.masked_fill(x_mask_BxS == 0, eps)
            b_BxS = F.softmax(b_BxS, -1)
            q_wt_list_BxS.append(b_BxS)

              # document-level predictions
            y_BxC = torch.sum(q_BxSxC * b_BxS.unsqueeze(-1), 1)
            y_list_BxC.append(y_BxC)

        y_BxHxC = torch.stack(y_list_BxC, -2)
        y_BxC = y_BxHxC.max(dim=-2)[0]

        q_wt_BxS = torch.stack(q_wt_list_BxS, -2).max(-2)[0]

        if y_true_BxC is not None:
            eps = 1e-9
            loss_BxC = torch.log(1 + torch.exp(-y_BxC * y_true_BxC))
            loss = loss_BxC.sum(dim=-1).mean()
        else:
            loss = None

        if p_true_BxSxTxC is not None:
            p_true_mask_BxSxTxC = torch.where(p_true_BxSxTxC != 0, 1, 0)
            reg_loss_BxSxTxC = torch.log(1 + torch.exp(-p_BxSxTxC * p_true_BxSxTxC)) * p_true_mask_BxSxTxC
            reg_loss = reg_loss_BxSxTxC.view(B, -1).sum(dim=-1).mean()
        else:
            reg_loss = None

        return {
          'document': y_BxC,
          'sentence': q_BxSxC,
          'word': p_BxSxTxC,
          'loss': loss,
          'reg_loss': reg_loss,
          'sentence_weight': q_wt_BxS,
          'word_weight': p_wt_BxSxT,
        }
