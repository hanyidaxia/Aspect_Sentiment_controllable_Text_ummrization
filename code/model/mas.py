import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel as robert_encoder
# new_loss = nn.CrossEntropyLoss()
# pos_weight = torch.tensor([20.02, 3.89, 1.85, 5.68, 3.20, 1.16, 0.59, 1.855, 1.87, 2.318], dtype=torch.float64).cuda()
# pos_weight = torch.tensor([18.519, 3.531, 1.719, 5.168, 3.207, 1.0918, 0.5773, 1.855, 1.87, 2.318], dtype=torch.float64).cuda()
# pos_weight = torch.tensor([18.4872, 3.7189, 1.9146, 5.2818, 3.4055, 1.1269, 0.6156, 2.003, 1.985, 2.009], dtype=torch.float64).cuda()
pos_weight = torch.tensor([18.4872, 3.7189, 1.9146, 5.2818, 3.4055, 1.1269, 0.6156], dtype=torch.float64).cuda()


new_loss = nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight=pos_weight)
# new_loss = nn.BCELoss()
# new_loss = nn.MultiLabelMarginLoss()
# new_loss = nn.MultiLabelSoftMarginLoss()
# s = nn.Sigmoid()
relu = nn.ReLU()

class MAS(nn.Module):

    def __init__(self, args):
        super(MAS, self).__init__()
        self.args = args
        self.word_enc = robert_encoder.from_pretrained(args.model_type, return_dict=True)
        for p in self.word_enc.parameters():
            p.requires_grad = False
        self.word_linear = nn.Linear(args.model_dim, args.num_aspects, bias=False)
        self.word_key = nn.Parameter(torch.Tensor(args.model_dim))
        self.word_transform = nn.Linear(args.model_dim, args.model_dim)
        self.sent_key = nn.Parameter(torch.Tensor(args.model_dim))
        self.dropout = nn.Dropout(0.5)
        nn.init.normal_(self.word_key)
        nn.init.normal_(self.sent_key)



    def forward(self, x_BxSxT, y_true_BxC=None, step=None):
        B, S, T = x_BxSxT.size()
        # print('current data dimension is %s start a new batch bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' % str((B, S, T)))
        H = self.args.num_heads
        D = self.args.model_dim
        eps = -1e9
        # get word encodings
        x_BSxT = x_BxSxT.view(B*S, T)
        # print('first dimension transfer got (this transfer is B times S)%s' % str(x_BSxT.size()))
        x_mask_BSxT = torch.where(x_BSxT != 0, 1, 0)
        x_BSxTxD = self.word_enc(x_BSxT, x_mask_BSxT, output_hidden_states=True).last_hidden_state
        # print('last word_vec dimension transfer got %s' % str(x_BSxTxD.size()))
        if self.training:
            assert step is not None
            """assert = make sure this condition"""
            drop_rate = max(0.2 * (step - self.args.no_warmup_steps) / float(self.args.no_warmup_steps), 0)
            drop_rate = min(drop_rate, 0.2)
            drop_BSxTx1 = torch.rand(B*S, T, 1).cuda()
            drop_BSxTx1 = torch.where(drop_BSxTx1 > drop_rate, 1, 0)
            x_BSxTxD = x_BSxTxD * drop_BSxTx1
            # print('after drop step dimension transfer got %s' % str(x_BSxTxD.size()))
        x_BSxTxD = self.dropout(x_BSxTxD)
        # word-level predictions

        # w_BSxTxC = torch.tanh(self.word_linear(x_BSxTxD))
        w_BSxTxC = relu(self.word_linear(x_BSxTxD))

        # print('word dimension tanh and linear transfer got, This is Zt %s' % str(p_BSxTxC.size()))
        w_BxSxTxC = w_BSxTxC.view(B, S, T, -1)
        # print('word dimension transfer back to B, S, T size got, Also another format of Zt %s' % str(p_BxSxTxC.size()))

        # word-level representation/value
        # print('word-level dimension tanh transfer to value got, this step is getting the key for later calculation %s' % str(z_BSxTxD.size()))

        # z_key_BSxTxE = torch.tanh(self.word_transform(x_BSxTxD)).chunk(H, -1)
        z_key_BSxTxE = relu(self.word_transform(x_BSxTxD)).chunk(H, -1)

        # print('first dimension chunk transfer got 12 heads of chunks, last dimension/12, this is the key in paper, it has a tanh and linear process %s, the size is %s' % (str(np.shape(z_list_BSxTxE[0])), str(len(z_list_BSxTxE))))
        z_qry_list_E = self.word_key.chunk(H, -1)
        # print('dimension word_key chunk transfer got, function is nn.Parameter.chunk this is the query in paper, this vector has nothing to do with the data %s, the size is %s' % (str(np.shape(z_key_list_E[0])), str(len(z_key_list_E))))


        s_list_BxSxC = [] # the container takes zzzzzzhhhhhhh
        h_list_BxSxE = [] # the container takes zt element wise times head attention ah, which means the key followed the steps in the model
        p_wt_list_BxSxT = [] # the container takes aaaaaahhhhhh
        mini_step = 0
        for z_BSxTxE, z_qry_E in zip(z_key_BSxTxE, z_qry_list_E):
            a_BSxT = torch.matmul(z_BSxTxE, z_qry_E)
            # if mini_step <= 1:
            #     print('\n')
                # print('this step query z_key_E %s times key z_BSxTxE %s get aaaaaahhhhhh a_BSxT %s' %(str(np.shape(z_key_E)), str(np.shape(z_BSxTxE)), str(np.shape(a_BSxT))))
            a_BSxT = a_BSxT.masked_fill(x_mask_BSxT == 0, eps) #fill every o in the matrix as eps

            a_BSxT = F.softmax(a_BSxT, -1)
            # if mini_step <= 1:
            #     print('\n')
            #     print('this step aaaahhhh been softmaxed %s ' %(str(a_BSxT.size())))
            p_wt_list_BxSxT.append(a_BSxT.view(B, S, T))
            # if mini_step < 1:
            #     print('\n')
            #     print('after softmax each attention ah size is: %s, ah appened to a list p_wt_list_BxSxT size is: %s, each element inside is %s' % (str(a_BSxT.size()),str(np.shape(p_wt_list_BxSxT)), str(a_BSxT.view(B, S, T).size())))

              # sentence-level predictions
            s_BSxC = torch.sum(w_BSxTxC * a_BSxT.unsqueeze(-1), 1)
            # if mini_step < 1:
            #     print('start the sentence prediction ------------------------------------------------------------------------------------------')
            #     print('sum up the product of Zt %s and ah %s got q_BSxC which is zzzzzzzzhhhhhhhhhhh %s' % (str(p_BSxTxC.size()), str(a_BSxT.unsqueeze(-1).size()), str(q_BSxC.size())))
            s_list_BxSxC.append(s_BSxC.view(B, S, -1))
            # if mini_step < 1:
            #     print('append all the ZZZZZZZZZZZZZZZZZZZZhhhhhhhhhhhhhhhhhh to a list %s, inside the list each head prediction with size %s' % (len(q_list_BxSxC), str(q_list_BxSxC[0].size())))
            h_BSxE = torch.sum(z_BSxTxE * a_BSxT.unsqueeze(-1), 1)
            # if mini_step < 1:
                # print('\n')
                # print('Get encodings of sentence-level %s, its a sumup of key z_BSxTxE %s and ah last dimension unsqueeze %s' % (str(h_BSxE.size()), str(z_BSxTxE.size()), str(a_BSxT.unsqueeze(-1).size())))
                # print('this step perform a token-level likely encoding got h_BSxE size is: %s' % str(h_BSxE.size()))
            h_list_BxSxE.append(h_BSxE.view(B, S, -1))
            # if mini_step < 1:
            #     print('this step append all word query that has same calculation procedure got h_list_BxSxE size is: %s, one of the query wirh size %s' % (str(np.shape(h_list_BxSxE)), str(h_list_BxSxE[0].size())))
            # mini_step += 1
        # print('after the attention transfer, each updated sentence level key was calculated: %s' % str(np.shape(h_list_BxSxE)))
        # print('append all the ZZZZZZZZZZZZZZZZZZZZhhhhhhhhhhhhhhhhhh to a list %s, inside the list each head prediction with size %s' % (len(q_list_BxSxC), str(q_list_BxSxC[0].size())))
        # print('This step stack all the attention heads prediction Zh, : %s the length of this list is %s' % (str(q_BxSxHxC.size()), str(len(q_list_BxSxC))), q_BxSxHxC)
        s_BxSxC = torch.stack(s_list_BxSxC, -2).max(dim=-2)[0]
        # print('This step get max of all the attention heads prediction Zh which is the max-pooling step: ZZZZZZZZZZZZZZZZZSSSSSSSSSSSSSSSS get %s' % str(q_BxSxC.size()), q_BxSxC)
        p_wt_BxSxT = torch.stack(p_wt_list_BxSxT, -2).max(-2)[0]
        # print('this p_wt_BxSxT stack all the attention in a sentence toghther which has a size %s' % str(p_wt_BxSxT.size()))
        h_BxSxHxE = torch.stack(h_list_BxSxE, -2)
        # print('this h_BxSxHxE stack all the Zh in a sentence toghther which has a size %s' % str(h_BxSxHxE.size()))
        h_BxSxD = h_BxSxHxE.view(B, S, D)
        # print('this h_BxSxD stack all the h_BxSxHxE in a sentence toghther get sentence level key which has a size %s' % str(h_BxSxD.size()))


        # sentence-level attention weights
        x_mask_BxS = x_mask_BSxT.view(B, S, T).sum(dim=-1)
        x_mask_BxS = torch.where(x_mask_BxS != 0, 1, 0)
        h_list_BxSxE = h_BxSxD.chunk(H, -1)
        h_qry_list_E = self.sent_key.chunk(H, -1)
        y_list_BxC = []
        q_wt_list_BxS = []
        for h_BxSxE, h_qry_E in zip(h_list_BxSxE, h_qry_list_E):
            b_BxS = torch.matmul(h_BxSxE, h_qry_E)
            b_BxS = b_BxS.masked_fill(x_mask_BxS == 0, eps)
            b_BxS = F.softmax(b_BxS, -1)
            q_wt_list_BxS.append(b_BxS)
              # document-level predictions
            y_BxC = torch.sum(s_BxSxC * b_BxS.unsqueeze(-1), 1)
            y_list_BxC.append(y_BxC)
        # print('this is the final step get the y_BxHxC label which size is %s' % str(y_BxHxC.size()))
        y_BxC = torch.stack(y_list_BxC, -2).max(dim=-2)[0]
        # print('this is the final step get all the y_BxC label which size is %s' % str(y_BxC.size()))
        # print('document prediction output size %s' % str(y_BxC.size()), y_BxC)
        # miemie = []
        # if mini_step % 500 == 0:
        #     for pp in y_BxC.tolist():
        #         contains_aspect = np.any(len([1 for x in pp if x>0])>1)
        #         # print(pp)
        #         if not contains_aspect:
        #             miemie.append('True')
        #     print('how many aspect has not been detected at all from the results >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  %d' % len(miemie))
        #     mini_step += 1
        q_wt_BxS = torch.stack(q_wt_list_BxS, -2).max(-2)[0]
        # print(' the q_wt_BxS size is %s' % str(q_wt_BxS.size()))


        if y_true_BxC is not None:
            # y_true_BxC = torch.div((y_true_BxC + 1), 2) # turn the label range from (-1, 1) to (0, 1)
            # loss_BxC = torch.log(1 + torch.exp(-y_BxC * y_true_BxC))
            # print('y_BxC size is %s' % str(y_BxC.size()))
            loss_BxC = new_loss(y_BxC, y_true_BxC)
            loss = loss_BxC.sum(dim=-1).mean()
        else:
            loss = None


        return {
          'document': y_BxC,
          'sentence': s_BxSxC,
          'word': w_BxSxTxC,
          'loss': loss,
          'sentence_weight': q_wt_BxS,
          'word_weight': p_wt_BxSxT,
        }
