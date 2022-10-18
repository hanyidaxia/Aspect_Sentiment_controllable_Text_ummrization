import json
import numpy as np
import torch
import argparse
import itertools

def split_data(file_dir, args):
    if args.no_vali:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic), type(man_dic[0]))
        train_size = int(0.85 * len(man_dic))
        test_size = len(man_dic) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, test_size])
        print(type(test_dataset[0]))



        with open (file_dir + args.result_data_name , 'w') as f:
            for i in test_dataset:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.data_name , 'w') as f:
            for i in train_dataset:
                f.write(json.dumps(i) + '\n')
    else:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic), type(man_dic[0]))


        train_size = int(0.7 * len(man_dic))
        test_size = int((len(man_dic) - train_size)/2)
        vali_size = int(len(man_dic) - train_size - test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, vali_size, test_size])

        print(type(test_dataset[0]))


        with open (file_dir + args.result_data_name , 'w') as f:
            for i in test_dataset:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.vali_data_name , 'w') as f:
            for i in validation_dataset:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.data_name , 'w') as f:
            for i in train_dataset:
                f.write(json.dumps(i) + '\n')

def split_train_data(file_dir, args):

    if args.no_vali:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic), type(man_dic[0]))


        train_size = int(0.85 * len(man_dic))
        test_size = len(man_dic) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, test_size])

        print(type(test_dataset[0]))



        with open (file_dir + args.result_data_name , 'w') as f:
            for i in test_dataset:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.data_name , 'w') as f:
            for i in train_dataset:
                f.write(json.dumps(i) + '\n')
    else:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic))



        train_size = int(0.7 * len(man_dic))
        test_size = int((len(man_dic) - train_size)/2)
        vali_size = int(len(man_dic) - train_size - test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, vali_size, test_size])


        new_train = []
        for i in validation_dataset:
            tmp_key = {}
            for k, v in i.items():
                # if np.amax(i['switch'][-3:]) > 0:
                #     print('correct')
                # print(k, v)
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
                tmp_2 = [0, 0, 0]
                if k == 'switch':
                    tmp_a = v[-3:]
                    tmp_1 = list(np.argsort(tmp_a))
                    if tmp_1[-1] == 1 and tmp_a[1] <= 0.02:
                            tmp_2[-1] = 1
                    else:
                        tmp_2[tmp_1[-1]] = 1
                    tmp_key['switch'][-3:] = tmp_2
            new_train.append(tmp_key)


        new_vali = []
        for i in validation_dataset:
            tmp_key = {}
            for k, v in i.items():
                # if np.amax(i['switch'][-3:]) > 0:
                #     print('correct')
                # print(k, v)
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
                tmp_2 = [0, 0, 0]
                if k == 'switch':
                    # print(v)
                    tmp_a = v[-3:]
                    # print(tmp_a)
                    tmp_1 = list(np.argsort(tmp_a))
                    if tmp_1[-1] == 1 and tmp_a[1] <= 0.02:
                            tmp_2[-1] = 1
                    else:
                        tmp_2[tmp_1[-1]] = 1
                    tmp_key['switch'][-3:] = tmp_2
            new_vali.append(tmp_key)

        new_test = []
        for i in test_dataset:
            tmp_key = {}
            for k, v in i.items():
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
                tmp_2 = [0, 0, 0]
                if k == 'switch':
                    tmp_1 = list(np.argsort(v[-3:]))
                    if tmp_1[-1] == 1 and tmp_a[1] <= 0.02:
                            tmp_2[-1] = 1
                    else:
                        tmp_2[tmp_1[-1]] = 1
                    tmp_key['switch'][-3:] = tmp_2
            new_test.append(tmp_key)

        print(type(test_dataset[0]))


        with open (file_dir + args.result_data_name , 'w') as f:
            for i in new_test:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.vali_data_name , 'w') as f:
            for i in new_vali:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.data_name , 'w') as f:
            for i in new_train:
                f.write(json.dumps(i) + '\n')



def re_split_train_data(file_dir, args):

    if args.no_vali:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic), type(man_dic[0]))
        train_size = int(0.85 * len(man_dic))
        test_size = len(man_dic) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, test_size])
        print(type(test_dataset[0]))
        with open (file_dir + args.result_data_name , 'w') as f:
            for i in test_dataset:
                f.write(json.dumps(i) + '\n')
        with open (file_dir + args.data_name , 'w') as f:
            for i in train_dataset:
                f.write(json.dumps(i) + '\n')
    else:
        man_dic = []
        with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                man_dic.append(inst)
        print(len(man_dic))
        train_size = int(0.7 * len(man_dic))
        test_size = int((len(man_dic) - train_size)/2)
        vali_size = int(len(man_dic) - train_size - test_size)
        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(man_dic, [train_size, vali_size, test_size])
        new_train = []
        for i in validation_dataset:
            tmp_key = {}
            for k, v in i.items():
                # if np.amax(i['switch'][-3:]) > 0:
                #     print('correct')
                # print(k, v)
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
            new_train.append(tmp_key)


        new_vali = []
        for i in validation_dataset:
            tmp_key = {}
            for k, v in i.items():
                # if np.amax(i['switch'][-3:]) > 0:
                #     print('correct')
                # print(k, v)
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
            new_vali.append(tmp_key)

        new_test = []
        for i in test_dataset:
            tmp_key = {}
            for k, v in i.items():
                tmp_key['summary'] = i['summary']
                tmp_key['reviews'] = i['reviews']
                tmp_key['keywords'] = i['keywords']
                tmp_key['switch'] = i['switch']
            new_test.append(tmp_key)

        print(type(test_dataset[0]))

        with open (file_dir + args.result_data_name , 'w') as f:
            for i in new_test:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.vali_data_name , 'w') as f:
            for i in new_vali:
                f.write(json.dumps(i) + '\n')

        with open (file_dir + args.data_name , 'w') as f:
            for i in new_train:
                f.write(json.dumps(i) + '\n')



def deal_re (file_dir, args):
    man_dic = []
    with open (file_dir + args.data_name,  'r', encoding="UTF-8") as f:
        for line in f:
            inst = json.loads(line)
            man_dic.append(inst)
    print(len(man_dic), type(man_dic[0]))
    re_list = []

    for i in man_dic:
        tmp_dic = {}
        tmp_dic['review'] = i['review']
        tmp_dic['aspects'] = dict(itertools.islice(i['aspects'].items(), 7))
        re_list.append(tmp_dic)

    with open (file_dir + args.result_data_name , 'w') as f:
        for i in re_list:
            f.write(json.dumps(i) + '\n')







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', default='train_sum.jsonl', type=str)
    parser.add_argument('-no_vali', default=False, type=bool)


    args = parser.parse_args()
    if args.data_name == 'S_7_5_man.json':
        parser.add_argument('-result_data_name', default='dev_S_7_5_man.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)
    if args.data_name == 'S_7_5_woman.json':
        parser.add_argument('-result_data_name', default='dev_S_7_5_woman.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)
    if args.data_name == 'NB_S_woman.json':
        parser.add_argument('-result_data_name', default='dev_NB_S_woman.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)
    if args.data_name == 'NB_S_man.json':
        parser.add_argument('-result_data_name', default='dev_NB_S_man.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)
    if args.data_name == 'ascis_S_woman.json':
        parser.add_argument('-result_data_name', default='S_7_5_man.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)
    if args.data_name == 'ascis_S_man.json':
        parser.add_argument('-result_data_name', default='S_7_5_man.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)
    if args.data_name == 'S_7_5_finishline.json':
        parser.add_argument('-result_data_name', default='test_S_7_5_finishline.json', type=str)
        args = parser.parse_args()
        split_data('/home/jade/untextsum/data/finishline/split/', args)
    if args.data_name == 'S_7_5_finishline_neg.json':
        parser.add_argument('-result_data_name', default='dev_S_7_5_finishline_neg.json', type=str)
        args = parser.parse_args()
        split_data('/home/jade/untextsum/data/finishline/', args)
    if args.data_name == 'NB_S.json':
        parser.add_argument('-result_data_name', default='dev_NB_S.json', type=str)
        args = parser.parse_args()
        split_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'train_sum.jsonl':
        parser.add_argument('-result_data_name', default='test_fin.jsonl', type=str)
        parser.add_argument('-vali_data_name', default='dev_fin.jsonl', type=str)
        args = parser.parse_args()
        re_split_train_data('/home/jade/untextsum/data/\
allCampusmin60max120summary_source_num5review_source_num60keywords5\
/Sentiment_filtered7_BCE_Sum_weightstep.24000 average_loss.421.54 \
sentence_f1.71.16 document_f1.74.65_re_bestBCE_Sum_weight30000/', args)

    if args.data_name in ['best_data.jsonl','dev_best.jsonl']:
        '''
        Following three lines were the original training dataset obtaining method
        '''
        # parser.add_argument('-result_data_name', default='dev_best.jsonl', type=str)
        # args = parser.parse_args()
        # split_data('/media/jade/yi_Data/Data/New_Data/Text_data/all/', args)
        if args.data_name == 'best_data.jsonl':
            parser.add_argument('-result_data_name', default='re_best.jsonl', type=str)
            args = parser.parse_args()
            deal_re('/home/jade/untextsum/data/all/', args)
        elif args.data_name == 'dev_best.jsonl':
            parser.add_argument('-result_data_name', default='redev_best.jsonl', type=str)
            args = parser.parse_args()
            deal_re('/home/jade/untextsum/data/all/', args)
