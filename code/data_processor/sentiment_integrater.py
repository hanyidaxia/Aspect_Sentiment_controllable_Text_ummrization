import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')
import argparse
import json
from tqdm import tqdm
import copy
import random
"""the multi-instance model requires a review, aspect pair input format, this code serves this purpose"""



def split_sentence(review):
    text = [str(i) for i in list(nlp(review).sents)]
    return text


def create_data(filedir, args):

    with open (args.attribute_lexicon_name, 'r') as f:
        lexicon = json.load(f)
    aspects = list(lexicon.keys()) # + ['general']

    instance_dict = {}
    with open(filedir + args.data_name, 'r') as f:
#     for line in tqdm(f):
        inst = json.load(f)
        print('origin data length is %d' % len(inst))
        # domain = args.domain_name
        # if domain not in instance_dict:
        # instance_dict = {}

        reviews = []
        for i, j in tqdm(inst.items()):
            for idx, sub_j in enumerate(j):
                review_dic = {}
                if i + str(idx) not in review_dic:
                    review_dic['text'] = (split_sentence(sub_j['text']))
                    review_dic['rate'] = (sub_j['rate'])
                    reviews.append(review_dic) # should be in a format [{text:.... rate:......}, {}, {}]

        for t_review in reviews:
            # print(t_review)
          # sanity check
            listfy_review = copy.deepcopy(t_review['text'])
            if len(t_review['text']) > 35 or len(t_review['text']) < 2:
                continue
            if max([len(sentence.split()) for sentence in t_review['text']]) > 50:
                continue
            if min([len(sentence.split()) for sentence in t_review['text']]) < 5:
                continue
            review = ' '.join(t_review['text']).split()

              # check whether aspect keywords in review
            class_list = []
            for aspect in aspects:
                keywords = lexicon[aspect]
                includes = int(any([keyword in review for keyword in keywords]))
                class_list.append(includes)
                class_list.append()
            # print(class_list)
            assert len(class_list) == 7
            if t_review['rate'] == "5":
                class_list += [1, 0, 0]
            elif t_review['rate'] == "3" or t_review['rate'] == "4":
                class_list += [0, 1, 0]
            elif t_review['rate'] == "1" or t_review['rate'] == '2':
                class_list += [0, 0, 1]
      # add general class
      # if any(class_list):
      #   class_list.append(0)
      # else:
      #   class_list.append(1)
      # add review to corresponding aspect buckets
            instance_tuple = (listfy_review, class_list)
            class_list = tuple(class_list)
            if class_list not in instance_dict:
                instance_dict[class_list] = []
            #     print('domain')
            instance_dict[class_list].append(instance_tuple)
    # print(instance_dict)
    # for domain in instance_dict:
        # print('domain', domain)
    lengths = [len(instance_dict[key]) for key in instance_dict]
    print('in the prepared instance_dict, keys are %s  and lengths are %s' % (str(instance_dict.keys()), str(lengths)))
    # min_length = sorted(lengths)[1]
    # print('min_length is', min_length)
    for i in range(len(aspects)):
        c = [0] * len(aspects)
        c[i] = 1
        print(c)
        print(len(instance_dict[tuple(c)]))
    # print('mininum instances per tuple', min_length)
    data = []
    for key in instance_dict:
        if key == (0, 0, 0, 0, 0, 0, 0):
            data += (instance_dict[key][:1000])
        else:
            instances = instance_dict[key]
            random.shuffle(instances)
            if len(instances) >= 10:
                data += instances

    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0
    domain_aspects = aspects

    with open(filedir + args.result_data_name, 'w') as f:
        count_dict = {aspect:0 for aspect in domain_aspects}
        for inst in data:
            # print(inst)
            new_inst = {}
            new_inst['review'] = inst[0]
            max_text_length = max(max_text_length, len(inst[0]))
            class_dict = {}

            for i, aspect in enumerate(domain_aspects):
                class_dict[aspect] = 'yes' if inst[1][i] else 'no'
                if inst[1][i]:
                    count_dict[aspect] += 1
            new_inst['aspects'] = class_dict
            f.write(json.dumps(new_inst) + '\n')

    print('max text length', max_text_length)
    print(count_dict)


def create_sythe_data(filedir, args):

    with open (args.attribute_lexicon_name, 'r') as f:
        lexicon = json.load(f)
    aspects = list(lexicon.keys()) # + ['general']

    instance_dict = {}
    with open(filedir + args.data_name, 'r') as f:
#     for line in tqdm(f):
        inst = json.load(f)
        print('origin data length is %d' % len(inst))
        # domain = args.domain_name
        # if domain not in instance_dict:
        # instance_dict = {}

        reviews = []
        for i, j in tqdm(inst.items()):
            for idx, sub_j in enumerate(j):
                review_dic = {}
                if i + str(idx) not in review_dic:
                    review_dic['text'] = (split_sentence(sub_j['text']))
                    review_dic['rate'] = (sub_j['rate'])
                    reviews.append(review_dic) # should be in a format [{text:.... rate:......}, {}, {}]


        for t_review in reviews :
            # print(t_review)
          # sanity check
            listfy_review = copy.deepcopy(t_review['text'])
            if len(t_review['text']) > 35 or len(t_review['text']) < 2:
                continue
            if max([len(sentence.split()) for sentence in t_review['text']]) > 50:
                continue
            if min([len(sentence.split()) for sentence in t_review['text']]) < 5:
                continue
            review = ' '.join(t_review['text']).split()

              # check whether aspect keywords in review
            class_list = []
            for aspect in aspects:
                keywords = lexicon[aspect]
                includes = int(any([keyword in review for keyword in keywords]))
                class_list.append(includes)
            # print(class_list)
            assert len(class_list) == 7
            if t_review['rate'] == "5" or t_review['rate'] == "4":
                class_list += [1, 0, 0]
            elif t_review['rate'] == "3":
                class_list += [0, 1, 0]
            elif t_review['rate'] == "1" or t_review['rate'] == '2':
                class_list += [0, 0, 1]

      # add general class
      # if any(class_list):
      #   class_list.append(0)
      # else:
      #   class_list.append(1)

      # add review to corresponding aspect buckets
            instance_tuple = (listfy_review, class_list)
            # print(instance_tuple)
            class_list = tuple(class_list)
            if class_list not in instance_dict:
                instance_dict[class_list] = []
            #     print('domain')
            instance_dict[class_list].append(instance_tuple)

    # print(instance_dict)
    # for domain in instance_dict:
        # print('domain', domain)
    lengths = [len(instance_dict[key]) for key in instance_dict]
    print('in the prepared instance_dict, keys are %s  and lengths are %s' % (str(instance_dict.keys()), str(lengths)))
    min_length = sorted(lengths)[1]
    print('min_length is', min_length)
    # for i,j in instance_dict.items():
    #     # c = [0] * (len(aspects)+3)
    #     # c[i] = 1
    #     # c[-1] = 1
    #     # print(tuple(i))
    #     print(len(instance_dict[tuple(i)]))
    # print('mininum instances per tuple', min_length)
    data = []
    for key in instance_dict:
        if key[:7] == (0, 0, 0, 0, 0, 0, 0):
            data += (instance_dict[key][:args.threshold])
        else:
            instances = instance_dict[key]
            random.shuffle(instances)
            # if len(instances) >= 10:
            data += instances

    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0
    domain_aspects = aspects
    aspects += ['Pos', 'Neu', 'Neg']

    with open(filedir + args.result_data_name, 'w') as f:
        count_dict = {aspect:0 for aspect in domain_aspects}
        for inst in data:
            # print(inst)
            new_inst = {}
            new_inst['review'] = inst[0]
            max_text_length = max(max_text_length, len(inst[0]))
            class_dict = {}

            for i, aspect in enumerate(domain_aspects):
                class_dict[aspect] = 'yes' if inst[1][i] else 'no'
                if inst[1][i]:
                    count_dict[aspect] += 1
            new_inst['aspects'] = class_dict
            f.write(json.dumps(new_inst) + '\n')

    print('max text length', max_text_length)
    print('Write done')

def create_good_data(filedir, args):
    with open (args.attribute_lexicon_name, 'r') as f:
        lexicon = json.load(f)
    aspects = list(lexicon.keys()) # + ['general']
    instance_dict = {}
    reviews = []
    with open(filedir + args.data_name, 'r') as f:
        for line in f:
            inst = json.loads(line)
            t_inst = {k:v for k, v in inst.items() if k != 'attribute'}
            # tt_inst = {k:split_sentence(v) for k, v in tqdm(inst.items()) if k != 'text'}
            reviews.append(t_inst) # should be in a format [{text:.... rate:......}, {}, {}]
    for t_review in reviews :
        listfy_review = copy.deepcopy(t_review['text'])
        review = t_review['text']
        class_list = []
        for aspect in aspects:
            keywords = lexicon[aspect]
            includes = int(any([keyword in review for keyword in keywords]))
            class_list.append(includes)
        assert len(class_list) == 7
        if t_review['rate'] == "5":
            class_list += [1, 0, 0]
        elif t_review['rate'] == "3" or t_review['rate'] == "4":
            class_list += [0, 1, 0]
        elif t_review['rate'] == "1" or t_review['rate'] == '2':
            class_list += [0, 0, 1]
        instance_tuple = (listfy_review, class_list)
        class_list = tuple(class_list)
        if class_list not in instance_dict:
            instance_dict[class_list] = []
        instance_dict[class_list].append(instance_tuple)
    lengths = [len(instance_dict[key]) for key in instance_dict]
    print('in the prepared instance_dict, keys are %s  and lengths are %s' % (str(instance_dict.keys()), str(lengths)))
    min_length = sorted(lengths)[1]
    data = []
    for key in instance_dict:
        instances = instance_dict[key]
        random.shuffle(instances)
        data += instances
    print('total data', len(data))
    random.shuffle(data)
    max_text_length = 0
    domain_aspects = aspects
    aspects += ['Pos', 'Neu', 'Neg']
    with open(filedir + args.result_data_name, 'w') as f:
        count_dict = {aspect:0 for aspect in domain_aspects}
        for inst in tqdm(data):
            new_inst = {}
            new_inst['review'] = split_sentence(inst[0])
            max_text_length = max(max_text_length, len(inst[0]))
            class_dict = {}
            for i, aspect in enumerate(domain_aspects):
                class_dict[aspect] = 'yes' if inst[1][i] else 'no'
                if inst[1][i]:
                    count_dict[aspect] += 1
            new_inst['aspects'] = class_dict
            f.write(json.dumps(new_inst) + '\n')
    print('max text length', max_text_length)
    print('Write done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', type = str)
    parser.add_argument('-result_data_name', type = str)
    parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)
    # parser.add_argument('-domain_name', default='finishline', type = str)

    args = parser.parse_args()
    if args.data_name == 'NB_man.json':
        parser.add_argument('-threshold', default=300, type=int)
        args = parser.parse_args()
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'NB_woman.json':
        parser.add_argument('-threshold', default=300, type=int)
        args = parser.parse_args()
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'man.json':
        parser.add_argument('-threshold', default=700, type=int)
        args = parser.parse_args()
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'woman.json':
        parser.add_argument('-threshold', default=700, type=int)
        args = parser.parse_args()
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'ascis_man.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)

    if args.data_name == 'ascis_woman.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)


    if args.data_name == 'ran_balan.jsonl':
        args = parser.parse_args()
        print(args)
        create_good_data('/media/jade/yi_Data/Data/New_Data/Text_data/', args)

    if args.data_name == 'good_rest.jsonl':
        args = parser.parse_args()
        print(args)
        create_good_data('/media/jade/yi_Data/Data/New_Data/Text_data/', args)
