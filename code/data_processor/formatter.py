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
            for sub_j in j:
                reviews.append(split_sentence(sub_j['text']))


        for t_review in reviews :
            # print(t_review)
          # sanity check
            listfy_review = copy.deepcopy(t_review)
            if len(t_review) > 35 or len(t_review) < 2:
                continue
            if max([len(sentence.split()) for sentence in t_review]) > 50:
                continue
            if min([len(sentence.split()) for sentence in t_review]) < 5:
                continue
            review = ' '.join(t_review).split()

              # check whether aspect keywords in review
            class_list = []
            for aspect in aspects:
                keywords = lexicon[aspect]
                includes = int(any([keyword in review for keyword in keywords]))
                class_list.append(includes)
            # print(class_list)
            assert len(class_list) == 7

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
            for sub_j in j:
                reviews.append(split_sentence(sub_j['text']))


        for t_review in reviews :
            # print(t_review)
          # sanity check
            listfy_review = copy.deepcopy(t_review)
            if len(t_review) > 35 or len(t_review) < 2:
                continue
            if max([len(sentence.split()) for sentence in t_review]) > 50:
                continue
            if min([len(sentence.split()) for sentence in t_review]) < 2:
                continue
            review = ' '.join(t_review).split()

              # check whether aspect keywords in review
            class_list = []
            for aspect in aspects:
                keywords = lexicon[aspect]
                includes = int(any([keyword in review for keyword in keywords]))
                class_list.append(includes)
            # print(class_list)
            assert len(class_list) == 7

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
        # if key == (0, 0, 0, 0, 0, 0, 0):
        #     data += (instance_dict[key][:1000])
        # else:
        instances = instance_dict[key]
        random.shuffle(instances)
        # if len(instances) >= 10:
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
    print('Write done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_name', type = str)
    parser.add_argument('-result_data_name', type = str)
    parser.add_argument('-attribute_lexicon_name', default='/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', type = str)
    # parser.add_argument('-domain_name', default='finishline', type = str)

    args = parser.parse_args()
    if args.data_name == 'NB_man.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'NB_woman.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'man.json':
        create_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'woman.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'ascis_man.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)

    if args.data_name == 'ascis_woman.json':
        create_sythe_data('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)
