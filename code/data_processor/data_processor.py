import argparse
import json
import numpy as np
import tqdm
import random
import sys
import os
print (os.getcwd())

def write_json(file_name, dic):
        with open(file_name, "w") as f:
            json.dump(dic, f)
            print("Done")


def clean_data(file_dir, args):
    with open(file_dir + args.data_name, 'r', encoding = "UTF-8") as f:
        clean_review = f.readlines()
        clean_review = "".join(clean_review)
        clean_review = clean_review.split('*****************************************************************')
    review_dic = {}


    for product_review in clean_review:
        reviews_temp = product_review.split('\n')
        reviews = []
        [reviews.append(x) for x in reviews_temp if x not in reviews]
        for review in reviews:
            if "==='" in review:
                p_name = review.split("==='")[-1].strip()
            if 'rate' in review and 'recommend' in review and 'text' in review:
                mini_dic = {}
                rate = review.split('  recommend')[0].split('rate:')[-1]
                recommend  = review.split('  text')[0].split('recommend:')[-1]
#                 if recommend == 'Yes':
#                     recommend = True
#                 else:
#                     recommend = False
                text = review.split('text:')[-1]
                mini_dic['rate'] = rate
                mini_dic['recommend'] = recommend
                mini_dic['text'] = text

                if p_name not in review_dic:
                    review_dic[p_name] = [mini_dic]
                else:
                    review_dic[p_name].append(mini_dic)


    print(type(review_dic))

    write_json(file_dir + args.result_data_name, review_dic)



def clean_NB(file_dir, args):
    with open(file_dir + args.data_name, 'r', encoding = "UTF-8") as f:
        clean_review = f.readlines()
        clean_review = "".join(clean_review)
        clean_review = clean_review.split('*****************************************************************')
    review_dic = {}


    for product_review in clean_review:
        reviews_temp = product_review.split('\n')
        reviews = []
        [reviews.append(x) for x in reviews_temp if x not in reviews]
        for review in reviews:
            if "==='" in review:
                p_name = review.split("==='")[-1].strip()
            if 'rate' in review and 'head' in review:
                mini_dic = {}
                rate = review.split('out of 5 stars  ')[0].split('rate:Rated ')[-1].strip()
                head = review.split('  text')[0].split('head:')[-1]
#                 if recommend == 'Yes':
#                     recommend = True
#                 else:
#                     recommend = False
                text = review.split('text:')[-1]
                mini_dic['rate'] = rate
                mini_dic['review_title'] = head
                mini_dic['text'] = text

                if p_name not in review_dic:
                    review_dic[p_name] = [mini_dic]
                else:
                    review_dic[p_name].append(mini_dic)


    print(type(review_dic))

    write_json(file_dir + args.result_data_name, review_dic)




def clean_ascis(file_dir, args):
    with open(file_dir + args.data_name, 'r', encoding = "UTF-8") as f:
        clean_review = f.readlines()
        clean_review = "".join(clean_review)
        clean_review = clean_review.split('*****************************************************************')
    review_dic = {}


    for product_review in clean_review:
        reviews_temp = product_review.split('\n')
        reviews = []
        [reviews.append(x) for x in reviews_temp if x not in reviews]
        for review in reviews:
            if "==='" in review:
                p_name = review.split("==='")[-1].strip()
            if 'rate' in review and 'stars' in review:
                mini_dic = {}
                rate = review.split('out of 5 stars.  ')[0].split('rate:')[-1].strip()
                # head = review.split('  text')[0].split('head:')[-1]
#                 if recommend == 'Yes':
#                     recommend = True
#                 else:
#                     recommend = False
                text = review.split('text:     ')[-1].strip()
                mini_dic['rate'] = rate
                # mini_dic['review_title'] = head
                mini_dic['text'] = text

                if p_name not in review_dic:
                    review_dic[p_name] = [mini_dic]
                else:
                    review_dic[p_name].append(mini_dic)


    print(type(review_dic))

    write_json(file_dir + args.result_data_name, review_dic)





def get_key(dic, word):
    for i, j in dic.items():
        if word in j:
            return i
        else:
            continue





def attribute_annotation (file_dir, args):
    with open ('/media/jade/yi_Data/Data/20220111_attribute_lexicon.json', 'r') as f:
        lexicon = json.load(f)
    with open(file_dir + args.data_name, 'r') as f:
        data = json.load(f)


#         print(type(w))
    for i, j in data.items():
        for sub_j in j:
            sub_j['attribute'] = []
            for w in sum(lexicon.values(), []):
                if w in sub_j['text'].lower():
                    sub_j['attribute'].append(get_key(lexicon, w))

    write_json(file_dir + args.result_data_name, data)













if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_name', type = str)
    parser.add_argument('-result_data_name', type = str)
    parser.add_argument('-attribute_lexicon_name', type = str)


    args = parser.parse_args()
    print(args)

    if args.data_name == 'product_names_with_reviews_man_v1_results.txt':
        clean_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'product_names_with_reviews_woman_v1_results.txt':
        clean_data('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'NB_product_with_reviews_man_v1_results.txt':
        clean_NB('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'NB_product_with_reviews_woman_v1_results.txt':
        clean_NB('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'ASICS_shoes_names_with_reviews_man_results1.txt':
        clean_ascis('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)

    if args.data_name == 'ASICS_shoes_names_with_reviews_woman_results1.txt':
        clean_ascis('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)

    if args.data_name == 'NB_man.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'NB_woman.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)

    if args.data_name == 'man.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'woman.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)

    if args.data_name == 'ascis_man.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)

    if args.data_name == 'ascis_woman.json':
        attribute_annotation('/media/jade/yi_Data/Data/New_Data/Text_data/ascis/', args)
