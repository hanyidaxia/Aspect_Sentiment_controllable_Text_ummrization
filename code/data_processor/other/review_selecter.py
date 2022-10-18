""""""
import spacy
import os
nlp = spacy.load('en_core_web_sm')
import json
from tqdm import tqdm
import random
import pandas as pd
import csv
import sys
from statistics import mean


def information_count(file_dir):
    brands  = os.listdir(file_dir)
    print(brands)
    all_file = []
    all_all_file = []
    for brand in brands:
        files = os.listdir(file_dir + brand + '/')
        # print(files)
        for file in files:
            if file == 'man_attribute.json' or file == 'woman_attribute.json' or \
            file == 'NB_man_attribute.json' or file == 'NB_woman_attribute.json':
                count = 1
                brand_count = 1
                long_count = 1
                short_count = 1
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for j in tqdm(dic.values()):
                        brand_count += 1
                        for i in j:
                            count += 1
                            tmp_sen_len = len(i["text"].split('. '))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 60\
                            and tmp_sen_len > 1 and tmp_sen_len <= 10:
                                all_file.append(i)
                                all_all_file.append(i)
                            elif len(i["text"].split()) > 60:
                                long_count += 1
                            elif len(i["text"].split()) < 10:
                                short_count += 1
                print('current there are long %d reviews' % long_count)
                print('current there are %d reviews' % count)
                print('current there are brand %d reviews' % brand_count)
                print('current there are short %d reviews' % short_count)
            elif file == 'ascis_man_attribute.json' or file == 'ascis_woman_attribute.json':
                count = 1
                brand_count = 1
                long_count = 1
                short_count = 1
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for j in tqdm(dic.values()):
                        brand_count += 1
                        for i in j:
                            count += 1
                            tmp_sen_len = len(i["text"].split('. '))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 60\
                            and tmp_sen_len > 1 and tmp_sen_len <= 10:
                                all_all_file.append(i)
                            elif len(i["text"].split()) > 60:
                                long_count += 1
                            elif len(i["text"].split()) < 10:
                                short_count += 1
                print('current there are long %d reviews' % long_count)
                print('current there are %d reviews' % count)
                print('current there are brand %d reviews' % brand_count)
                print('current there are short %d reviews' % short_count)

    ball_file = []
    star5 = []
    star4 = []
    star3 = []
    star2 = []
    star1 = []
    buffer = len(all_file)
    print(buffer, len(all_file))
    for i in range (0, len(all_file)):
        if all_file[i]['rate'] == "5":
            star5.append(all_file[i])
        elif all_file[i]['rate'] == "4":
            star4.append(all_file[i])
        elif all_file[i]['rate'] == "3":
            star3.append(all_file[i])
        elif all_file[i]['rate'] == "2":
            star2.append(all_file[i])
        elif all_file[i]['rate'] == "1":
            star1.append(all_file[i])
        else:
            continue
    shortest = min([len(star1), len(star3), len(star4),len(star5)])
    print([len(star1), len(star2), len(star3), len(star4), len(star5)])



def random_selection(file_dir):
    brands  = os.listdir(file_dir)
    print(brands)
    all_file = []
    all_all_file = []
    for brand in brands:
        files = os.listdir(file_dir + brand + '/')
        # print(files)
        for file in files:
            if file == 'man_attribute.json' or file == 'woman_attribute.json' or file == 'NB_man_attribute.json' or file == 'NB_woman_attribute.json':
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for j in tqdm(dic.values()):
                        for i in j:
                            tmp_sen_len = len(list(nlp(i["text"]).sents))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 80 and len(i["attribute"]) != 0 \
                            and tmp_sen_len > 1 and tmp_sen_len <= 7:
                                all_file.append(i)
                                all_all_file.append(i)
            elif file == 'ascis_man_attribute.json' or file == 'ascis_woman_attribute.json':
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for j in tqdm(dic.values()):
                        for i in j:
                            tmp_sen_len = len(list(nlp(i["text"]).sents))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 80 and len(i["attribute"]) != 0 \
                            and tmp_sen_len > 1 and tmp_sen_len <= 7:
                                all_all_file.append(i)

    ball_file = []
    star5 = []
    star4 = []
    star3 = []
    star2 = []
    star1 = []
    buffer = len(all_file)
    print(buffer, len(all_file))
    for i in range (0, len(all_file)):
        if all_file[i]['rate'] == "5":
            star5.append(all_file[i])
        elif all_file[i]['rate'] == "4":
            star4.append(all_file[i])
        elif all_file[i]['rate'] == "3":
            star3.append(all_file[i])
        elif all_file[i]['rate'] == "2":
            star2.append(all_file[i])
        elif all_file[i]['rate'] == "1":
            star1.append(all_file[i])
        else:
            continue
    shortest = min([len(star1), len(star3), len(star4),len(star5)])
    print([len(star1), len(star2), len(star3), len(star4), len(star5)])

    while len(star5) > shortest + len(star2):
        star5.pop(random.randint(0, len(star5) - 1))
    while len(star4) > shortest - (shortest - len(star2)):
        star4.pop(random.randint(0, len(star4) - 1))
    while len(star3) > shortest:
        star3.pop(random.randint(0, len(star3) - 1))
    # while len(star2) > :
    #     star2.pop(random.randint(0, len(star2) - 1))
    while len(star1) > shortest:
        star1.pop(random.randint(0, len(star1) - 1))

    # while buffer >= len(all_file) - 2000:
    #     evict_idx = random.randint(0, buffer-1)
    #     if all_file[evict_idx]['rate'] == "5" and len(star5) < 400:
    #         star5.append(all_file[evict_idx])
    #         buffer -= 1
    #         all_file.pop(evict_idx)
    #     elif all_file[evict_idx]['rate'] == "4" and len(star4) < 400:
    #         star5.append(all_file[evict_idx])
    #         buffer -= 1
    #         all_file.pop(evict_idx)
    #     elif all_file[evict_idx]['rate'] == "3" and len(star3) < 400:
    #         star5.append(all_file[evict_idx])
    #         buffer -= 1
    #         all_file.pop(evict_idx)
    #     elif all_file[evict_idx]['rate'] == "2" and len(star2) < 400:
    #         star5.append(all_file[evict_idx])
    #         buffer -= 1
    #         all_file.pop(evict_idx)
    #     elif all_file[evict_idx]['rate'] == "1" and len(star1) < 400:
    #         star5.append(all_file[evict_idx])
    #         buffer -= 1
    #         all_file.pop(evict_idx)
    #     else:
    #         print("else condition")
    #         break
    #         #continue

    ball_file = star5+star4+star3+star2+star1
    high_quality_review = [x for x in all_all_file if x not in ball_file]
    # print(len(ball_file))
    # buffer_size = len(ball_file)
    # shufbuf = []
    # results = []
    # for inst in all_file:
    #     try:
    #         for i in range(buffer_size):
    #             shufbuf.append(inst)
    #     except:
    #         buffer_size = len(shufbuf)
    #     try:
    #         while len(results) < 2000:
    #             evict_idx = random.randint(0, buffer_size-1)
    #             results = shufbuf[evict_idx]

    with open ('/home/jade/untextsum/data/ran_balan.jsonl', 'w') as f:
        # assert len(ball_file) == 2000
        for i in tqdm(ball_file):
            f.write(json.dumps(i) + '\n')
    print('Write Done')

    with open ('/home/jade/untextsum/data/good_rest.jsonl', 'w') as f:
        # assert len(ball_file) == 2000
        for i in tqdm(high_quality_review):
            f.write(json.dumps(i) + '\n')
    print('Write Done')

def random_selection_name(file_dir):
    brands  = os.listdir(file_dir)
    print(brands)
    all_file = []
    all_all_file = []
    for brand in brands:
        files = os.listdir(file_dir + brand + '/')
        # print(files)
        for file in files:
            if file == 'man_attribute.json' or file == 'woman_attribute.json' or file == 'NB_man_attribute.json' or file == 'NB_woman_attribute.json':
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for k, j in tqdm(dic.items()):
                        for i in j:
                            tmp_sen_len = len(list(nlp(i["text"]).sents))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 80 \
                            and tmp_sen_len > 1 and tmp_sen_len <= 7:
                                i['name'] = k
                                all_file.append(i)
                                all_all_file.append(i)
            elif file == 'ascis_man_attribute.json' or file == 'ascis_woman_attribute.json':
                with open (file_dir + brand + '/' + file, 'r') as f:
                    print(file_dir + brand + '/' + file)
                    dic = json.load(f)
                    for k, j in tqdm(dic.items()):
                        for i in j:
                            tmp_sen_len = len(list(nlp(i["text"]).sents))
                            if len(i["text"].split()) >= 10 and len(i["text"].split()) <= 80 and \
                            tmp_sen_len > 1 and tmp_sen_len <= 7:
                                i['name'] = k
                                all_all_file.append(i)

    ball_file = []
    star5 = []
    star4 = []
    star3 = []
    star2 = []
    star1 = []
    buffer = len(all_file)
    print(buffer, len(all_file))
    for i in range(0, len(all_file)):
        if all_file[i]['rate'] == "5":
            star5.append(all_file[i])
        elif all_file[i]['rate'] == "4":
            star4.append(all_file[i])
        elif all_file[i]['rate'] == "3":
            star3.append(all_file[i])
        elif all_file[i]['rate'] == "2":
            star2.append(all_file[i])
        elif all_file[i]['rate'] == "1":
            star1.append(all_file[i])
        else:
            continue
    # shortest = min([len(star1), len(star2) len(star3), len(star4),len(star5)])
    shortest = 400
    print([len(star1), len(star2), len(star3), len(star4), len(star5)])

    while len(star5) > shortest:
        star5.pop(random.randint(0, len(star5) - 1))
    while len(star4) > shortest:
        star4.pop(random.randint(0, len(star4) - 1))
    while len(star3) > shortest:
        star3.pop(random.randint(0, len(star3) - 1))
    while len(star2) > shortest:
        star2.pop(random.randint(0, len(star2) - 1))
    while len(star1) > shortest:
        star1.pop(random.randint(0, len(star1) - 1))

    ball_file = star5+star4+star3+star2+star1
    high_quality_review = [x for x in all_all_file if x not in ball_file]

    with open ('/home/jade/untextsum/data/ran_balan_2000.jsonl', 'w') as f:
        assert len(ball_file) == 2000
        for i in tqdm(ball_file):
            f.write(json.dumps(i) + '\n')
    print('Write Done')

    with open ('/home/jade/untextsum/data/good_rest_after2000.jsonl', 'w') as f:
        # assert len(ball_file) == 2000
        for i in tqdm(high_quality_review):
            f.write(json.dumps(i) + '\n')
    print('Write Done')


def for_ada(file, re_file_name, file_dir):
    files  = os.listdir(file_dir)
    dic = {}
    for t_file in files:
        with open (file_dir + t_file, 'r') as f:
            tmp_dic = json.load(f)
            dic.update(tmp_dic)
    if 'summary' not in file:
        df =  pd.read_csv(file)
        result_list = []
        for index, row in tqdm(df.iterrows()):
            record = {}
            for k, v in dic.items():
                for j in v:
                    if eval(row['review'])[0].lower() in j['text'].lower().strip():
                        if eval(row['review'])[1]:
                            if eval(row['review'])[1].lower() in j['text'].lower().strip():
                                record['product_name'] = k
                        else:
                            record['product_name'] = k
                            record['gender'] = 'man' if 'man' in file else 'woman'
            record['review'] = row['review']
            for ap in row['aspects'].strip('{}').split(', ')[:-3]:
                if 'yes' in ap:
                    record[ap.split(':')[0]] = random.uniform(-1, 1)
                if 'no' in ap:
                    record[ap.split(':')[0]] = 'Not found in review'
            if row['review polarity'] > 0.1:
                record['Stars'] = random.randint(4,5)
            elif -0.2 < row['review polarity'] < 0.1:
                record['Stars'] = 3
            else:
                record['Stars'] = random.randint(1,2)
            record['Date'] = 'Currently unavailable'
            record['Gender'] = ['M', 'F'][random.randint(0,1)]
            record['Shoe Model'] = 'Currently unavailable'
            result_list.append(record)
        with open(re_file_name, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, result_list[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(result_list)
        print('modification done')
    else:
        data = []
        with open (file,  'r', encoding="UTF-8") as f:
            for line in f:
                inst = json.loads(line)
                data.append(inst)
        result_list = []
        for index, review in enumerate(data):
            record = {}
            record['Summary ID'] = 'ID' + str(index)
            record['Summary'] = review['summary']
            for ap in ['aspects'].strip('{}').split(', ')[:-3]:
                if 'yes' in ap:
                    record[ap.split(':')[0]] = random.uniform(-1, 1)
                if 'no' in ap:
                    record[ap.split(':')[0]] = 'Not found in review'
            if row['review polarity'] > 0.1:
                record['Stars'] = random.randint(4,5)
            elif -0.2 < row['review polarity'] < 0.1:
                record['Stars'] = 3
            else:
                record['Stars'] = random.randint(1,2)
            record['Date'] = 'Currently unavailable'
            record['Gender'] = ['M', 'F'][random.randint(0,1)]
            record['Shoe Model'] = 'Currently unavailable'
            result_list.append(record)
        with open(re_file_name, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, result_list[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(result_list)
        print('modification done')

def ada_sum_table(file, re_file_name):
    data = []
    with open (file, 'r', encoding='UTF-8') as f:
        for line in f:
             data.append(json.loads(line.strip()))
    result_list = []
    for inst in data:
        tmp = {}
        tmp['aspect summary'] = inst['summary']
        tmp['general summary'] = ''
        tmp['Permeability'] = 1 if inst['switch'][0] == 1 else 0
        tmp['Impact absorption'] = 1 if inst['switch'][1] == 1 else 0
        tmp['Stability'] = 1 if inst['switch'][2] == 1 else 0
        tmp['Durability'] = 1 if inst['switch'][3] == 1 else 0
        tmp['Shoe parts'] = 1 if inst['switch'][4] == 1 else 0
        tmp['Exterior'] = 1 if inst['switch'][5] == 1 else 0
        tmp['Fit'] = 1 if inst['switch'][6] == 1 else 0
        rdice = random.randint(0,1)
        if inst['switch'][-3] == 1:
            tmp['sentiment'] = 'POS'
        elif inst['switch'][-2] == 1:
            tmp['sentiment'] = 'NEU'
        elif inst['switch'][-1] == 1:
            if rdice == 0:
                tmp['sentiment'] = 'NEG'
            else:
                tmp['sentiment'] = 'NEU'
        result_list.append(tmp)
    with open(re_file_name, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, result_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(result_list)
    print('modification done')

def ada_casual(file_path, result_name):
    all_data = []
    files  = os.listdir(file_path)
    for file in files:
        with open (file_path + file, 'r') as f:
            dic = json.load(f)
            for k, v in dic.items():
                tmp_dic = {}
                tmp_dic['shoe_name'] = k
                if 'ascis' in file or 'NB' in file:
                    tmp_dic['brand'] = file.split('_')[0]
                    if 'NB' not in file:
                        tmp_dic['overall_rating'] = ''
                    else:
                        tmp_dic['overall_rating'] = mean([int(i['rate']) for i in v])
                else:
                    tmp_dic['brand'] = ''
                    # try:
                    tmp_dic['overall_rating'] = mean([int(i['rate']) for i in v if i['rate'] != "" and
                    i['rate'] != "None"])
                    # except:
                    #     print(v)
                tmp_dic['number_reviews'] = len(v)
                tmp_dic['M/F'] = 'M' if 'man' in file else 'F'
                all_data.append(tmp_dic)
    with open(result_name, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, all_data[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(all_data)
    print('write done')















if __name__ == '__main__':
    ada_casual('/home/jade/Downloads/Text_data729/Text_data/ada/', '/home/jade/Downloads/Product_table.csv')
    # random_selection_name('/media/jade/yi_Data/Data/New_Data/Text_data/')
    # for_ada('/home/jade/Downloads/review_sentiment.csv','/home/jade/Downloads/sentiment_table.csv',
    #         '/home/jade/Downloads/Text_data729/Text_data/ada/')
    # information_count('/home/jade/Downloads/Text_data729/Text_data/')
#     ada_sum_table('/home/jade/untextsum/data/\
# allCampusmin60max120summary_source_num5review_source_num60keywords5/\
# Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 \
# sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000/retest_fin.jsonl',
# '/home/jade/untextsum/data/ada_sum.csv')
