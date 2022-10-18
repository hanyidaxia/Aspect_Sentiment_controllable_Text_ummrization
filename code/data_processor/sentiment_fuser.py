import json
import argparse


def split_sentiment (file_dir, args):
    data1 = []
    with open (file_dir + args.data_name1, 'r') as f:
        for line in f:
            inst = json.loads(line)
            data1.append(inst)
    data_2 = []
    with open (file_dir + args.data_name2, 'r') as f:
        for line in f:
            inst = json.loads(line)
            data_2.append(inst)

    data1.extend(data_2)

    with open (file_dir + args.result_data_name, 'w') as f:
        for i in data1:
            f.write(json.dumps(i) + '\n')
        print('Write Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name1', default='S_7_5_man.json', type=str)
    parser.add_argument('-data_name2', default='S_7_5_woman.json', type=str)
    args = parser.parse_args()

    if args.data_name1 == 'S_7_5_man.json':
        parser.add_argument('-result_data_name', default='S_7_5_finishline.json', type=str)
        args = parser.parse_args()
        split_sentiment('/media/jade/yi_Data/Data/New_Data/Text_data/finishline/', args)
    if args.data_name1 == 'NB_S_man.json':
        parser.add_argument('-result_data_name', default='NB_S.json', type=str)
        # parser.add_argument('-data_name2', type=str)
        args = parser.parse_args()
        split_sentiment('/media/jade/yi_Data/Data/New_Data/Text_data/NB/', args)
