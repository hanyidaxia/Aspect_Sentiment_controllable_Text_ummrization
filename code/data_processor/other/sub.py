from textblob import TextBlob
import pandas as pd
import json
import os
# df = pd.read_csv('shoes_exterior.csv')
# df['polarity'] = df.apply(lambda x: TextBlob(x['Reviews']).sentiment.polarity, axis=1)
# ​df['subjectivity'] = df.apply(lambda x: TextBlob(x['Reviews']).sentiment.subjectivity, axis=1)
# print(df)
# df.to_csv('shoes_exterior_sentiment.csv')


def read_file(file_name, result_data_name):
    all_test = []
    if 'best_step' in file_name:
        with open (file_name) as f:
            all_test = f.readlines()
            clean_review = "".join(all_test)
            clean_review = clean_review.split('</s>')
    else:
        with open (file_name) as f:
            for line in f:
                inst = json.loads(line.strip())
                all_test.append(inst)
    df = pd.DataFrame(all_test)
    if 'best_data.jsonl' in file_name:
        df['review polarity'] = df.apply(lambda x: TextBlob(''.join(x['review'])).sentiment.polarity, axis=1)
        df['review subj'] = df.apply(lambda x: TextBlob(''.join(x['review'])).sentiment.subjectivity, axis=1)
    elif 'test_fin' in file_name:
        df['summary polarity'] = df.apply(lambda x: TextBlob(x['summary']).sentiment.polarity, axis=1)
        df['summary subj'] = df.apply(lambda x: TextBlob(''.join(x['summary'])).sentiment.subjectivity, axis=1)
    elif 'best_step' in file_name:
        True
    df.to_csv(result_data_name)
    print(df)


if __name__ == '__main__':
    read_file(
    '/home/jade/untextsum/data/all/Sentiment_filtered10_BCE_Sum_weightstep.10000 average_loss.613.18 sentence_f1.79.83 document_f1.81.44_best_dataBCE_Sum_weight16000/test_fin.jsonl',
    'test_data_summary_sentiment.csv')
    read_file(
    '/home/jade/untextsum/data/all/best_data.jsonl', 'review_sentiment.csv'
    )
    read_file(
    '/home/jade/untextsum/output/all/Rouge.best_step.19400_rouge.0.147.out.aspect_2.jsonl', 'final_sentiment.csv'
    )
