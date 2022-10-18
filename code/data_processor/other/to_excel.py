import json
import csv
import pandas as pd


print('excuted')
# Opening JSON file and loading the data
# into the variable data
dic = []
with open('/home/jade/untextsum/data/ran_balan_2000.jsonl' , 'r') as f:
    data = f.readlines()
    for line in data:
        inst = json.loads(line)
        dic.append(inst)

# now we will open a file for writing
# with open('/home/jade/untextsum/output/finishline/data_file.csv', 'w') as f:

# create the csv writer object
# df = pd.DataFrame(data)
    # writer = csv.writer(f)
    # # df.to_csv(writer,index=False)
    # # writer.save()
    # for i in data:
    #     writer.writerow(i)
df = pd.DataFrame(dic)
# writer = pd.ExcelWriter(filedir + file_name + '.xlsx', engine='xlsxwriter')
df.to_csv('/home/jade/untextsum/data/ran_balan_2000.csv',index=False)
# writer.save()
