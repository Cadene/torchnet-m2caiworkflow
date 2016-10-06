import pandas as pd
import numpy as np
import re

experiments = [
    'inceptionv3_1/16_08_05_05:04:07',
    'resnet_1/16_09_06_11:31:46',
    'resnet_1/16_09_06_11:32:33',
    'inceptionv3_2/16_09_13_08:35:55',
    'inceptionv3_2/16_09_13_08:36:10',
    'inceptionv3_3/16_09_13_18:36:37',
    'inceptionv3_3/16_09_13_18:36:39',
    'inceptionv3_4/16_09_14_01:45:23',
    'inceptionv3_4/16_09_14_01:44:57',
    'inceptionv3_5/16_09_14_11:57:25',
    'inceptionv3_fold_0/16_09_15_09:53:05',
    'inceptionv3_fold_0/16_09_15_01:03:38',
    'inceptionv3_fold_1/16_09_15_01:04:10',
    'inceptionv3_fold_2/16_09_15_14:59:24',
    'inceptionv3_fold_2/16_09_15_14:58:53',
    'inceptionv3_fold_4/16_09_16_10:37:56',
    'inceptionv3_fold_4/16_09_16_10:38:19',
    'inceptionv3_fold_5/16_09_16_19:42:36',
    'inceptionv3_6b/16_09_14_16:21:56',
    'inceptionv3_6b/16_09_14_16:21:08',
    'inceptionv3_fold_6/16_09_16_19:34:42',
    'inceptionv3_fold_6/16_09_16_19:35:01'
]

def processEmissionTrain(l):
    data = l.strip().split(";")
    return (data[0], data[1], map(float, data[3:]))

data = []

for exp in experiments:
    f = open('/local/robert/m2cai/workflow/extract/'+exp+'/trainset.csv', "r")
    data += map(processEmissionTrain, f.readlines()[1:])

print data[0]
print data[10000]
print len(data)

#print(len(experiments))
#dfvote = vote(experiments,mode='trainval',aggregation='vote',softmax=False)
#dfvote = vote(experiments,mode='test',aggregation='vote',softmax=False)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# print(df_vote.loc[df_vote['1'] <= 55].loc[df_vote['1'] >= 30])
