import pandas as pd
import numpy as np

def do_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def vote(experiments, mode, aggregation='sum', softmax=True):
    dataframes = []
    predcols = ['pred1','pred2','pred3','pred4','pred5','pred6','pred7','pred8']
    exportcols = ['path']
    if mode == 'train':
        exportcols += ['gttarget','gtclass']
    for i, exp in enumerate(experiments):
        if mode == 'trainval':
            dftrain = pd.read_csv('/local/robert/m2cai/workflow/extract/'+exp+'/trainset.csv', sep=';')
            dfval = pd.read_csv('/local/robert/m2cai/workflow/extract/'+exp+'/valset.csv', sep=';')
            df = pd.concat([dftrain,dfval])
        else:
            df = pd.read_csv('/local/robert/m2cai/workflow/extract/'+exp+'/'+mode+'set.csv', sep=';')
        print(len(df), exp)
        #df = df.set_index('Id')
        if softmax:
            df[predcols] = df[predcols].apply(do_softmax)
        if aggregation == 'vote':
            df['predtarget'] = df[predcols].idxmax(axis=1)
        dataframes.append(df)
    zero = np.zeros(len(dataframes[0].index))
    dfvote = dataframes[0][exportcols]
    dfvote.loc[:,'pred1'] = zero.copy()
    dfvote.loc[:,'pred2'] = zero.copy()
    dfvote.loc[:,'pred3'] = zero.copy()
    dfvote.loc[:,'pred4'] = zero.copy()
    dfvote.loc[:,'pred5'] = zero.copy()
    dfvote.loc[:,'pred6'] = zero.copy()
    dfvote.loc[:,'pred7'] = zero.copy()
    dfvote.loc[:,'pred8'] = zero.copy()
    for i, df in enumerate(dataframes):
        if aggregation == 'vote':
            for j in range(1,9):
                dfvote['pred'+str(j)] += (df['predtarget'] == 'pred'+str(j))
        if aggregation == 'sum':
            for col in predcols:
                dfvote[col] += df[col]
    dfvote[exportcols+predcols].to_csv(
        '/local/robert/m2cai/workflow/concat/'+mode+'concat_ensemble_vote.csv',
        index=False, sep=";"
    )
    return dfvote

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
print(len(experiments))
dfvote = vote(experiments,mode='trainval',aggregation='vote',softmax=False)
dfvote = vote(experiments,mode='test',aggregation='vote',softmax=False)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# print(df_vote.loc[df_vote['1'] <= 55].loc[df_vote['1'] >= 30])