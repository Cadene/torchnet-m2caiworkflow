# coding: utf-8

import numpy as np
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from itertools import chain
import os
np.random.seed(42)


RUN_IDENTIFIER = "test_fulltrain_single"
CNN_PREDICTIONS_TRAIN_PATH = "/local/robert/m2cai/workflow/extract/finetuning_train/16_09_18_02:19:59_epoch,10/trainset.csv" # each line: "imgpath;stepInt;stepStr;probasSemicolonSeparated"
CNN_PREDICTIONS_TEST_PATH = "/local/robert/m2cai/workflow/extract/finetuning_train/16_09_18_02:19:59_epoch,10/testset.csv" # each line: "imgpath;probasSemicolonSeparated"
SAVE = True # Do we actually save stuff (use false to test the script)
VAL = False # Are the test data comming from a valset

#########################
# Data loading
#########################

print "Loading data"

# Load video annotations for HMM modelling
videos_true_length = []
videos = []
for i in range(1,28):
    steps = map(lambda x: x.strip().split("\t")[1], open("../annotations/workflow_video_%02d.txt" % i, "r").readlines()[1:])
    videos.append(steps[24::25]) # keep 1 frame out of 25
    videos_true_length.append(len(steps)) # keep 1 frame out of 25

ordered_unique_steps = ["TrocarPlacement", "Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
     "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
p = len(ordered_unique_steps)

step_index = {v: k for k, v in zip(range(p), ordered_unique_steps)}

# Load probabilities of step for train / test sequences
def processEmissionTrain(l):
    data = l.strip().split(";")
    vidInfo = re.search(r"_([0-9]+)-([0-9]+)", data[0])
    return (int(vidInfo.group(1)) - 1, int(vidInfo.group(2)) -1, int(data[1]) - 1, map(float, data[3:]))

def processEmissionTest(l):
    data = l.strip().split(";")
    vidInfo = re.search(r"_([0-9]+)-([0-9]+)", data[0])
    return (int(vidInfo.group(1)) - 1, int(vidInfo.group(2)) -1, map(float, data[1:]))

train_pred_seq = map(processEmissionTrain, open(CNN_PREDICTIONS_TRAIN_PATH, "r").readlines()[1:])
test_pred_seq = map(processEmissionTest, open(CNN_PREDICTIONS_TEST_PATH, "r").readlines()[1:])

# Order test data for sequential prediction
test_pred_seq_df = pd.DataFrame(test_pred_seq, columns=["vid","frame", "probas"])
train_pred_seq_df = pd.DataFrame(train_pred_seq, columns=["vid","frame", "class", "probas"])

test_pred_probas_ids = []
test_pred_probas = []
train_pred_probas = []

for vidId, group in test_pred_seq_df.groupby("vid"):
    group = group.sort("frame")

    test_pred_probas_ids.append(vidId)
    test_pred_probas.append(group["probas"].tolist())

# Nb of frames in test videos
videos_true_length_test = dict(
    map(lambda x: (int(x[0][-6:-4]) -1 , int(x[1])),
        map(lambda x: x.split("\t"), open("../dataset2/nb-frames.txt", "r").read().strip().split("\t\r\n"))
        )
    )

#########################
# HMM modelling
#########################

print "Modelling with HMM"

# Computing transition / start statistics

pi = np.zeros(p)

for steps in videos:
    pi[step_index[steps[0]]] += 1
pi /= len(videos)

A = np.zeros((p,p))

for steps in videos:
    for (s1, s2) in zip(steps[0:-1], steps[1:]):
        A[step_index[s1], step_index[s2]] += 1

A /= np.sum(A, axis=1, keepdims=True)

# Computing emission statistics

probas_by_class = [[], [], [], [], [], [], [], []]

for vidId, group in train_pred_seq_df.groupby("vid"):
    group = group.sort("frame")

    probas = np.array(group["probas"].tolist())
    classes = group["class"].tolist()

    for i in range(1,15):
        probas[i:] = (i * probas[i:] + probas[:-i]) / (i + 1)

    for i in range(len(probas)):
        probas_by_class[classes[i]].append(probas[i])

# Compute sigmas & mus

mus = np.zeros((p,p))
sigmas = np.zeros((p,p,p))

for i in range(p):
    X = np.array(probas_by_class[i])
    mus[i, :] = np.mean(X, axis=0)
    sigmas[i, :, :] = np.cov(X+np.random.randn(X.shape[0], X.shape[1])/100000, rowvar=False)

## Init HMM

model = hmm.GaussianHMM(n_components=p, covariance_type="full")
model.startprob_ = pi
model.transmat_ = A
model.means_ = mus
model.covars_ = sigmas

print "pi:"
print pi
print "A:"
print A
print "mus:"
print mus

#########################
# Decoding test sequences
#########################

print "Start decoding..."

for testInd in range(len(test_pred_probas_ids)):

    print "> Decoding video %d" % testInd

    vidInd = test_pred_probas_ids[testInd]
    probas = np.array(test_pred_probas[testInd])
    probas_cnn = np.array(probas)

    # lissage
    for i in range(1,15):
        probas[i:] = (i * probas[i:] + probas[:-i]) / (i + 1)

    # pred
    steps_hat_cnn = np.argmax(np.array(probas_cnn), axis=1)
    steps_hat_cnn_smooth = np.argmax(np.array(probas), axis=1)
    (_, steps_hat_hmm_off) = model.decode(probas)
    steps_hat_hmm = np.array(steps_hat_cnn) * 0
    for i in range(len(probas)):
        if i % 100 == 0:
            print ">> Step %d" % i
        (_, steps_hat_hmm_tmp) = model.decode(probas[:i+1])
        steps_hat_hmm[i] = steps_hat_hmm_tmp[-1]

    print steps_hat_hmm

    # adjust length and save
    if VAL:
        targetLen = videos_true_length[vidInd]
    else:
        targetLen = videos_true_length_test[vidInd]

    steps_hat_cnn = list(chain(*zip(*[steps_hat_cnn for _ in range(25)])))
    steps_hat_cnn_smooth = list(chain(*zip(*[steps_hat_cnn_smooth for _ in range(25)])))
    steps_hat_hmm = list(chain(*zip(*[steps_hat_hmm for _ in range(25)])))
    steps_hat_hmm_off = list(chain(*zip(*[steps_hat_hmm_off for _ in range(25)])))

    steps_hat_cnn = steps_hat_cnn[0:targetLen] # crop if too long
    steps_hat_cnn += [steps_hat_cnn[-1]] * (targetLen - len(steps_hat_cnn)) # pad if too short

    steps_hat_cnn_smooth = steps_hat_cnn_smooth[0:targetLen] # crop if too long
    steps_hat_cnn_smooth += [steps_hat_cnn_smooth[-1]] * (targetLen - len(steps_hat_cnn_smooth)) # pad if too short

    steps_hat_hmm = steps_hat_hmm[0:targetLen] # crop if too long
    steps_hat_hmm += [steps_hat_hmm[-1]] * (targetLen - len(steps_hat_hmm)) # pad if too short

    steps_hat_hmm_off = steps_hat_hmm_off[0:targetLen] # crop if too long
    steps_hat_hmm_off += [steps_hat_hmm_off[-1]] * (targetLen - len(steps_hat_hmm_off)) # pad if too short

    if SAVE:
        os.system("mkdir -p ../predictions/"+RUN_IDENTIFIER+"_cnn")
        os.system("mkdir -p ../predictions/"+RUN_IDENTIFIER+"_cnn_smooth")
        os.system("mkdir -p ../predictions/"+RUN_IDENTIFIER+"_hmm")
        os.system("mkdir -p ../predictions/"+RUN_IDENTIFIER+"_hmm_off")

        f = open("../predictions/"+RUN_IDENTIFIER+"_cnn/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_cnn):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open("../predictions/"+RUN_IDENTIFIER+"_cnn_smooth/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_cnn_smooth):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open("../predictions/"+RUN_IDENTIFIER+"_hmm/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_hmm):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open("../predictions/"+RUN_IDENTIFIER+"_hmm_off/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_hmm_off):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()
