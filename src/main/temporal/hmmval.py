# coding: utf-8
import time
import numpy as np
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from itertools import chain
import os
import seaborn
from optparse import OptionParser
np.random.seed(42)

parser = OptionParser()
parser.add_option("--dirannot", default="/local/robert/m2cai/workflow/annotations")
parser.add_option("--pathtraincsv", default="/local/robert/m2cai/workflow/cnnPreds/trainextract.csv")
parser.add_option("--pathvalcsv", default="/local/robert/m2cai/workflow/cnnPreds/valextract.csv")
parser.add_option("--save", default=True)
parser.add_option("--dirsave", default="experiments/hmmval/"+time.strftime("%y_%m_%d_%H:%M:%S"))
(options, args) = parser.parse_args()

# format traincsv : each line: "imgpath;stepInt;stepStr;probasSemicolonSeparated"
# format valcsv : each line: "imgpath;probasSemicolonSeparated"

#########################
# Data loading
#########################

print "Loading data"

# Load video annotations for HMM modelling
videos = []
videos_true_length = []
for i in range(1,28):
    steps = map(lambda x: x.strip().split("\t")[1], open(options.dirannot+"/workflow_video_%02d.txt" % i, "r").readlines()[1:])
    videos.append(steps[24::25]) # keep 1 frame out of 25
    videos_true_length.append(len(steps)) # keep 1 frame out of 25

ordered_unique_steps = ["TrocarPlacement", "Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
     "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
p = len(ordered_unique_steps)

step_index = {v: k for k, v in zip(range(p), ordered_unique_steps)}

# Load probabilities of step for train / val sequences
def processEmissionTrain(l):
    data = l.strip().split(";")
    vidInfo = re.search(r"_([0-9]+)-([0-9]+)", data[0])
    return (int(vidInfo.group(1)) - 1, int(vidInfo.group(2)) -1, int(data[1]) - 1, map(float, data[3:]))

train_pred_seq = map(processEmissionTrain, open(options.pathtraincsv, "r").readlines()[1:])
val_pred_seq   = map(processEmissionTrain, open(options.pathvalcsv, "r").readlines()[1:])

# Order val data for sequential prediction
train_pred_seq_df = pd.DataFrame(train_pred_seq, columns=["vid","frame", "class", "probas"])
val_pred_seq_df   = pd.DataFrame(val_pred_seq,   columns=["vid","frame", "class", "probas"])

# Extract probas and classes for each videos in frame order
val_pred_probas = []
val_pred_probas_gt = []
val_pred_probas_ids = []
train_pred_probas = []
for vidId, group in val_pred_seq_df.groupby("vid"):
    group = group.sort("frame")
    val_pred_probas_ids.append(vidId)
    val_pred_probas.append(group["probas"].tolist())
    val_pred_probas_gt.append(group["class"].tolist())

#########################
# HMM modelling
#########################

print "Modelling with HMM"

# Computing start statistics vector (pi)
pi = np.zeros(p)
for steps in videos:
    pi[step_index[steps[0]]] += 1
pi /= len(videos)

# Computing transitions matrix (A)
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

print "pi:",  pi
print "A:", A
print "mus:", mus
# print "sigmas", sigmas

#########################
# Decoding val sequences
#########################

print "Start decoding..."
if options.save:
    fig = plt.figure(figsize=(15,15))

for valInd in range(len(val_pred_probas_ids)):

    print "> Decoding video %d" % valInd
    vidInd = val_pred_probas_ids[valInd]
    probas = np.array(val_pred_probas[valInd])
    probas_cnn = np.array(probas)
    steps_gt = val_pred_probas_gt[valInd]

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
    target_len = videos_true_length[vidInd]

    steps_hat_cnn = list(chain(*zip(*[steps_hat_cnn for _ in range(25)])))
    steps_hat_cnn_smooth = list(chain(*zip(*[steps_hat_cnn_smooth for _ in range(25)])))
    steps_hat_hmm = list(chain(*zip(*[steps_hat_hmm for _ in range(25)])))
    steps_hat_hmm_off = list(chain(*zip(*[steps_hat_hmm_off for _ in range(25)])))

    steps_hat_cnn = steps_hat_cnn[0:target_len] # crop if too long
    steps_hat_cnn += [steps_hat_cnn[-1]] * (target_len - len(steps_hat_cnn)) # pad if too short

    steps_hat_cnn_smooth = steps_hat_cnn_smooth[0:target_len] # crop if too long
    steps_hat_cnn_smooth += [steps_hat_cnn_smooth[-1]] * (target_len - len(steps_hat_cnn_smooth)) # pad if too short

    steps_hat_hmm = steps_hat_hmm[0:target_len] # crop if too long
    steps_hat_hmm += [steps_hat_hmm[-1]] * (target_len - len(steps_hat_hmm)) # pad if too short

    steps_hat_hmm_off = steps_hat_hmm_off[0:target_len] # crop if too long
    steps_hat_hmm_off += [steps_hat_hmm_off[-1]] * (target_len - len(steps_hat_hmm_off)) # pad if too short

    if options.save:
        os.system("mkdir -p "+options.dirsave+"/cnn")
        os.system("mkdir -p "+options.dirsave+"/cnn_smooth")
        os.system("mkdir -p "+options.dirsave+"/hmm")
        os.system("mkdir -p "+options.dirsave+"/hmm_off")

        f = open(options.dirsave+"/cnn/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_cnn):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open(options.dirsave+"/cnn_smooth/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_cnn_smooth):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open(options.dirsave+"/hmm/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_hmm):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        f = open(options.dirsave+"/hmm_off/workflow_video_%02d_pred.txt" % (vidInd+1), "w")
        f.write("Frame\tPhase\n")
        for i, step in enumerate(steps_hat_hmm_off):
            f.write("%d\t%s\n" % (i, ordered_unique_steps[step]))
        f.close()

        # plot
        title = "vid %d, " % vidInd
        title += "cnn err. %.1f %%, " % (np.sum(np.array(steps_gt) != np.array(steps_hat_cnn)) * 100.0 / len(steps_gt))
        title += "hmm err. %.1f %%, " % (np.sum(np.array(steps_gt) != np.array(steps_hat_hmm)) * 100.0 / len(steps_gt))
        title += "hmm off err. %.1f %%" % (np.sum(np.array(steps_gt) != np.array(steps_hat_hmm_off)) * 100.0 / len(steps_gt))
        plt.subplot(5,1,valInd+1)
        plt.title(title)
        plt.plot(range(len(steps_hat_cnn)), steps_hat_cnn, label="cnn (output averaged over 5 frames)")
        plt.plot(range(len(steps_gt)), steps_gt, lw=5, label="gt")
        plt.plot(range(len(steps_hat_hmm_off)), steps_hat_hmm_off, lw=3, label="hmm offline")
        plt.plot(range(len(steps_hat_hmm)), steps_hat_hmm, lw=2, label="hmm")
        plt.xlim([0,2500])
        plt.legend(loc="best")

if options.save:
    os.system("mkdir -p "+options.dirsave+"/fig")
    fig.savefig(options.dirsave+"/fig/pred.png")
    plt.close()

    fig=plt.figure(figsize=(3,3))
    plt.imshow(mus, interpolation="nearest")
    plt.title("mu for each class")
    plt.yticks(range(p),ordered_unique_steps)
    plt.xlabel("Avg. output probas of model")
    fig.savefig(options.dirsave+"/fig/mus.png")
    plt.close()

    fig=plt.figure(figsize=(3,3))
    plt.imshow(np.log(A), interpolation="nearest")
    plt.xticks(range(p),ordered_unique_steps, rotation=90)
    plt.yticks(range(p),ordered_unique_steps)
    fig.savefig(options.dirsave+"/fig/A.png")
    plt.close()
