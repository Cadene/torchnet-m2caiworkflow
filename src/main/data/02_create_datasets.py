# coding: utf-8
import numpy as np
import pandas as pd
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dirannot", default="data/raw/annotations")
parser.add_option("--dirinterim", default="data/interim", help="Directory to save files.txt, trainset.txt and testset.txt")

(options, args) = parser.parse_args()

# # Load annotations

videos = []
for i in range(1,28):
    steps = map(lambda x: x.strip().split("\t")[1], open(options.dirannot+"/workflow_video_%02d.txt" % i, "r").readlines()[1:])
    videos.append(steps[24::25]) # keep 1 frame out of 25

unique_steps = list(reduce(lambda s1, s2: s1.union(s2), map(lambda s: set(s), videos)))

# Manual writting of steps to have meaningful order
ordered_unique_steps = ["TrocarPlacement", "Preparation", "CalotTriangleDissection",
                        "ClippingCutting", "GallbladderDissection",
                        "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
p = len(ordered_unique_steps)

if len(set(ordered_unique_steps).difference(set(unique_steps))) > 0:
    raise Exception("Sets do not match")
else:
    print ordered_unique_steps

# This map will be used to get the index of a string step
step_index = {v: k for k, v in zip(range(p), ordered_unique_steps)}


# # Number of elements sanity check
#
# Let's check if the number of annotations extracted from annotations file match
# the number of images in the folder. Seems like they don't, there seem to be 1 or 2
# frames more than annotations. Whatever, let's ignore the additional images and simply use the annotations.

videos_annotations_len = []
for steps in videos:
    videos_annotations_len.append(len(steps))

videos_images_len = []
files = map(lambda x: x.strip().replace(".jpg", "").split("-"), open(options.dirinterim+"/files.txt", "r").readlines())
groups = pd.DataFrame(files).groupby(0)
for name, group in groups:
    videos_images_len.append(len(group[1]))

print zip(videos_annotations_len, videos_images_len)
print np.array(videos_annotations_len) - np.array(videos_images_len)


# # Data split
#
# Let's compute a train / val split and write the list of images / annotations in a file

n = len(videos)
# Test videos
test_inds = set(np.random.choice(n, int(np.floor(n * 0.20))))
train_inds = set(range(n)) - test_inds

print "Test"
print test_inds
print "Train"
print train_inds

out_test = ""
for test_ind in test_inds:
    for i, step in enumerate(videos[test_ind]):
        out_test += "workflow_video_%02d-%04d.jpg, %d\n" % (test_ind, i + 1, step_index[step])
open(options.dirinterim+"/valset.txt", "w").write(out_test)


out_train = ""
for train_ind in train_inds:
    for i, step in enumerate(videos[train_ind]):
        out_test += "workflow_video_%02d-%04d.jpg, %d\n" % (train_ind, i + 1, step_index[step])
open(options.dirinterim+"/trainset.txt", "w").write(out_test)
