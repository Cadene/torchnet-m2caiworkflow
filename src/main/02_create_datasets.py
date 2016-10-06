
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd


# # Load annotations

# In[37]:

videos = []
for i in range(1,28):
    steps = map(lambda x: x.strip().split("\t")[1], open("../annotations/workflow_video_%02d.txt" % i, "r").readlines()[1:])
    videos.append(steps[24::25]) # keep 1 frame out of 25


# In[3]:

unique_steps = list(reduce(lambda s1, s2: s1.union(s2), map(lambda s: set(s), videos)))

# Manual writting of steps to have meaningful order
ordered_unique_steps =     ["TrocarPlacement", "Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
     "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
p = len(ordered_unique_steps)

if len(set(ordered_unique_steps).difference(set(unique_steps))) > 0:
    raise Exception("Sets do not match")
else:
    print ordered_unique_steps


# In[4]:

# This map will be used to get the index of a string step
step_index = {v: k for k, v in zip(range(p), ordered_unique_steps)}


# # Number of elements sanity check
# 
# Let's check if the number of annotations extracted from annotations file match the number of images in the folder. Seems like they don't, there seem to be 1 or 2 frames more than annotations. Whatever, let's ignore the additional images and simply use the annotations.

# In[40]:

videos_annotations_len = []
for steps in videos:
    videos_annotations_len.append(len(steps))


# In[46]:

videos_images_len = []
files = map(lambda x: x.strip().replace(".jpg", "").split("-"), open("../files.txt", "r").readlines())
groups = pd.DataFrame(files).groupby(0)
for name, group in groups:
    videos_images_len.append(len(group[1]))


# In[47]:

print zip(videos_annotations_len, videos_images_len)
print np.array(videos_annotations_len) - np.array(videos_images_len)


# # Data split
# 
# Let's compute a train / val split and write the list of images / annotations in a file

# In[70]:

n = len(videos)
# Test videos
test_inds = set(np.random.choice(n, int(np.floor(n * 0.20))))
train_inds = set(range(n)) - test_inds


# In[77]:

print "Test"
print test_inds
print "Train"
print train_inds


# In[86]:

out_test = ""

for test_ind in test_inds:
    for i, step in enumerate(videos[test_ind]):
        out_test += "workflow_video_%02d-%04d.jpg, %d\n" % (test_ind, i + 1, step_index[step])
        
open("../dataset2/valset.txt", "w").write(out_test)


# In[87]:

out_train = ""

for train_ind in train_inds:
    for i, step in enumerate(videos[train_ind]):
        out_test += "workflow_video_%02d-%04d.jpg, %d\n" % (train_ind, i + 1, step_index[step])
        
open("../dataset2/trainset.txt", "w").write(out_test)

