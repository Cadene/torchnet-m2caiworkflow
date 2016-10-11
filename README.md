# M2CAI Workflow Challenge

This is the repo for the M2CAI Workflow Challenge of the following fellow PhD students :

- The well known [Thomas Robert](http://www.thomas-robert.fr/en/), I mean if you search its name on Google it will be the first website, that's badass.
- The super hero of fine tuning [Remi Cadene](http://remicadene.com), the equivalent of Neo in the great movie Matrix.

You can find our submitted paper in this repo: [PDF](https://github.com/Cadene/torchnet-m2caiworkflow/raw/master/docs/m2cai_workflow_lip6_report.pdf)

Do not hesitate to contact us. If you have any questions about our code and/or paper, please create an issue. For other queries, please send us an email.

### Dependencies

To train CNN:

- torch
- torchnet
- torchnet-vision
- (gpu) cuda
- (gpu) cudnn

To train HMM:

- python
- numpy
- pandas
- sklearn
- matplotlib
- seaborn

To evaluate with Jaccar score:

- matlab

### 1. Extracting data

First we have to unzip `train_workflow_challenge_m2cai2016.zip` and `test_workflow_challenge_m2cai2016.zip` in `data/raw`.

Then, in `data/interim`, you extract images from videos and create several files that you will be using for training CNNs and HMMs.

```
$ ./src/main/data/01_extract_images.sh data/raw
$ python 02_create_datasets.py
$ python 03_create_testset.py
```

### 2. Training a Convolutional Neural Network (CNN)

First we train a CNN and we save the best model regarding the validation score for each epochs.
The best set of hyperparameters is the default set of options for each models.

#### Fine tuning ResNet-200 (best result)

```
$ th src/main/classif/resnet.lua
```

#### Fine tuning Inception-V3

```
$ th src/main/classif/inceptionv3.lua
```

#### Fine tuning Inception-V3 with Weldon

```
$ th src/main/classif/inceptionv3weldon.lua
```

#### Inception-V3 as features extractor

The first way consists of fixing the parameters of the pretrain layers and to train only the last layer.

```
$ th src/main/classif/inceptionv3extraction.lua
```

The other way (often the usual way) consists of extracting features of a pretrain CNN on ImageNet and then to train a SVM. We only provide a code to extract features to a .csv.

```
$ th src/main/classif/extraction.lua
```


### 3. Extracting predictions from a CNN

Then we extract the log probabilities from a trained CNN on the training set, validation set and testing set.

```
$ th src/main/classif/extractpreds.lua \
   -pathnet experiments/resnet/datetime/net.t7 \
   -pathpreds experiments/resnet/datetime
```

### 4. Training a Hidden Markov Model (HMM)

Once we have extracted the log probabilities, we can train a Gaussian HMM on the training set and compute predictions on the validation set and testing set. We provide 4 temporal predictions :

- cnn: no smoothing
- cnn_smooth: smoothing on 5 frames
- hmm: smoothing on 5 frames then smoothing with hmm in online mode
- hmm: same but in offline model

TODO EXPLAIN

```
$ python src/main/temporal/hmmval.py
$ python src/main/temporal/hmmtest.py
```

### 5. Evaluating predictions

Finally, to evaluate the temporal predictions, we must use the code given for the challenge. We can edit several variables: `phaseGroundTruths`, `phaseGroundThruth` and `predFile`.

```
$ matlab -nodisplay -nodesktop -r "run src/main/eval/Main.m"
```
