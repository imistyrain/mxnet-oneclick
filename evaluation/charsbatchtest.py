import mxnet as mx
import logging
import numpy as np
# Note: The decoded image should be in BGR channel (opencv output)
# For RGB output such as from skimage, we need to convert it to BGR
# WRONG channel will lead to WRONG result
from skimage import io, transform
from sklearn import metrics
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

prefix = "chars/lenetweights"
num_round = 50
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=40)

mean_img = mx.nd.load("chars/mean.bin")["mean_img"]

synset = [l.strip() for l in open('chars/synset.txt').readlines()]

batch_size=1
data_shape = (3, 20, 20)
test = mx.io.ImageRecordIter(
    path_imgrec = "charstest/test.rec",
    mean_img="chars/mean.bin",
    rand_crop   = False,
    rand_mirror = False,
    data_shape  = data_shape,
    batch_size  = batch_size)
with open("charstest/chars.lst","r")as f :
    lsts =f.readlines()
labels=[]
for line in lsts:
    lst =line.split()
    #print lst[1]
    labels.append(int(lst[1]))

preds =[]
predictions = model.predict(test)
for prediction in predictions:
    pred=np.argsort(prediction)[::-1]
    #print synset[pred[0]]
    preds.append(pred[0])

labels=labels[:len(preds)]
print metrics.precision_score(preds,labels)