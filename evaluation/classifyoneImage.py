import mxnet as mx
import logging
import numpy as np
import os
# Note: The decoded image should be in BGR channel (opencv output)
# For RGB output such as from skimage, we need to convert it to BGR
# WRONG channel will lead to WRONG result
from skimage import io, transform
from sklearn import metrics
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
testdir="../platechars"
prefix = testdir+"/lenetweights"
num_round = 200
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

mean_img = mx.nd.load(testdir+"/mean.bin")["mean_img"]

#mx.viz.plot_network(model.symbol,shape={"data",(1,1,20,20)})
synset = [l.strip() for l in open(testdir+'/synset.txt').readlines()]
batch_size=1
data_shape = (3, 20, 20)
def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (20, 20))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
   #sample = np.swapaxes(sample, 0, 2)
    #sample = np.swapaxes(sample, 1, 2)
    # sub mean 
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3,20, 20)
    return normed_img
batch = PreprocessImage(testdir+'/0/0.jpg', True)
prediction = model.predict(batch)[0]
label=synset[prediction.argmax()]
print str(label)+':'+str(prediction[label])
