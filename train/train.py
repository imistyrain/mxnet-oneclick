import mxnet as mx
import logging

datasetname="chars"
datadir="../platechars"

def get_lenet1():
    ## define lenet
    # input
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=50)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3,3), num_filter=100)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    lenet = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return lenet
def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=34)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet


def main():
    batch_size=1000
    num_epoch = 200
    num_gpus = 1
    logging.basicConfig(level=logging.DEBUG)
    gpus = [mx.gpu(i) for i in range(num_gpus)]
    lenet=get_lenet()
    model = mx.model.FeedForward(ctx=gpus, symbol=lenet, num_epoch=num_epoch,
                                     learning_rate=0.01, momentum=0.9, wd=0.0001,
                                     initializer=mx.init.Uniform(0.07))
    train_dataiter = mx.io.ImageRecordIter(
            path_imgrec=datadir+"/"+datasetname+"_train.rec",
            mean_img=datadir+"/mean.bin",
            rand_crop=True,
            rand_mirror=True,
            data_shape=(3,20,20),
            batch_size=batch_size,
            preprocess_threads=1)
    test_dataiter = mx.io.ImageRecordIter(
            path_imgrec=datadir+"/"+datasetname+"_val.rec",
            mean_img=datadir+"/mean.bin",
            rand_crop=False,
            rand_mirror=False,
            data_shape=(3,20,20),
            batch_size=batch_size,
            preprocess_threads=1)
    model.fit(X=train_dataiter, eval_data=test_dataiter,
              batch_end_callback=mx.callback.Speedometer(100))
    model.save(datadir+'/lenetweights',num_epoch)
if __name__=="__main__":
    main()