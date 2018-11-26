"""
MxNet:
Conv2D > MaxPooling > Relu Activation > Conv2D > AvgPooling > Relu Activation > Conv2D > AvgPooling > Relu Activation > Flatten > F.Connected * 4 > Concat > Softmax
( ref Also has LSTM+CTC )
https://blog.csdn.net/u013203733/article/details/79140499


Modified LeNets:
# network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->max_pool2d->fully_connected->fully_connected
https://www.cnblogs.com/skyfsm/p/8443107.html#!comments

VGG16 ?
https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

(multi-label classification)
smaller VGG
https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/



"""