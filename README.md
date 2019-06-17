# DeepLabV3plus
This repo is an implementation of network described in [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611). DeepLab is a deep learning model for semantic image segmentation.

The implementation of the authors of the paper can be found here [here](https://github.com/tensorflow/models/tree/master/research/deeplab).

# Results
The figure below is the result of training Deeplabv3+ on [SYNTHIA dataset](http://synthia-dataset.net/)
![alt text](https://github.com/makashy/DeepLabV3plus/blob/master/images/train_on_SYNTHIA.png)

Batch loss:
![batch_loss](https://github.com/makashy/DeepLabV3plus/blob/master/images/batch_loss.png)

# Model Structure
![redesigned](https://github.com/makashy/DeepLabV3plus/blob/master/images/redesigned.png)

## 1. Xception 41

### 1.1. Entry flow
![entry_flow](https://github.com/makashy/DeepLabV3plus/blob/master/images/entry_flow.png)

### 1.2. Middle flow
![middle_flow](https://github.com/makashy/DeepLabV3plus/blob/master/images/middle_flow.png)

### 1.3. Exit flow
![exit_flow](https://github.com/makashy/DeepLabV3plus/blob/master/images/exit_flow.png)

## 2. Atrous Spatial Pyramid Pooling(ASPP)
![aspp](https://github.com/makashy/DeepLabV3plus/blob/master/images/aspp.png)

## 3. Decoder
![decoder](https://github.com/makashy/DeepLabV3plus/blob/master/images/decoder.png)

## 4. Logits
![Logits](https://github.com/makashy/DeepLabV3plus/blob/master/images/logits.png)
