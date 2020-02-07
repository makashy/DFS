# DeepLabV3plus
This repo is an implementation of network described in [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611). DeepLab is a deep learning model for semantic image segmentation.

The implementation of the authors of the paper can be found here [here](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Install
conda install -c anaconda pytables\
conda install -c anaconda tensorflow-gpu\
conda install -c anaconda scikit-image \
conda install -c anaconda pandas \
conda install -c anaconda jupyterlab \
conda install -c anaconda numba \
conda install -c anaconda opencv \

In case you want to use [lyft dataset](https://level5.lyft.com/dataset/#data-collection):\
conda install -c conda-forge fire cachetools
conda install -c anaconda black flake8 matplotlib numpy Pillow plotly
conda install -c anaconda pyquaternion pytest scikit-learn scipy Shapely tqdm

Not needed: \
conda install -c open3d-admin open3d \
conda install -c conda-forge nodejs \
jupyter labextension install @jupyter-widgets/jupyterlab-manager \
conda install -c anaconda pylint \
conda install -c conda-forge yapf \
conda install -c conda-forge nb_conda \