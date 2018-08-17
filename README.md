# Fine Tuning
* Create a deep learning virtual machine on Ubuntu, as this comes with tensorflow-gpu preinstalled. It is tensorflow 1.8, and requirements.txt specifies tensorflow 1.7, but this works well.
* Clone my fork of david sandberg's facenet: https://github.com/michaelperel/facenet-1.git
* Run export PYTHONPATH=~/facenet/src
* Do not install requirements.txt -> This includes tensorflow1.7, the cpu version. If you change requirements.txt to specify the tensorflow-gpu, the DLVM throws an error most likely because of mismatching versions of CUDA
* Instead, activate py35 (a virtual environment that comes on the DLVM, with tensorflow-gpu), with source activate py35
* To fine-tune, we first need (1) pretrained model and (2) images
* Get the pretrained model from: https://github.com/davidsandberg/facenet#pre-trained-models, choosing VGGFace2 model
* If you would like to use LFW images, get the images from https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW
* Facenet works on 160x160 images. To "align" the images to this format, run this script: (which is align.sh in my fork) https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW#4-align-the-lfw-dataset
* Alternatively, we pretrain on Arman's prealigned photos. NOTE: these prealigned images may have files such as .DS_Store and others, which are not images, and will cause the program to crash. Delete them first, using the script delete_non_images.py
* Once you have aligned images and the model, we can fine-tune the network on the images.
* This requires modifying code in src/train_softmax.py
* We will retrain the network as a classifier, not using triplet loss. This means that in the pretrained model, there is a fixed amount of output classes in the logit layer. We want to load the pretrained model and weights, but we cannot load the logits layer since we have a different number of image classes (folders with people's names on it) than when the classifier was trained. -> My repo contains this fix (we "restore" everything from the pretrained model except for the last layer)
* When specifying the model to reload, in the models folder there will be many files. Pass in the name {whatever_model}.ckpt-{num} even though that is not actually a file.
* When we fine tune, we do not want to just restore the weights from the pretrained model (except for the last layer). As the last layer changes, big errors will propagate to lower layers in the network and ruin their learned weights.
* Instead, we want to freeze all of the layers except for the last layer, and train that for awhile
* As it trains, models will be saved in the model folder.
* Once this converges, we will load in this checkpoint (keeping the logits layer this time)
* Unfreeze layers just before the last layer and train those. Which layers should we unfreeze? Well, while there are nearly 500 variables that are trainable, since this is a resnet inception network, perhaps a good idea would be to unfreeze everything from the last block on.
* Train this for awhile, and you have fine tuned the model
* Check the diff of my fork versus sandberg's, which shows how to replace logits/fine-tune
* Here is the command I have used to run it.
```python src/train_softmax.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/20180402-114759/ --data_dir ~/test_mtcnn_1/ --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160/ --optimizer ADAM --learning_rate -1 --max_nrof_epochs 150 --keep_probability 0.8 --random_crop --random_flip --use_fixed_image_standardization --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-4 --embedding_size 512 --lfw_distance_metric 1 --lfw_use_flipped_images --lfw_subtract_mean --validation_set_split_ratio 0.05 --validate_every_n_epochs 5 --prelogits_norm_loss_factor 5e-4 --pretrained_model ~/models/facenet/20180402-114759/model-20180402-114759.ckpt-275```
# Face Recognition using Tensorflow [![Build Status][travis-image]][travis]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master
[travis]: http://travis-ci.org/davidsandberg/facenet

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Compatibility
The code is tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test) and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

## News
| Date     | Update |
|----------|--------|
| 2018-04-10 | Added new models trained on Casia-WebFace and VGGFace2 (see below). Note that the models uses fixed image standardization (see [wiki](https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset)). |
| 2018-03-31 | Added a new, more flexible input pipeline as well as a bunch of minor updates. |
| 2017-05-13 | Removed a bunch of older non-slim models. Moved the last bottleneck layer into the respective models. Corrected normalization of Center Loss. |
| 2017-05-06 | Added code to [train a classifier on your own images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images). Renamed facenet_train.py to train_tripletloss.py and facenet_train_classifier.py to train_softmax.py. |
| 2017-03-02 | Added pretrained models that generate 128-dimensional embeddings.|
| 2017-02-22 | Updated to Tensorflow r1.0. Added Continuous Integration using Travis-CI.|
| 2017-02-03 | Added models where only trainable variables has been stored in the checkpoint. These are therefore significantly smaller. |
| 2017-01-27 | Added a model trained on a subset of the MS-Celeb-1M dataset. The LFW accuracy of this model is around 0.994. |
| 2017&#8209;01&#8209;02 | Updated to run with Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.   |

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

## Pre-trained models
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model. The datasets has been aligned using [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Performance
The accuracy on LFW for the model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) is 0.99650+-0.00252. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). Note that the input images to the model need to be standardized using fixed image standardization (use the option `--use_fixed_image_standardization` when running e.g. `validate_on_lfw.py`).
