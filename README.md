# Hierarchical MRI tumor segmentation with densely connected 3D CNN

By Lele Chen, Yue Wu, [Adora M. DSouza](https://www.rochester.edu/college/gradstudies/profiles/adora-dsouza.html),Anas Z. Abidin, [Axel W. E. Wismuelle](https://www.urmc.rochester.edu/people/27063859-axel-w-e-wismueller), [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/).

University of Rochester.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Running](#running)
0. [Model](#model)
0. [Disclaimer and known issues](#disclaimer-and-known-issues)
0. [Results](#results)

### Introduction

This repository contains the original models (dense24, dense48, no-dense) described in the paper "Hierarchical MRI tumor segmentation with densely connected 3D CNN" (https://arxiv.org/abs/1802.02427). This code can be applied directly in [BTRAS2017](http://braintumorsegmentation.org/). 

![model](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/spie.gif)


### Citation

If you use these models or the ideas in your research, please cite:
	
	@inproceedings{lchen63spie,
	Author = {Lele, Chen and Yue, Wu and Adora M., DSouza and Anas Z., Abidin and Axel, Wismuller and Chenliang, Xu},
	Booktitle = {arXiv:1802.02427},
	Date-Added = {2018-01-18 05:16:26 +0000},
	Date-Modified = {2018-01-18 05:16:26 +0000},
	Title = {MRI Tumor Segmentation with Densely Connected 3D CNN},
	Venue = {ARXIV},
	Year = {2018}}
### Running


0. Pre-installation:[Tensorflow](https://www.tensorflow.org/install/),[Ants](https://github.com/ANTsX/ANTs),[nibabel](http://nipy.org/nibabel/),[sklearn](http://scikit-learn.org/stable/),[numpy](http://www.numpy.org/)

0. Download and unzip the training data from [BTRAS2017](http://braintumorsegmentation.org/)

0. Use N4ITK to correct the data: `python n4correction.py /mnt/disk1/dat/lchen63/spie/Brats17TrainingData/HGG`
0. Train the model:  `python train.py`
	- `-gpu`: gpu id
	- `-bs`: batch size 
	- `-mn`: model name, 'dense24' or 'dense48' or 'no-dense' or 'dense24_nocorrection'
	- `-nc`:  [n4ITK bias correction](https://www.ncbi.nlm.nih.gov/pubmed/20378467),True or False
	- `-e`: epoch number 
	- `-r`: data path
	- `-sp`: save path/name
	- ...

For example:
`python train.py -bs 2 -gpu 0  -mn dense24 -nc True -sp dense48_correction -e 5  -r /mnt/disk1/dat/lchen63/spie/Brats17TrainingData/HGG`

0. Test the model: `python test.py`
	- `-gpu`: gpu id
	- `-m`: model path, the saved model name
	- `-mn`: model name, 'dense24' or 'dense48' or 'no-dense' or 'dense24_nocorrection'
	- `-nc`:  [n4ITK bias correction](https://www.ncbi.nlm.nih.gov/pubmed/20378467), True or False
	- `-r`: data path
	- ...

For example:
`python test.py -m Dense24_correction-2 -mn dense24 -gpu 0 -nc True  -r /mnt/disk1/dat/lchen63/spie/Brats17TrainingData/HGG`


### Model

0. Hierarchical segmentation
	![model](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/2.png)

	
0. 3D densely connected CNN

	![model](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/1.png)

### Disclaimer and known issues

0. These codes are implmented in Tensorflow
0. In this paper, we only use the glioblastoma (HGG) dataset.
0. I didn't config nipype.interfaces.ants.segmentation. So if you need to use `n4correction.py` code, you need to copy it to the bin directory where antsRegistration etc are located. Then run `python n4correction.py`
0. If you want to train these models using this version of tensorflow without modifications, please notice that:
	- You need at lest 12 GB GPU memory.
	- There might be some other untested issues.
	

### Results
0. Result visualization :
	![visualization](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/h.png)
	![visualization](https://github.com/lelechen63/MRI-tumor-segmentation-Brats/blob/master/image/v.png)

0. Quantitative results:

	model|whole|peritumoral edema (ED)|FGD-enhan. tumor (ET)
	:---:|:---:|:---:|:---:
	Dense24 |0.74| 0.81| 0.80
	Dense48 | 0.61|0.78|0.79
	no-dense|0.61|0.77|0.78
	dense24+n4correction|0.72|0.83|0.81
