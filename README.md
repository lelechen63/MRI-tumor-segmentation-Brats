# Hierarchical MRI tumor segmentation with densely connected 3D CNN

By [Lele Chen], [Yue Wu], [Adora M. DSouza](https://www.rochester.edu/college/gradstudies/profiles/adora-dsouza.html),[Anas Z. Abidin], [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/), [Axel W. E. Wismuelle](https://www.urmc.rochester.edu/people/27063859-axel-w-e-wismueller).

University of Rochester.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Disclaimer and known issues](#disclaimer-and-known-issues)
0. [Models](#models)
0. [Results](#results)
0. [Third-party re-implementations](#third-party-re-implementations)

### Introduction

This repository contains the original models (dense24, dense48, no-dense) described in the paper "Hierarchical MRI tumor segmentation with densely connected 3D CNN" (http://arxiv.org/abs/1512.03385). These models are those used in [BTRAS2017](http://braintumorsegmentation.org/). 

###Run the code


0. Pre-installation:[Tensorflow](https://www.tensorflow.org/install/),[Ants](https://github.com/ANTsX/ANTs),[nibabel](http://nipy.org/nibabel/),[sklearn](http://scikit-learn.org/stable/),[numpy](http://www.numpy.org/)

0. Download and unzip the training data from [BTRAS2017](http://braintumorsegmentation.org/)

0. In this paper, we only use the glioblastoma (HGG) dataset:
'' python n4correction.py /mnt/disk1/dat/lchen63/spie/Brats17TrainingData/HGG'
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



### Citation

If you use these models in your research, please cite:

	@article{DBLP:journals/corr/ChenSDX17,author    = {Lele Chen and
               Sudhanshu Srivastava and
               Zhiyao Duan and
               Chenliang Xu},title     = {Deep Cross-Modal Audio-Visual Generation},journal   = {CoRR},volume    = {abs/1704.08292},year      = {2017},url       = {http://arxiv.org/abs/1704.08292},archivePrefix = {arXiv},eprint    = {1704.08292},timestamp = {Wed, 07 Jun 2017 14:40:44 +0200},biburl    = {http://dblp.org/rec/bib/journals/corr/ChenSDX17},bibsource = {dblp computer science bibliography, http://dblp.org}
}

### Disclaimer and known issues

0. These codes are implmented in Tensorflow
0. If you want to train these models using this version of tensorflow without modifications, please notice that:
	- You need at lest 12 GB GPU memory.
	- There might be some other untested issues.
	

### Results
0. Result visualization :
	![visualization](https://github.com/lelechen63/Hierarchical-MRI-tumor-segmentation-with-densely-connected-3D-CNN/blob/master/image/result.jpg)

0. Quantitative results:

	model|whole|peritumoral edema (ED)|FGD-enhan. tumor (ET)
	:---:|:---:|:---:|:---:
	Dense24 |0.74| 0.81| 0.80
	Dense48 | 0.61|0.78|0.79
	no-dense|0.61|0.77|0.78
	dense24+n4correction|0.72|0.83|0.81
