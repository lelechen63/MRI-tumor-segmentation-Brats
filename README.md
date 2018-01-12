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



### Citation

If you use these models in your research, please cite:

	@article{DBLP:journals/corr/ChenSDX17,author    = {Lele Chen and
               Sudhanshu Srivastava and
               Zhiyao Duan and
               Chenliang Xu},title     = {Deep Cross-Modal Audio-Visual Generation},journal   = {CoRR},volume    = {abs/1704.08292},year      = {2017},url       = {http://arxiv.org/abs/1704.08292},archivePrefix = {arXiv},eprint    = {1704.08292},timestamp = {Wed, 07 Jun 2017 14:40:44 +0200},biburl    = {http://dblp.org/rec/bib/journals/corr/ChenSDX17},bibsource = {dblp computer science bibliography, http://dblp.org}
}

### Disclaimer and known issues

0. These codes are implmented in Tensorflow
0. You need to install [Ants](https://github.com/ANTsX/ANTs)
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
