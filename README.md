# Domain Impression: A Source Data Free Domain Adaptation Method (SFDA)

Torch Lua code for SFDA model. For more information, please refer the[ [Paper] ](https://openaccess.thecvf.com/content/WACV2021/papers/Kurmi_Domain_Impression_A_Source_Data_Free_Domain_Adaptation_Method_WACV_2021_paper.pdf)

Accepted at [[WACV 2021](http://wacv2021.thecvf.com/home)]

#####  [[Project]](https://delta-lab-iitk.github.io/SFDA//)     [[Paper Link ]](https://arxiv.org/abs/2102.09003)

#### Abstract
Unsupervised Domain adaptation methods solve the adaptation problem for an unlabeled target set, assuming that the source dataset is available with all labels. However, the availability of actual source samples is not always possible in practical cases. It could be due to memory constraints, privacy concerns, and challenges in sharing data. This practical scenario creates a bottleneck in the domain adaptation problem. This paper addresses this challenging scenario by proposing a domain adaptation technique that does not need any source data. Instead of the source data, we are only provided with a classifier that is trained on the source data. Our proposed approach is based on a generative framework, where the trained classifier is used for generating samples from the source classes. We learn the joint distribution of data by using the energy-based modeling of the trained classifier. At the same time, a new classifier is also adapted for the target domain. We perform various ablation analysis under different experimental setups and demonstrate that the proposed approach achieves better results than the baseline models in this extremely novel scenario.

![Result](https://delta-lab-iitk.github.io/SFDA/img/model.png)


### Requirements
This code is written in Lua and requires [Torch](http://torch.ch/).


You also need to install the following package in order to sucessfully run the code.
- [Torch](http://torch.ch/docs/getting-started.html#_)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


#### Download Dataset
- [MNIST]
-[MNIST-M]

##### Prepare Datasets
- Download the dataset


### Training Steps

We have prepared everything for you ;)

####Clone the repositotry

``` git clone https://github.com/DelTA-Lab-IITK/SFDA  ```

#### Train model
```
cd SFDA/
./train.sh
```




### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@InProceedings{Kurmi_2021_WACV,
    author    = {Kurmi, Vinod K. and Subramanian, Venkatesh K. and Namboodiri, Vinay P.},
    title     = {Domain Impression: A Source Data Free Domain Adaptation Method},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {615-625}
}
```

## Contributors
* [Vinod K. Kurmi][1] (vinodkumarkurmi@gmail.com)



[1]: https://github.com/vinodkkurmi




