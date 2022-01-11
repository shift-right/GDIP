# GDIP: A Fast Deep Image Prior Framework with Guided Filtering
Combine Deep Image Prior and Guided Filtering

In this repository we provide *Jupyter Notebooks* to reproduce each figure from the paper:

> **Deep Image Prior**  
> CVPR 2018  
> Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky  
[[paper]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)  
[[code]](https://github.com/DmitryUlyanov/deep-image-prior)  

> **Deep Hyperspectral Prior: Single Image Denoising, Inpainting, Super-Resolution**  
> ICCV 2019  
> O Sidorov, JY Hardeberg  
[[paper]](https://arxiv.org/abs/1902.00301) 
[[code]](https://github.com/acecreamu/deep-hs-prior)  

> **Fast End-to-End Trainable Guided Filter**  
> CVPR 2018  
> Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang  
[[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Fast_End-to-End_Trainable_CVPR_2018_paper.pdf)  
[[code]](https://github.com/wuhuikai/DeepGuidedFilter)  


# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.8.11
- [pytorch](http://pytorch.org/) = 1.9.1
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```


or create an conda env with all dependencies via environment file

```
conda env create -f environment.yml
```
