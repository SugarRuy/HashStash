# HashStash
A PyTorch Implementation of "Evade Deep Image Retrieval by Stashing Private Images in the Hash Space" accepted by CVPR 2020

It is based on [HashNet](https://github.com/thuml/HashNet) pytorch version, tested on PyTorch 0.3.1.

## Requirements
Python 2.7

PyTorch 0.3.1 (Newer version may work with a little modification but we cannot guarantee.)

A CUDA device

## Datasets
ImageNet, Places365, CIFAR-10, Fashion-MNIST and MNIST are supported. 
### ImageNet
We use a subset that includes 10% of all classes of the ImageNet following HashNet's implementation, which contains only 100 class. You may download them via [thuml's link](https://drive.google.com/open?id=0B7IzDz-4yH_HSmpjSTlFeUlSS00). 

Remember to modify the path in files in [./HashNet/pytorch/data/imagenet/](./HashNet/pytorch/data/imagenet/) directory. 

### Places365
We use a subset that includes 10% of all classes of Places365. Please use this [link](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) to download the whole set. 

Relative path is used. Please check [./HashNet/pytorch/data/places365_standard/](./HashNet/pytorch/data/places365_standard/) directory for where to place the train and the val folders. 

### CIFAR-10, Fashion-MNIST, MNIST
We use the full dataset of these tiny datasets. Since they are all supported by torchvision package, there is no need to download the original dataset.

## Running Our Algorithm

Running our algorithm can be easy to do by simply typing:
```
python myGetAdv.py 
```

However, before typing it, a model file and hash code files for corresponding dataset should be ready to make it run smoothly.  The model file should be saved into ./HashNet/pytorch/snapshot/[job_dataset]/_48bit_/[net]_hashnet/. The hash code files should be saved into ./HashNet/pytorch/src/save_for_load/blackbox/[net]/[job_dataset]/. 

We provide link for downloading all models and hash code files we have used. Here is the [link](https://drive.google.com/open?id=1d_jBCcMfKgJ_dKfWAlwDxQXQGb0r2HqI). (**This method is RECOMMENDED**)

### Optional Step
The trained models are created by *myExtractCodeLabel.py* after the dataset are ready. Run it using ```python myTrain.py ```

The Hash Code files are created from *myExtractCodeLabel.py* after setting up the trained models. Run it using ```python myExtractCodeLabel.py```.
 
Both of them support all the five datasets we mentioned earlier. For the details, you may need to dig into them.

## Contacts and Issues
    Yanru Xiao: [yxiao002@odu.edu](mailto:yxiao002@odu.edu)
    For any further issues, please oepn a new issue on github. 

