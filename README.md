# Final Project Model
This repo currently contains an implementation of PointNet (https://arxiv.org/abs/1612.00593) as well as code for processing the ModelNet40 dataset. 

# Downloading and Installing
Create a new folder to act as your root directory. In this folder, create a second folder called `data`. Download and extract the ModelNet40 dataset from https://modelnet.cs.princeton.edu/ into this folder. 

Next, clone this repo in the root folder.

```
cd <root folder>
git clone https://github.com/andrewyarovoi/final_model.git
cd final_model
pip install -e .
```

Navigate to the dataset folder and run `convert_off_to_numpy.py` to convert the .off files into numpy arrays. This should create a new folder called `ModelNet40_numpy` in the data folder.

```
cd dataset
python convert_off_to_numpy.py
```

# Running PointNet with Static Dataset

There are two ways to train PointNet. You can resample point clouds from the 3D models before each epoch (dynamic dataset), or you can sample the point clouds once and then use the static set of pointclouds to train the model (static dataset). In our experiments, both give similar results, but the static dataset takes about a tenth of the time to train.

To use the static dataset, you first have to preprocess the data.
Navigate to the dataset folder and run `create_static_dataset.py`. This will take a while to run, but eventually will create a new folder in the `data` folder called `ModelNet40_sampled`.

```
cd dataset
python create_static_dataset.py
```

Now run training with the --static_dataset tag.

```
cd utils
python train_classification.py --static_dataset --nepoch=30 --dataset ..\..\data\ModelNet40_sampled
```

Use `--feature_transform` to use feature transform.

# Running PointNet with Dynamic Dataset

Alternatively, you can run PointNet directly on the dataset and sample points off the model, thereby giving a small amount variability between each epoch. To do this, just ommit the --static_dataset tag.

```
cd utils
python train_classification.py --nepoch=30 --dataset ..\..\data\ModelNet40_numpy
```

# Performance

Both methods give about 88% accuracy on the test set.
