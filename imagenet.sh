#!/bin/bash

# Download ImageNet dataset
curl -C - -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar  # training (138 GB)
curl -C - -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar  # validation
curl -C - -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar  # test
curl -C - -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz  # devkit w/ labels

# Create directories
mkdir -p imagenet/train imagenet/val imagenet/test imagenet/devkit

# Extract datasets
cd imagenet/train
 tar -xvf ../../ILSVRC2012_img_train.tar
cd ../val
 tar -xvf ../../ILSVRC2012_img_val.tar
cd ../test
 tar -xvf ../../ILSVRC2012_img_test_v10102019.tar
cd ../devkit
 tar -xzvf ../../ILSVRC2012_devkit_t12.tar.gz
cd ../..

# Cleanup (uncomment to remove tar files after extraction)
# rm ILSVRC2012_img_train.tar
# rm ILSVRC2012_img_val.tar
# rm ILSVRC2012_img_test_v10102019.tar
# rm ILSVRC2012_devkit_t12.tar.gz
