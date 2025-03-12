wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar #training (138 gb)
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar  #validation
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar #test
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz #devkit w/ labels


mkdir imagenet

mkdir imagenet/train
cd imagenet/train
tar -xvf ILSVRC2012_img_train.tar
cd ..

mkdir imagenet/val
cd imagenet/val
tar -xvf ILSVRC2012_img_val.tar 
cd ..

mkdir imagenet/test
cd imagenet/val
tar -xvf ILSVRC2012_img_test_v10102019.tar
cd ..

mkdir imagenet/devkit
cd imagenet/devkit
tar -xzvf image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
cd ..

rm ILSVRC2012_img_val.tar
rm ILSVRC2012_devkit_t12.tar.gz
rm ILSVRC2012_img_train.tar
rm ILSVRC2012_img_test_v10102019.tar

