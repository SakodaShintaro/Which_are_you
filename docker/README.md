# How to build
docker build -t which_are_you_image:20230107 .
docker run --gpus all -it --ipc=host --name which_are_you_container which_are_you_image:20230107 bash

# Download libtorch
../shellscript/download_libtorch.sh 

# Set path to libtorch/lib
cat LD_LIBRARY_PATH=/root/libtorch-1.11.0/lib/ >> ~/.bashrc
