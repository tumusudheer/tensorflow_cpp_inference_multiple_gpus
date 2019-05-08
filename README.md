# tensorflow_cpp_inference_multiple_gpus
TF inference using multiple gpus in C++
------------

## Compiling
```
git clone https://github.com/tensorflow/tensorflow.git
git checkout r1.12
./configure

bazel build --config=monolithic --config=cuda //tensorflow:libtensorflow_cc.so
```
