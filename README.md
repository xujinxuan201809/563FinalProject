# 563FinalProject
In this project, we develope a 3 layer and a 4 layer Neural Network in CPU and GPU and record the running time respectively. 
As for the data, we use the 42'000 Handwritten Images as our training data from the MNIST dataset. And each image has 784 dimensions. The batch size, the number of images input each time, is 256.
And each neural network has 500 units per hidden layer.

CPU_3LAYER_500.cpp

This file includes code of three layer neural network build on CPU.

CPU_4LAYER_500.cpp

This file includes code of four layer neural network build on CPU.

3layer_500.cu

This file includes code of three layer neural network build on GPU.

4layer_500.cu

This file includes code of four layer neural network build on CPU.

Train.txt

This file includes 42'000 handwritten images as training data for CPU program.

Train1.txt

Trainning data file for GPU program.

# Running CPU experiment
Download the "train.txt", "CPU_3LAYER_500.cpp", "CPU_4LAYER_500.cpp".

Before running the program, you have to do two things below.

First, you have to change the "train.txt" file path which should be the path in your computer manually in each program, specifically in the beginning of main function. 

Second, this program will generate a txt file including the results, so you have to change the result file path manually wherever you want.

# Running GPU experiment
First you need to make sure your computer has built-in NVIDIA GPU. Then a CUDA version of at least 9.2 should be installed which can be found in the website https://developer.nvidia.com/cuda-downloads.

Download the "train1.txt", "3layer_500.cu", "4layer_500.cu".

In order to input the trainning data from the file, you need to change the "train1.txt" file path in code manually. This program will output one result .txt file, and you need to change the output file path to you computer.

Then input command below:
nvcc -arch=sm_50 -std=c++11 -rdc=true -lcudadevrt "train1.txt"path -o ./4L_500

./4L_500
