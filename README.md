# FAOMCCV

## Project Introduction

This project focuses on optimizing lane-changing decisions in intelligent transportation systems, particularly in MCCV scenarios. Our method enhances fairness, safety, and efficiency by integrating Bayesian networks for fairness assessment, real-time lane detection, speed recommendations, and a KNN-enhanced deep reinforcement learning approach to handle unpredictable traffic conditions.

## Environmental Dependence

The code requires python3 (>=3.8) with the development headers. The code also need system packages as bellow:

numpy == 1.24.3

matplotlib == 3.7.1

pandas == 1.5.3

pytorch == 2.0.0

gym == 0.22.0

sumolib == 1.17.0

traci == 1.16.0

keras == 2.15.0

numpy == 1.24.3

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.

## Project structure introduction
FAOMCCV is the root directory of the project, and the secondary directory includes three folders. The folder "DRL" contains the DRL model code we designed. It mainly uses deep reinforcement learning models to control connected vehicles to perform safe, efficient and fair driving decisions. The folder "section1_fairness" contains the code we designed for fairness. This module mainly uses the improved Gaussian-based Bayesian network to accurately predict the probability of decision fairness. The folder "section2" contains the code we designed for the adaptive decision tree. This module mainly uses the adaptive decision tree to determine the availability of the left lane when overtaking.


## Statement

In this project, due to the different parameter settings such as the location of the connected vehicle, the location of the target vehicle, and the state of surrounding vehicles, etc., the parameters of the reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results. In addition, the parameters of the specific network model refer to the parameter settings in the experimental part of the paper. If you want to know more, please refer to our paper "Fairness-Aware Overtaking Decision Optimization for Mixed Connected and Connectionless Vehicles".