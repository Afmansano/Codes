# Project 1: Navigation

### Introduction

This is the impelementation of Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)   Navigation project. 

For the environment we use the [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents) Banana environment.

The Banana environment is a 3D "box" with randomly distributed bananas.  These bananas are either yellow or blue, and the agent's goal is to collect as many as the yellow ones while avoiding the blues. It receives reward of +1 is when collecting a yellow banana, and a reward of -1 for collecting a blue banana.

The table below depicts the environment:

![banana_env](env.gif)



#### State Space

The environment state space is composed of a 37-dimensional np-array.:

```python
States look like: [ 1.          0.          0.          0.          0.84408134  0.          0.
                    1.          0.          0.0748472   0.          1.          0.          0.
                    0.25755     1.          0.          0.          0.          0.74177343
                    0.          1.          0.          0.          0.25854847  0.          0.
                    1.          0.          0.09355672  0.          1.          0.          0.
                    0.31969345  0.          0.        ]
States have length: 37
```


#### Action Space

For this environment we have 4 different actions, integers in the range [0,3] that represents the direction of movement:

- `0` forward
- `1` backward
- `2` left.
- `3` right.

#### Solving the environment

The Banana environment is considered "solved" when the agent scores an average of +13 points over a single 100-episode block.


## Getting started

### Installation requirements

- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
- Of course you have to clone this project and have it accessible in your Python environment
- Then you have to install the Unity environment as described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) (The Unity ML-agant environment is already configured by Udacity)

  - Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.


- Finally, unzip the environment archive in the 'project's environment' directory and eventually adjust thr path to the UnityEnvironment in the code.

### Train the agent
    
Execute the navigation.ipnb notebook within this Nanodegree Udacity Online Workspace for "project #1  Navigation" (or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code )
