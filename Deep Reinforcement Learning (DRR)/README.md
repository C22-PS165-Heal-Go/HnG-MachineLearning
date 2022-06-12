This is a [TensorFlow](https://www.tensorflow.org/) **implementation to DRR (Deep Reinforcement Learning)**, that I wrote from scratch. It took me solid 10 days to reimplement the code into `Tensorflow Frameworks`.

This project is built by referring to these papers [[1]](https://arxiv.org/pdf/1810.12027.pdf) and [[2]](https://aclanthology.org/P19-1064.pdf)

Also, this project follows the algorithm from [This Repo](https://github.com/irskid5/drr_restaurants), which implementing DRR with
restaurant dataset in PyTorch Framework.

Questions, suggestions, or correctness can be posted as issues.

I'm using `Tensorflow 2.7.0` in `Python 3.9`.

# Contents
[***Objectives***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#objectives)

[***Concepts***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#concepts)

[***Overview***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#overview)

[***Implementation***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#implementation)

[***Training***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#training)

[***Inference***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#inference)

[***Reference***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#reference)

[***TensorFlowTutorial***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#tensorflowtutorial)

# Objectives
**To build a models for a system that can interact with the user and consider both dynamic adaptation and long-term rewards**

* Firstly, most of them consider the recommendation procedure as a static process, for example,
they assume the user’s underlying preference keeps unchanged. However, it is very common 
that a user’s preference is dynamic with respect to time, for example, a user’s
preference on previous items will affect her choice on the next
items. Hence, it would be more reasonable to model the recommendation as a sequential decision making process

* Secondly, the immediate and long-term rewards need to be taken into account, what does this mean?
let's take an example, first if someone got a recommendation about a mountain and a beach, in this example
after going to a mountain, the user probably is not willing going to a mountain again, while in the other hand
the user like beach more, then the user will more likely to get more recommendation with different beaches.

# Concepts
In this section i will present some introduction to reinforcement learning

The essential underlying model for reinforcement learning is Markov Decision Process(MDP). MDP is defined as:
* **State**, in this model, state refers to positive interaction between user and recommendation, for example, last liked items
* **Action** in this model, action refers to action that the model will take based one the state
* **Transition Function** in this model, transition function is determined by feedbacks reward from the user
* **Reward Function** in this model, reward function is the user feedbacks that come from calculated action
* **Discount Rate** in this model, the discount rate can be described such as, item today worth more than tomorrow item

Basically the recommender user-item interaction can be described such as:
* Considering the current state and the last immediate feedbacks, the **model** will calculate the **action**.
* The result of the **action** is the vector parameter that will be used for calculate item rankings.
* The top rankings items will be recommended to the user.
* The users will give their feedbacks or **rewards** on the recommended items.
* Lastly, the model will update the **state** according to **the rewards** given.

# Overview
In this section, I will present the models that is used in DRR frameworks. For all the models, I create a `model.py` file,([Here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/model.py)) just to make
things more manageable. Before I place all the models into one file, I create each model in jupyter notebook first.

These are the list to the jupyter notebook model
* [PMF Module](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/PMF_TensorFlow.ipynb)
* [State Representation Module](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/DRR_Ave_Models_Tensorflow.ipynb)
* [Actor and Critic Module](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/Actor-Critic_Tensorflow.ipynb)

# Implementation

### Datasets
I am or we are using our own [dataset](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL). It took us about seven days to collect the dataset and clean the dataset.
This dataset consist of nearly 75.000 rows of data user that rates a destination. 

* The ratio I used for splitting the data to train DRR Frameworks is is 80:20, with training and testing respectively.
* The ratio I used for splitting the data to train PMF is 80:10:10, with training, validation, and testing respectively.

In the early stages, I still use this dataset [data_RL_25000](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL/data_RL_25000.npy), [data_RL_75000](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL/data_RL_75000.npy),
this dataset is already converted into numpy type data,
so the only thing you need before use the data is to normalize the rating, using the following expression

`1/2 * (R_i - 3)`

R_i refers to rating in coresponding index.

In the later stages, I use some **undersampling-oversampling methods** just to smoothen a bit the unbalanced data,
in order to do that, I need to split the data into training and test data. The methods only apply on training data,
because of that I need to split it beforehand. The best result of the dataset is the [train_data_RL_75000](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL/train_data_RL_75000.npy) and [test_data_RL_75000](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL/test_data_RL_75000.npy)

### Probabilistic Matrix Factorization(PMF) Module
Probabilistic Matrix Factorization is a module that is used for environment simulator and reward functions, based on this [[3]](https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf). This model then will be used as environment simulator meaning that in this frameworks it will be used to predict an item that has not been yet rated by the coresponding user.

This model is composed of four embedding layers, two of them has 100-dimension, and the other two has 1-dimension each. This number is used based on the result in paper [[1]](https://arxiv.org/pdf/1810.12027.pdf).

This model need to be trained on its own, because it will be our reward function later and act as an environment simulator.

The detail for the PMF training, you can go [here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/PMF_TensorFlow.ipynb)
and if you want to dig deeper on how to create custom loop using TensorFlow framework you can go to [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch). My code is based on that TensorFlow tutorial.

For the dataset, you can use Movielens(100K) or Movielens(1M). Or you can choose the dataset that is already available in this repo. If you want to create your own dataset, you need to make sure the structure of the dataset is following the open publicly dataset such as movielens, jester, etc.

### State Representation Module
State representation module, as the name suggest will work as the state representation that model the interactions between the users and the items.

In my case, the state representation module composed of one attention layer. The call method in the state representation modul consist of some
mathematic operation, those are:
* First, matrix multiplication between the weighted matrix with input item embedding matrix
* Second, element wise multiplication between items embedding matrix and users matrix
* Third, the results from first and second operation are then concatenated with users matrix

The output of the state representation module is a matrix with the shape size of `(3 * feature size)`. In this case, I define the feature size
based on the PMF module before, that is `100-dimension`.

For the model creation, I made in [DRR_Ave_Models_Tensorflow](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/DRR_Ave_Models_Tensorflow.ipynb)
, there are two version of it, you can experiment with your own take.

---
![](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/assets/Diagram%20State%20Representation.png)
---

### Actor Module
Actor module, is also called policy module, this module will calculate an actions based on the state. The action is a weighted vector
which will be used for ranking the possible items/destinations in this case to the users.

Actor module consists of three dense layers, those are: 
* the first and second layers have input shape of `(1, state size)` with **ReLu** activation
* the third layer has the output of a size of the features, then the result is activated with **tanh** activation,

Finally actor module will output the features of size `(features size,)` which in this case is defined at 100. This output then will
be computed with items embeddings and resulting with top ranks item that will be recommended to the user.

Actor module first created is in [Actor_Critic_Tensorflow](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/Actor-Critic_Tensorflow.ipynb).
There is some extra operation such as reshape, just be careful with the shape size, usually the bug comes from the shape size or data type.
For all the models you can check [here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/model.py)

---
![](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/assets/DIagram%20Actor%20Network.png)
---

### Critic Module
Critic module is a Deep Q Network (paper [[4]](https://www.nature.com/articles/nature14236?wm=book_wap_0005) and [tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)). This network will output a Q-value which will reflect the output from actor network.
Basically the critic module is used to train the actor network by updating the actor network based on deterministic policy gradient [[5]](https://arxiv.org/abs/1509.02971).
The inputs to this module are the state and the action from actor module.

Critic module consists of four dense layers. Those are:
* The first layer will have the unit size of the input features, that is the state,
* The second and third layers will have the unit size of the combination of first layer output and action size
* The fourth layer will have a unit size of one, that is the result or q-value.

Critic module first created is in [Actor_Critic_Tensorflow](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/Actor-Critic_Tensorflow.ipynb).
For all the models you can check [here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/model.py)

---
![](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/assets/Diagram%20Critic.png)
---

# Training
For the training, the main code is in `train.py` which is [Here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/train.py)
to be precise it's in the `learn()` method. 

The whole algorithm is following the training procedure described in [[1]](https://arxiv.org/pdf/1810.12027.pdf) and the algorithm from [This Repo](https://github.com/irskid5/drr_restaurants).

There are a lot of differences in implementation between PyTorch framework and TensorFlow framework, but I try my best to make the algorithm as
close as possible in order to get very similar result. For the training can be seen in [DRR-Train](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/DRR-Train.ipynb) (sometimes there is an error viewing this file through github, because
there are too many training logs), and the implemented training is in `train.py` [here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/train.py). I still follow the aforementioned referenced repo by building the training
in object oriented paradigm so that it can be easily managed and easily called.

The pseudocode for the detailed algorithm is shown in the image below. The algorithm basically can be explained such as:
* First, build all the needed modules, those are **state representation module, one actor module, one target actor module, one critic module, and one target critic modules**.
* Secund, build two loops, one for iterating through the user data, and the other one is for the item coresponding to that user.
* Third, in the first loop, you will need to collect some data such as user embeddings, items rated by user, and build a variable history of a previously liked items or just use `HistoryBuffer()`.
* Fourth, in the second loop, you will calculate **a state, an action, a rewards, and the next state**. All of those result will be stored in `NaivePrioritizedReplayMemory()`, don't worry, it is just a **named tuple** data structure. I use that because it is easy to maintain and get the data.
* Fifth, after **N many batches** of those result that have been stored, the `training_step()` method will be called.
* Sixth, in this process the critic and actor parameters will be minimized by critic loss and actor loss respectively.
* Seventh, after all of those, both target networks will be updated by `soft_update()` method. 

---
![](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/assets/Referensi_1_Algoritma.JPG)
---

# Inference
This section will describe on how to run the example simulated demo
* First, download this repository
* Second, run the command prompt shell and locate the current directory 
* Third, to run the code, type the following command `python workflow.py`

# Reference
* [1] Liu, F., Tang, R., Li, X., Zhang, W., Ye, Y., Chen, H., ... & Zhang, Y. (2018). Deep reinforcement learning based recommendation with explicit user-item interactions modeling. arXiv preprint arXiv:1810.12027.
* [2] Liu, F., Guo, H., Li, X., Tang, R., Ye, Y., & He, X. (2020, January). End-to-end deep reinforcement learning based recommendation with supervised embedding. In Proceedings of the 13th International Conference on Web Search and Data Mining (pp. 384-392).
* [3] Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
* [4] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236
* [5] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.

# Tensorflow Tutorial
In this section, I will share the link that I think have helped me in the journey of these implementation
* [Making new Layers and Models via Subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
* [Save and load Keras Models](https://www.tensorflow.org/guide/keras/save_and_serialize)
* [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
* [Introductions to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff)
* [Advanced automatic differentiations](https://www.tensorflow.org/guide/advanced_autodiff)
* [Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) (This is for inspecting the methods)
* [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (This is for inspecting the methods)
