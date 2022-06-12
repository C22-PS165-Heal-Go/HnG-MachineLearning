This is a [TensorFlow](https://www.tensorflow.org/) implementation to DRR (Deep Reinforcement Learning), that I wrote from scratch. It took me solid 10 days with full 10-hours each day to reimplement the code from PyTorch to Tensorflow Frameworks.

This project is built by referring to these papers [[1]](https://arxiv.org/pdf/1810.12027.pdf) and [[2]](https://aclanthology.org/P19-1064.pdf)

Also, this project follows the algorithm from [This Repo](https://github.com/irskid5/drr_restaurants), which implemented in PyTorch Framework.

Questions, suggestions, or correctness can be posted as issues.

I'm using `Tensorflow 2.7.0` in `Python 3.9`.

# Contents
[***Objectives***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#objectives)

[***Concepts***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#concepts)

[***Overview***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#overview)

[***Training***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#training)

[***Inference***](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/Deep%20Reinforcement%20Learning%20(DRR)#inference)

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
In this section, I will present the models that is used in DRR frameworks

### Probabilistic Matrix Factorization(PMF) Module
Probabilistic Matrix Factorization is a module that is used to model user and items interactions, based on this [[3]](https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf). This model then will be used as environment simulator meaning that in this frameworks it will be used to predict an item that has not been yet rated by the coresponding user.

This model is composed of four embedding layers, two of them has 100-dimension, and the other two has 1-dimension each. This number is used based on the result in paper [[1]](https://arxiv.org/pdf/1810.12027.pdf).

This model need to be trained on its own, because it will be our reward function later and act as an environment simulator.

The detail for the PMF training, you can go [here](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/PMF_TensorFlow.ipynb)
and if you want to dig deeper on how to create custom loop using TensorFlow framework you can go to [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch). My code is based on that TensorFlow tutorial.

For the dataset, you can use Movielens(100K) or Movielens(1M). Or you can choose the dataset that is already available in this repo. If you want to create your own dataset, you need to make sure the structure of the dataset is following the open publicly dataset such as movielens, jester, etc.

### State Representation Module

### Actor Module

### Critic Module

# Training

# Inference
