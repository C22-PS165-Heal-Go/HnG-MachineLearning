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

### State Representation Module

### Actor Module

### Critic Module

# Training

# Inference
