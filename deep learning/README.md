# DEEP LEARNING 
This is repository for Deep Learning Part


## Contents
- Objective
- Library
- Implementation
- Training

---------------------------------------------
## Objective

<br>

In this project, Deep Learning works to determine 5 initial recommendations whose input is obtained from a questionnaire filled out by the user.

The model architecture used is only Multilayer Perceptron (MLP) which uses Dense Layer and Dropout layer.
The optimizer used is adam and the loss is sparsecategoricalentrophy.

The output of the model is 5 tourist destinations which will then be used in Deep Reinforcement Learning modeling.

---------------------------------------------
## Library
Below are the libraries we used to build the Deep Learning model.
- Pandas : for data manipulation
- Tensorflow : for build and training model
- Imblearn : for random oversampling data
- Matplotlib : for create graph
- Sklearn : for normalization and encode data

---------------------------------------------
## Implementation
This section contains an explanation of the implementation of the Deep Learning model.

### Dataset
Dataset for Deep Learning uses a dataset that we created ourselves by distributing questionnaires to the public. The questionnaire contains 9 questions related to their travel experiences. Details of these questions can be accessed [at the following link](https://docs.google.com/forms/d/e/1FAIpQLSfZD-CX_XJtDl3ICQhAMbml38lpW8Bp6xKXv8z_xoz-qYb0ng/viewform).
There are more than 1000 data records from respondents. However, collecting 1000 records of this data is not easy. It took us approximately 1 week to reach that amount.

However, the data still needs to be preprocessed again. The final dataset can be accessed via the [following link](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/dataset/questionnaire_dataset3.csv).

### Preprocess
