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
After getting the dataset, the next step is preprocessing. In this dataset there are several columns that must be deleted such as **Timestamp** and **Email** because they are not used. This dataset has no missing values, so we don't need to fill in the blanks in the dataset.

#### Encode Data
Some fields like **Activity** and **Member's Age** have multivalues. So that in the data encode process, the method used is multilabel encode using `MultiLabelBinarizer()`. The results obtained are in the form of a list of numbers.

For the other columns encode the data using `LabelEncode()`. `LabelEncode()` first collects all the values from a column, sorts them alphabetically and then converts them to numbers. As a result, the value of the column which was originally a string will change to an int according to the index of the value.

In the process of converting the values that have been encoded into tensor form, the value from the multilabel encode experienced an error because it was in the form of a list, so we decided to create a new column containing the values from the multilabel encoded result.

#### OverSampling
After doing several times of training, the accuracy and loss obtained are not so good. The difference between train accuracy and validation accuracy is very large, up to 50%. As shown in the image below.

We suspect that this is **Variance Error**. Our dataset is indeed imbalanced, many classes have very little data. So, to reduce the Variance Error several ways that can be done are to get more data, perform data augmentations, dropouts, regularization, and modify the model architecture.

We use `RandomOverSample()` to oversampling the data. This oversampling makes the amount of data that was initially only around 1000 data, is sampled to 4400 data.

This method is quite successful because it can reduce the variance although the accuracy is still not very good.
