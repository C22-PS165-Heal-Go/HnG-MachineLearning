# DEEP LEARNING 
This is repository for Deep Learning Model. The development of this Deep Learning model took about 5 days, but the process of fine-tuning the model took about 4 days. Deploy model is done on [this repository](https://github.com/C22-PS165-Heal-Go/HnG-ML_Runner).

We use `Python 3.7` and `tensorflow 2.8` to build this model.


## Contents
- [Objective](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/edit/main/deep%20learning/README.md#objective)
- [Library](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/deep%20learning#library)
- [Implementation](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/deep%20learning#implementation)
- [Training](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/deep%20learning#training)
- [References](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/edit/main/deep%20learning/README.md#references)

<br>

## Objective
In this project, Deep Learning works to determine 5 initial recommendations whose input is obtained from a questionnaire filled out by the user.

The model architecture used is only Multilayer Perceptron (MLP) which uses Dense Layer and Dropout layer.
The optimizer used is adam and the loss is sparsecategoricalentrophy.

The output of the model is 5 tourist destinations which will then be used in Deep Reinforcement Learning modeling.

<br>

## Library
Below are the libraries we used to build the Deep Learning model.
- Pandas : for data manipulation
- Tensorflow : for build and training model
- Imblearn : for random oversampling data
- Matplotlib : for create graph
- Sklearn : for normalization and encode data

<br>

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

<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/prev_acc.png" alt="Acc">
</p>

We suspect that this is **Variance Error**. Our dataset is indeed imbalanced, many classes have very little data. So, to reduce the Variance Error several ways that can be done are to get more data, perform data augmentations, dropouts, regularization, and modify the model architecture.

We use `RandomOverSample()` to oversampling the data. This oversampling makes the amount of data that was initially only around 1000 data, is sampled to 4400 data.

This method is quite successful because it can reduce the variance although the accuracy is still not very good.

#### Spliting Data
With a total of about 4400 data. The data is divided into 3 parts, namely **train data**, **validation data**, and **test data** with each ratio of 70:20:10. Train data obtained is 3099 data.

We also divide the data into **x** and **y**. Class or label is entered into the variable **y**, while for the other columns it is entered into the variable **x**.

#### Normalization Data
After training, the accuracy obtained after oversampling is quite good than before, but we think it can still be improved.

We are trying to normalize the data by using `MinMaxScaler()` on the data **x**. `MinMaxScaler()` makes the value of the data in the range 0-1. This `MinMaxScaler()` calculation can be seen in the following figure.
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/minmax.png" alt="MinMax">
</p>
This method is quite successful in making the accuracy of the model increase even if only slightly.

### Model Architecture
The architecture that we use in this Deep Learning model is illustrated in the image below.
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/mlp.jpg" alt="Architecture">
</p>
This architecture is based on reference [1] which only uses Dense Layer and Dropout Layer. At first, we tried to use the exact same architecture as the paper, but the results were not so good compared to using the architecture that we modified the hyperparameters.

This architecture is built using 4 Dense Layers with each unit of 125, 250, 250, and 41, then the relu and softmax activation functions for the last layer. In addition, the model architecture uses a Dropout Layer with a rate of 0.2.

<br>

## Training
The training model was carried out for 500 epochs. The optimizer used is `adam` with a learning rate of 0.001, the loss used is `sparse_categorical_crossentropy`, and also metrics accuracy.

The best last training result states that the highest accuracy of this model is 92-93% and the lowest loss is 12%. The graph of accuracy and loss can be seen in the image below.
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/acc.PNG" alt="Accuracy">
</p>
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/loss.PNG" alt="Loss">
</p>
The output of the model is the probability value of each class. We sort the class index with the highest probability class results and take the top 5 for use in the application. The following is an example of the model's prediction results.
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/pred.PNG" alt="Prediction">
</p>
We also compare the test accuracy of this model with the accuracy generated by conventional Machine Learning algorithms such as Decision Tree, SVM, and kNN. And as a result, the test accuracy of this model is much better than the three Machine Learning algorithms.
<p align="center">
  <img src="https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/assets/compare%20acc.PNG" alt="Comparison">
</p>

<br>

## References
[[1]](https://link.springer.com/article/10.1007/s00521-021-06872-0) Cepeda-Pacheco, J.C., Domingo, M.C. Deep learning and Internet of Things for tourist attraction recommendations in smart cities. Neural Comput & Applic 34, 7691–7709 (2022). https://doi.org/10.1007/s00521-021-06872-0 \
[[2]](https://ejournals.umn.ac.id/index.php/TI/article/view/339) Indriana, M., & Hwang, C.-S. (2014). Applying Neural Network Model to Hybrid Tourist Attraction Recommendations. Ultimatics : Jurnal Teknik Informatika, 6(2), 63-69. https://doi.org/https://doi.org/10.31937/ti.v6i2.339
