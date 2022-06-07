# Heal&Go App Machine Learning 💻

This repository contains the Machine Learning part of the Heal&Go App
<br>
<br>

## Dataset 💾
The dataset that we used when building this application is a dataset that we created ourselves by distributing questionnaires to the public and scraping from Google Reviews.
- Deep Learning Model : [Dataset](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/dataset/questionnaire_dataset3.csv)
- Deep Reinforcement Learning Model : [Dataset](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/dataset/Dataset_Dest_Rating.xlsx)
<br>

## Model 📈
We use the Deep Learning model to create a tourist destination recommendation system using datasets derived from questionnaires. Meanwhile, for the Deep Reinforcement Learning model using a dataset from Google Review.

- Deep Learning : [Code](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/src/dl_model.ipynb), [Model](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/src/model)\
📝 Note : **The train accuracy of this model has reached 92%**

- Deep Reinforcement : [Code PMF](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/PMF_TensorFlow.ipynb), [Code DRR-Ave](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/DRR_Ave_Models_Tensorflow.ipynb), [Code Actor-Critic](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/Actor_Network_Tensorflow.ipynb), [Model](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/model.py)
- [Demo](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/WorkFlow.py)

📝 Note : **The DRR offline evaluation**
- Precision@5: 0.7287
- Precision@10: 0.7368
- Precision@15: 0.7735
- Precision@20: 0.8122

**Re-implementing in TensorFlow Framework for deploying, New Training Algorithm on-progress**
