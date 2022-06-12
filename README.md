# Heal&Go App Machine Learning 💻

This repository contains the Machine Learning part of the Heal&Go App
<br>
<br>

## Machine Learning Team 🧍‍♂️🧍‍♀️
| [<img src="https://avatars.githubusercontent.com/u/79507977?v=4" width="75px;"/><br /><sub>Muhammad Ilham Malik</sub>](https://github.com/ilhamMalik51)<br /> | [<img src="https://avatars.githubusercontent.com/u/79434910?s=400&u=261666d21e81cac49b09bfc6e2f0869bb96de0de&v=4" width="75px;"/><br /><sub>Christina Prilla Rosaria Ardyanti</sub>](https://github.com/prillarosaria)<br /> | 
| :---: | :---: |
<br>

## Dataset 💾
The dataset that we used when building this application is a dataset that we created ourselves by distributing questionnaires to the public and scraping from Google Reviews.
- Deep Learning Model : [Dataset](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/dataset/questionnaire_dataset3.csv)
- Deep Reinforcement Learning Model : [Dataset](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/dataset_RL/Dataset_RL_Full.csv)
<br>

## Model 📈
We use the Deep Learning model to create a tourist destination recommendation system using datasets derived from questionnaires. Meanwhile, for the Deep Reinforcement Learning model using a dataset from Google Review.

- Deep Learning : [Code](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/deep%20learning/dl_model.ipynb), [Model](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/tree/main/deep%20learning/model)\
📝 Note : **The highest train accuracy of this model has reached 93% and the lowest loss has reached 12%**

- Deep Reinforcement : [Code PMF](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/PMF_TensorFlow.ipynb), [Code DRR-Ave](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/DRR_Ave_Models_Tensorflow.ipynb), [Code Actor-Critic](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/Actor_Network_Tensorflow.ipynb), [Model](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/model.py), [Demo](https://github.com/C22-PS165-Heal-Go/HnG-MachineLearning/blob/main/Deep%20Reinforcement%20Learning%20(DRR)/WorkFlow.py)\
📝 Note : **The DRR offline evaluation**
  - Precision@5: 0.6962
  - Precision@10: 0.7225
  - Precision@15: 0.7835
  - Precision@20: 0.8622
<br>

## Reference 📚
Some of the papers or research that we use in building Deep Learning models and Deep Reinforcement Learning models.
| Deep Learning| Deep Reinforcement Learning |
| :---: | :---: |
| [Deep learning and Internet of Things for tourist attraction recommendations in smart cities](https://link.springer.com/article/10.1007/s00521-021-06872-0) | [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/pdf/1810.12027.pdf) |
| [Applying Neural Network Model to Hybrid Tourist Attraction Recommendations](https://ejournals.umn.ac.id/index.php/TI/article/view/339) | [End-to-end Deep Reinforcement Learning Based Coreference Resolution](https://aclanthology.org/P19-1064.pdf) |

