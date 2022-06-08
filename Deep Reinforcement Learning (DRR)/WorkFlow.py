import pickle
import numpy as np
import tensorflow as tf

from model import Actor, Critic, DRRAveStateRepresentation, PMF
from utils.history_buffer import HistoryBuffer

def load_parameter():
    reward_function.load_weights('trained/pmf_weights/pmf_150')
    actor_net.load_weights('trained/actor_weights/actor_150')
    critic_net.load_weights('trained/critic_weights/critic_150')
    state_rep_net.load_weights('trained/state_rep_weights/state_rep_150')

## ngebuka data
users = pickle.load(open('dataset_RL/user_id_to_num.pkl', 'rb'))
items = pickle.load(open('dataset_RL/item_lookup.pkl', 'rb'))
data = np.load('dataset_RL/data_RL_25000.npy')
data[:, 2] = 0.5 * (data[:, 2] - 3)



## deklarasi model
state_rep_net = DRRAveStateRepresentation(5, 100, 100)
actor_net = Actor(300, 100)
critic_net = Critic(100, 300, 1)
reward_function = PMF(len(users), len(items), 100)
## deklarasi build
state_rep_net.build_model()
actor_net.build_model()
critic_net.build_model()
reward_function.build_model()
## load pretrained

## load data
train_data = tf.convert_to_tensor(data[:int(0.8 * data.shape[0])], dtype='float32')

## sifat user dan item
## untuk candidate item buat dihitung
user_embeddings = tf.convert_to_tensor(reward_function.user_embedding.get_weights()[0])
item_embeddings = tf.convert_to_tensor(reward_function.item_embedding.get_weights()[0])

## Declare Deque dari utils
history_buffer = HistoryBuffer(5)
candidate_items = tf.identity(tf.stop_gradient(item_embeddings))

## random user
e = np.random.randint(0, len(users))
print("User Indeks ke-{}".format(e))

## user review ini bisa diacak" diganti aja angka 3-nya
## bisa pake variabel 'e' atau angka sendiri
user_reviews = train_data[train_data[:, 0] == 3]
## kalo ini variabel ngambil data yang sebelumnya disuka
pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

if pos_user_reviews.shape[0] < 5:
    while pos_user_reviews.shape[0] < 5:
        e = np.random.randint(0, len(users))
        user_reviews = train_data[train_data[:, 0] == e]
        pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

## copas item embedding
candidate_items = tf.identity(tf.stop_gradient(item_embeddings))

int_user_reviews_idx = user_reviews[:, 1].numpy().astype(int)
user_candidate_items = tf.identity(tf.stop_gradient(tf.gather(item_embeddings, indices=int_user_reviews_idx)))

user_emb = user_embeddings[e]

## item terakhir yang dilike/ga dilike
## yang like positif yang ga like negatif
## bisa eksplor" saja

## ========== DUMMY DATA ============= ##
# last_item = [[0, 0.8], [50, 0.8], [2, -0.8], [120, -0.8], [86, -0.8]]
# last_item = [[20, 0.8], [1, 0.8], [2, -0.8], [3, 0.8], [86, -0.8]]
# last_item = [[100, 0.8], [5, 0.8], [25, 0.8], [45, 0.8], [75, 0.8]]
last_item = [[30, -0.8], [49, 0.8], [8, -0.8], [12, -0.8], [20, -0.8]]
## ========== DUMMY DATA ============= ##

ignored_items = []
print("Terakhir Liked Items:")
for i in range(5):
    ## ini buat ngecek apakah lastnya itu dilike atau engga
    ## kalo dilike masukin buffer buat diproses
    if last_item[i][1] > 0:
        emb = candidate_items[last_item[i][0]]
        history_buffer.push(emb)
        print(items.get(last_item[i][0]))
        ignored_items.append(last_item[i][0])
        
if len(history_buffer) < 5:
    i = 0
    while len(history_buffer) < 5:
        emb = candidate_items[int(pos_user_reviews[i, 1].numpy())]
        history_buffer.push(tf.identity(tf.stop_gradient(emb)))
        i+=1

print("=====================")
print("Panjang Item yang diabaikan: {}".format(len(ignored_items)))

t = 0
explore = 0.2
state = None
action = None
reward = None
next_state = None
recommend_item = []
user_ignored_items = []

load_parameter()
while t < 10:
    state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
    if np.random.uniform(0, 1) < explore:
        action = tf.convert_to_tensor(0.1 * np.random.rand(1,100), dtype='float32')
    else:
        action = actor_net(state, training=False)
        
    ranking_scores = candidate_items @ tf.transpose(action)
    ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()
    
    if len(ignored_items) > 0:
        rec_items = tf.stack(ignored_items).numpy().astype(int)
        rec_items = np.array(rec_items)
    else:
        rec_items = []
    
    user_rec_items = tf.stack(user_ignored_items) if len(user_ignored_items) > 0 else []
    user_rec_items = np.array(user_rec_items)
    
    ranking_scores[rec_items if len(ignored_items) > 0 else []] = -float("inf")
    user_ranking_scores = tf.gather(ranking_scores, indices=user_reviews[:, 1].numpy().astype(int)).numpy()
    user_ranking_scores[user_rec_items if len(user_ignored_items) > 0 else []] = -float("inf")
    
    rec_item_idx = tf.math.argmax(user_ranking_scores).numpy()    
    rec_item_emb = user_candidate_items[rec_item_idx]
    
    recommend_item.append(rec_item_idx)
    reward = reward_function(float(e), float(rec_item_idx))
    ## reward_function ini buat prediksi aja usernya bakal suka
    ## atau engga dari item yang direkomendasiin
    ## reward_function juga mirip-mirip model prediksi
    
    if reward > 0:
        history_buffer.push(tf.identity(tf.stop_gradient(rec_item_emb)))
        next_state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()))
    else:
        next_state = tf.stop_gradient(state)
    
    user_ignored_items.append(rec_item_idx)
    t+=1
    
for item in recommend_item:
    print(items.get(item))