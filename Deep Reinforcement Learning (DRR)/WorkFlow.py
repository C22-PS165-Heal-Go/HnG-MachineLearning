import pickle
import numpy as np
import tensorflow as tf

from model import Actor, Critic, DRRAveStateRepresentation, PMF
from utils.history_buffer import HistoryBuffer

def build_state():
    items = np.random.rand(5, 100)
    users = np.random.rand(100,)
    state_rep_net(users, items)

def load_parameter():
    reward_function.load_weights('trained/pmf_weights/oversampling/pmf_oversampling_undersampling')
    actor_net.load_weights('trained/actor_weights/oversampling/actor_75')
    critic_net.load_weights('trained/critic_weights/oversampling/critic_75')
    state_rep_net.load_weights('trained/state_rep_weights/h5/state_rep_75.h5')

## ngebuka data
users = pickle.load(open('dataset_RL/user_id_to_num.pkl', 'rb'))
items = pickle.load(open('dataset_RL/item_lookup.pkl', 'rb'))

## deklarasi model
state_rep_net = DRRAveStateRepresentation(5, 100, 100)
actor_net = Actor(300, 100)
critic_net = Critic(100, 300, 1)
reward_function = PMF(len(users), len(items), 100)
## deklarasi build

state_rep_net.build_model()
build_state()
actor_net.build_model()
critic_net.build_model()
reward_function.build_model()
## load pretrained

try:
    ignored_items = np.load('saved/ignored_items.npy')
    ignored_items = list(ignored_items)
except FileNotFoundError:
    ignored_items = []
    print("File not found!")

## sifat user dan item
## untuk candidate item buat dihitung
user_embeddings = tf.convert_to_tensor(reward_function.user_embedding.get_weights()[0])
item_embeddings = tf.convert_to_tensor(reward_function.item_embedding.get_weights()[0])

## Declare Deque dari utils
history_buffer = HistoryBuffer(5)
## user buat sementara diacak dalam pemilihan embeddingnya

candidate_items = tf.identity(tf.stop_gradient(item_embeddings))
e = np.random.randint(0, 822)
user_emb = user_embeddings[0]

## history buffer itu gunanya buat nyimpen
## 5 item yang sebelumnya disukai user
for i in range(5):
    item_idx = np.random.randint(0, 142)
    emb = candidate_items[item_idx]
    history_buffer.push(tf.identity(tf.stop_gradient(emb)))

## list item yang tidak terpilih

## item terakhir yang dilike/ga dilike
## yang like positif yang ga like negatif
# item_yg_udh_dinilai = [[0, 0.8], [50, 0.8], [2, -0.8], [120, -0.8], [86, -0.8]]
# item_yg_udh_dinilai = [[0, 0.8], [1, 0.8], [2, -0.8], [3, 0.8], [86, -0.8]]
item_yg_udh_dinilai = [[0, 0.8], [5, 0.8], [25, 0.8], [45, 0.8], [75, 0.8]]
# item_yg_udh_dinilai = [[0, -0.8], [49, 0.8], [8, -0.8], [12, -0.8], [20, -0.8]]

## variabel buat nampung item hasil rekomendasi
item_for_recommendation = []
action_shape = 100

explore = 0.3

for item in item_yg_udh_dinilai:
    if item[1] > 0:
        emb = candidate_items[item[0]]
        history_buffer.push(tf.identity(tf.stop_gradient(emb)))
    ignored_items.append(item[0])
    
## declaring the needed variable
state = None
action = None
reward = None
next_state = None

if len(ignored_items) > 20:
    ignored_items = []
    
load_parameter()

for i in range(5):  
    state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
    if np.random.uniform(0, 1) < explore:
        action = tf.convert_to_tensor(np.random.rand(1, action_shape), dtype='float32')
    else:
        action = actor_net(tf.stop_gradient(state), training=False)
    
    ranking_scores = candidate_items @ tf.transpose(action)
    ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()

    rec_items = ignored_items if len(ignored_items) > 0 else []
    ranking_scores[rec_items] = -float("inf")
    
    rec_item_idx = tf.math.argmax(ranking_scores).numpy()
    rec_item_emb = candidate_items[rec_item_idx]
    
    reward = reward_function(float(e), float(rec_item_idx))
    if reward > 0:
        history_buffer.push(tf.identity(tf.stop_gradient(rec_item_emb)))
        next_state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
    else:
        next_state = tf.stop_gradient(state)
            
    ignored_items.append(rec_item_idx)
    item_for_recommendation.append(rec_item_idx)

for item in item_for_recommendation:
    print(items.get(item))

ignored_items = np.array(ignored_items)
np.save('saved/ignored_items.npy', ignored_items)