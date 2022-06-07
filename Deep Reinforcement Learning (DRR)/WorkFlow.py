import pickle
import numpy as np
import tensorflow as tf

from model import Actor, Critic, DRRAveStateRepresentation, PMF
from utils.history_buffer import HistoryBuffer

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
reward_function.load_weights('trained/pmf_weights/pmf_150')
actor_net.load_weights('trained/actor_weights/actor_150')
critic_net.load_weights('trained/critic_weights/critic_150')
state_rep_net.load_weights('trained/state_rep_weights/state_rep_150')

## sifat user dan item
## untuk candidate item buat dihitung
user_embeddings = tf.convert_to_tensor(reward_function.user_embedding.get_weights()[0])
candidate_items = tf.convert_to_tensor(reward_function.item_embedding.get_weights()[0])

## Declare Deque dari utils
history_buffer = HistoryBuffer(5)
## user buat sementara diacak dalam pemilihan embeddingnya
user_idx = np.random.randint(0, 549)
## ambil embedding user/sifat user
user_emb = user_embeddings[user_idx]

## pake data sampel buat ngambil
## 5 item terakhir yang disukai user
user_reviews = data[data[:, 0] == 3]
pos_user_reviews = user_reviews[user_reviews[:, 2] > 0]

## history buffer itu gunanya buat nyimpen
## 5 item yang sebelumnya disukai user
for i in range(5):
    emb = candidate_items[int(pos_user_reviews[i, 1])]
    history_buffer.push(tf.identity(tf.stop_gradient(emb)))

## list item yang tidak terpilih
ignored_items = []

## program mulai
t = 0
episode_length = 5

## item terakhir yang dilike/ga dilike
## yang like positif yang ga like negatif
item_yg_udh_dinilai = [[0, 0.8], [1, 0.8], [2, -0.8], [3, -0.8], [4, 0.8]]

## variabel buat nampung item hasil rekomendasi
item_for_recommendation = []

for item in item_yg_udh_dinilai:

    if item[1] > 0:
        ## ambil embeddingnya terus push ke buffer history
        emb = candidate_items[item[0]]
        history_buffer.push(tf.identity(tf.stop_gradient(emb)))

        ## statenya hitung
        state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()))

        ## actionnya hitung
        action = actor_net(tf.stop_gradient(state))

        # Perhitungan skor ranking item terbaik dan di-flatten
        ranking_scores = candidate_items @ tf.transpose(action)
        ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()

        ## ini jaga" biar ga direkomen dua kali
        if len(ignored_items) > 0:
            rec_items = tf.stack(ignored_items).numpy()
        else:
            rec_items = []

        ## nanti di list rankingnya dinegatifin biar di akhir posisinya
        ranking_scores[rec_items] = -float("inf")

        # ambil rekomen itemnya
        rec_item_idx = tf.argmax(ranking_scores).numpy()
        rec_item_emb = candidate_items[rec_item_idx]

        item_for_recommendation.append(rec_item_idx)

        # Remove new item from future recommendations
        ignored_items.append(tf.convert_to_tensor(rec_item_idx))

    else:
        ## statenya dihitung
        ## karena itemnya gadilike jadi keadaan state masih sama
        state = state_rep_net(user_emb, tf.stack(history_buffer.to_list()))

        ## actionnya hitung
        action = actor_net(tf.stop_gradient(state))

        # Perhitungan skor ranking item terbaik dan di-flatten
        ranking_scores = candidate_items @ tf.transpose(action)
        ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()

        ## ini jaga" biar ga direkomen dua kali
        if len(ignored_items) > 0:
            rec_items = tf.stack(ignored_items).numpy()
        else:
            rec_items = []

        ## nanti di list rankingnya dinegatifin biar di akhir posisinya
        ranking_scores[rec_items] = -float("inf")

        # ambil rekomen itemnya
        rec_item_idx = tf.argmax(ranking_scores).numpy()
        rec_item_emb = candidate_items[rec_item_idx]

        item_for_recommendation.append(rec_item_idx)

        # Remove new item from future recommendations
        ignored_items.append(tf.convert_to_tensor(rec_item_idx))

for indeks in item_for_recommendation:
    print(items.get(indeks))