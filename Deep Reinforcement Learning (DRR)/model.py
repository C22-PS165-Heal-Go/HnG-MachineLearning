import tensorflow as tf
import numpy as np
from numpy.random import RandomState

class Parameter(tf.keras.layers.Layer):
    # Create PyTorch like paramater layer
    def __init__(self, units, **kwargs):
        super(Parameter, self).__init__(**kwargs)
        self.w = self.add_weight(
            shape=(units), initializer=None, trainable=True
        )
                        
    def call(self, inputs):
        # make sure the shapes are compatible
        return inputs, self.w

class DRRAveStateRepresentation(tf.keras.Model):
    
    # Initialize method when model is instantiated
    def __init__(self, n_items=5, item_features=100, user_features=100):
        # Inherit from the upper
        super(DRRAveStateRepresentation, self).__init__()
        self.n_items = n_items
        self.random_state = RandomState(1)
        self.item_features = item_features
        self.user_features = user_features

        self.attention_weights = Parameter(tf.convert_to_tensor(0.1 * self.random_state.rand(self.n_items)))

        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, user, items):
        """
        DRR-AVE State Representation
        :param items: (torch tensor) shape = (n_items x item_features),
                Matrix of items in history buffer

        :param user: (torch tensor) shape = (1 x user_features),
                User embedding

        :return: output: (torch tensor) shape = (3 * item_features)
        """
        right = tf.transpose(items) @ self.attention_weights
        middle = user * right
        output = self.concat([user, middle, right], axis=0)

        return self.flatten(output)

class Actor(tf.keras.Model):
    def __init__(self, in_features=100, out_features=18):
        super(Actor, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(name='input_layer', input_shape=(in_features,))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(in_features, activation='relu'),
            tf.keras.layers.Dense(in_features, activation='relu'),
            tf.keras.layers.Dense(out_features, activation='tanh')
        ])
        
    def call(self, state):
        output = self.inputs(state)
        return self.fc(output)

class Critic(tf.keras.Model):
    def __init__(self, action_size=20, in_features=128, out_features=18):
        super(Critic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.action_size = action_size

        self.combo_features = in_features + action_size

        self.inputs = tf.keras.layers.InputLayer(input_shape=(self.in_features, 3*in_features))
        self.fc1 = tf.keras.layers.Dense(self.in_features, activation = 'relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(self.combo_features, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(self.combo_features, activation = 'relu')
        self.out = tf.keras.layers.Dense(self.out_features, activation = 'linear')

class PMF(tf.keras.Model):
    def __init__(self, n_users, n_items, n_factors=20):
        super(PMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.random_state = RandomState(1)

        self.user_embedding = tf.keras.layers.Embedding(n_users, n_factors)
