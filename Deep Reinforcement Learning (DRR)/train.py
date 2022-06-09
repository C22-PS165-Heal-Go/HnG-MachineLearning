import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.prioritized_replay_buffer import NaivePrioritizedReplayMemory, Transition
from utils.history_buffer import HistoryBuffer
from utils.general import export_plot

class DRRTrainer(object):
    def __init__(self,
                 config,
                 actor_function,
                 critic_function,
                 state_rep_function,
                 reward_function,
                 users,
                 items,
                 train_data,
                 test_data,
                 user_embeddings,
                 item_embeddings):
        
        ## importing reward function
        self.reward_function = reward_function
        ## importing training and testing data
        self.train_data = train_data
        self.test_data = test_data
        ## importing users and items
        self.users = users
        self.items = items
        ## importing user and item embeddings
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        ## declaring index identifier for dataset
        ## u for user, i for item, r for reward/rating
        self.u = 0
        self.i = 1
        self.r = 2
        
        ## dimensions
        ## self.item_embeddings already hold the weights array
        ## this should be 100
        self.item_features = self.item_embeddings.shape[1]
        self.user_features = self.user_embeddings.shape[1]
        
        ## number of user and items
        self.n_items = self.item_embeddings.shape[0]
        self.n_users = self.user_embeddings.shape[0]
        
        ## the shape of state space, action space
        ## this should be 300
        self.state_shape = 3 * self.item_features
        ## this should be 100
        self.action_shape = self.item_features
        
        self.critic_output_shape = 1
        self.config = config
        ## Data dimensions Extracted
        
        ## instantiate a drravestaterepresentation
        self.state_rep_net = state_rep_function(self.config.history_buffer_size,
                                                self.item_features,
                                                self.user_features)
        
        ## instantiate actor and target actor networks
        self.actor_net = actor_function(self.state_shape, self.action_shape)                                
        self.target_actor_net = actor_function(self.state_shape, self.action_shape)
        
        ## instantiate critic and target critics networks
        self.critic_net = critic_function(self.action_shape,
                                          self.state_shape,
                                          self.critic_output_shape)
        
        self.target_critic_net = critic_function(self.action_shape,
                                                 self.state_shape,
                                                 self.critic_output_shape)
        
        ## data flow for building the model
        flow_item = tf.convert_to_tensor(np.random.rand(5, 100), dtype='float32')
        flow_state = tf.convert_to_tensor(np.random.rand(1, 300), dtype='float32')
        flow_action = tf.convert_to_tensor(np.random.rand(1, 100), dtype='float32')
        
        ## flowing the data into the model to build the model
        self.state_rep_net(user_embeddings[0], flow_item)
        self.actor_net(flow_state)
        self.target_actor_net(flow_state)
        self.critic_net(flow_state, flow_action)
        self.target_critic_net(flow_state, flow_action)
        print("Actor-Critic model has successfully instantiated")
        
        self.state_rep_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.lr_state_rep)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.lr_actor)
        
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.lr_critic)
        
        print("DRR Instantiazed")
        
    def learn(self):
        # Initialize buffers
        print("NPRM and History Buffer Initialized")
        replay_buffer = NaivePrioritizedReplayMemory(self.config.replay_buffer_size,
                                                     prob_alpha=self.config.prob_alpha)

        history_buffer = HistoryBuffer(self.config.history_buffer_size)
        
        # Initialize trackers
        # initialize timesteps and epoch
        timesteps = 0
        epoch = 0
        ## this variable is for episode
        eps_slope = abs(self.config.eps_start - self.config.eps)/self.config.eps_steps
        eps = self.config.eps_start
        ## this variable is to hold the losses along the time
        actor_losses = []
        critic_losses = []
        ## this variable is to hold the episodic rewards
        epi_rewards = []
        epi_avg_rewards = []
        
        e_arr = []
        
        ## this variable holds the user index
        ## got from the dictionary
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)
        
        ## loop all the users based on indexes
        ## enumerates start with zero
        for idx, e in enumerate(user_idxs):
            ## starting the episodes
            
            ## the loops stop when timesteps-learning_start
            ## is bigger than the max timesteps
            if timesteps - self.config.learning_start > self.config.max_timesteps_train:
                break
            
            ## extracting positive user reviews
            ## e variable is an element right now
            user_reviews = self.train_data[self.train_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]
            
            ## check if the user ratings doesn't have enough positive review
            ## in this case history_buffer_size is 4
            ## get the shape object and 0 denote the row index
            if pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue
                
            candidate_items = tf.identity(tf.stop_gradient(self.item_embeddings))
            
            ## extracting user embedding tensors
            user_emb = self.user_embeddings[e]
            
            ## fill history buffer with positive item embeddings
            ## and remove item embeddings from candidate item sets
            ignored_items = []
            
            ## history_buffer_size has size of n items
            ## in this case 5
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[int(pos_user_reviews[i, self.i].numpy())]
                history_buffer.push(tf.identity(tf.stop_gradient(emb)))
                
            ## initialize rewards list
            rewards = []
            
            ## starting item index
            t = 0
            
            ## declaring the needed variable
            state = None
            action = None
            reward = None
            next_state = None
            print("Start User: {}".format(idx))

            while t < self.config.episode_length:
                ## observing the current state
                ## choose action according to actor network explorations
                
                ## inference calls start here
                # choose action according to actor network or exploration
                # config.eps = 0.1
                if eps > self.config.eps:
                    eps -= eps_slope
                else:
                    eps = self.config.eps
                
                ## state is the result of DRRAve model inference
                ## history_buffer.to_list get the list of previous items
                ## state representaton has the size (300, )
                state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()))
                
                if np.random.uniform(0, 1) < eps:
                    action = tf.convert_to_tensor(np.random.rand(1, self.action_shape), dtype='float32') 
                else:
                    action = self.actor_net(tf.stop_gradient(state), training=False)
                    
                ranking_scores = candidate_items @ tf.transpose(action)
                ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()
                ## calculating ranking scores accross items, discard ignored items
                
                if len(ignored_items) > 0:
                    rec_items = tf.stack(ignored_items).numpy()
                else:
                    rec_items = []
                
                ranking_scores[rec_items] = -float("inf")
                
                ## get the recommended items
                ## first get the maximum value index
                ## then get the items by index from candidate items
                rec_item_idx = tf.math.argmax(ranking_scores).numpy()
                rec_item_emb = candidate_items[rec_item_idx]
                
                ## add item to history buffer if positive reviews
                if rec_item_idx in user_reviews[:, self.i]:
                    user_rec_item_idx = np.where(user_reviews[:, self.i] == float(rec_item_idx))[0][0]
                    reward = user_reviews[user_rec_item_idx, self.r]
                else:
                    if self.config.zero_reward:
                        reward = tf.convert_to_tensor(0)
                    else:
                        reward = self.reward_function(float(e), float(rec_item_idx))

                rewards.append(reward.numpy())
                
                if reward > 0:
                    history_buffer.push(tf.identity(tf.stop_gradient(rec_item_emb)))
                    next_state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
                else:
                    next_state = tf.stop_gradient(state)
                
                ignored_items.append(rec_item_idx)
                print(rec_item_idx)
                replay_buffer.push(state, action, next_state, reward)
                
                ## Inference calling stops here
                ## Training start here
                if(timesteps > self.config.learning_start) and (len(replay_buffer) >= self.config.batch_size) and (timesteps % self.config.learning_freq == 0):
                    
                    #### TRAINING ####
                    critic_loss, actor_loss, critic_params_norm = self.training_step(timesteps,
                                                                                     replay_buffer,
                                                                                     True
                                                                                     )
                    ## storing the losses along the time
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    
                    ## outputting the result
                    if timesteps % self.config.log_freq == 0:
                        if len(rewards) > 0:
                            print(
                                f'Timestep {timesteps - self.config.learning_start} | '
                                f'Episode {epoch} | '
                                f'Mean Ep R '
                                f'{np.mean(rewards):.4f} | '
                                f'Max R {np.max(rewards):.4f} | '
                                f'Critic Params Norm {critic_params_norm:.4f} | '
                                f'Actor Loss {actor_loss:.4f} | '
                                f'Critic Loss {critic_loss:.4f} | ')
                            sys.stdout.flush()
            
                ## housekeeping
                t += 1
                timesteps += 1
            
                ## end of timesteps
            ## end of episodes
            if timesteps - self.config.learning_start > t:
                epoch += 1
                e_arr.append(epoch)
                epi_rewards.append(np.sum(rewards))
                epi_avg_rewards.append(np.mean(rewards))
        
        
        print("Training Finished")
        
        # self.actor_net.save_weights('trained/actor_weights/actor_150')
        # self.critic_net.save_weights('trained/critic_weights/critic_150')
        # self.state_rep_net.save_weights('trained/state_rep_weights/state_rep_150')
        # print("Model Saved")
        
        return actor_losses, critic_losses, epi_avg_rewards
    
    def training_step(self, t, replay_buffer, training):
        ## Get the created batches
        transitions, indicies, weights = replay_buffer.sample(self.config.batch_size, beta=self.config.beta)
        weights = tf.convert_to_tensor(weights, dtype='float32')
        
        ## create the tuple using Transition function     
        batch = Transition(*zip(*transitions))
        
        ## preparing the batch for each data
        ## the concat function will flatten the data
        ## the reshape will reshape the data so that it receive 64 rows
        next_state_batch = tf.reshape(tf.concat(batch.next_state, 0), [self.config.batch_size, -1])
        state_batch = tf.reshape(tf.concat(batch.state, 0), [self.config.batch_size, -1])
        action_batch = tf.reshape(tf.concat(batch.action, 0), [self.config.batch_size, -1])
        reward_batch = tf.reshape(tf.concat(batch.reward, 0), [self.config.batch_size, -1])
        
        ## updating the critic networks
        with tf.GradientTape(persistent=True) as tape:
            critic_loss, new_priorities = self.compute_prioritized_dqn_loss(tf.stop_gradient(state_batch),
                                                                            action_batch,
                                                                            reward_batch,
                                                                            next_state_batch,
                                                                            weights)
        ## apply the gradient
        grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
        
        replay_buffer.update_priorities(indicies, new_priorities)
        
        ## critic norm clipping
        critic_param_norm = [tf.clip_by_norm(layer.get_weights()[0] ,self.config.clip_val) for layer in self.critic_net.layers]
        critic_param_norm = tf.norm(critic_param_norm[0])
        
        ## step the optimizers
        self.critic_optimizer.apply_gradients(zip(grads, self.critic_net.trainable_variables))
                
        
        ## updating the actor networks
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(state_batch)
            actions_pred = self.actor_net(state_batch, training=True)
            actor_loss = -tf.reduce_mean(self.critic_net(tf.stop_gradient(state_batch), actions_pred, training=True))
            
        ## compute the gradient
        grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor_net.trainable_variables))
        
        ## compute another gradient
#         grads = tape.gradient(actor_loss, self.state_rep_net.trainable_variables)
#         self.state_rep_optimizer.apply_gradients(zip(grads, self.state_rep_net.trainable_variables))
        del tape
        
        ## updating the target networks
        self.soft_update(self.critic_net, self.target_critic_net, self.config.tau)
        self.soft_update(self.actor_net, self.target_actor_net, self.config.tau)
        
        return critic_loss.numpy(), actor_loss.numpy(), critic_param_norm
         
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: model which the weights will be copied from
            target_model: model which weights will be copied to
            tau (float): interpolation parameter
        """
        for t_layer, layer in zip(target_model.layers, local_model.layers):
            ## initiate list
            temp_w_arr = []
            for t_weights, weights in zip(t_layer.get_weights(), layer.get_weights()):
                ## fill the array list
                temp_w_arr.append(weights * tau + (1.0-tau) * t_weights)
            ## copy the weights
            t_layer.set_weights(temp_w_arr)
      
    def compute_prioritized_dqn_loss(self,
                                     state_batch,
                                     action_batch,
                                     reward_batch,
                                     next_state_batch,
                                     weights):
        '''
        :param state_batch: (tensor) shape = (batch_size x state_dims),
                The batched tensor of states collected during
                training (i.e. s)
        :param action_batch: (LongTensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
        :param reward_batch: (tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
        :param next_state_batch: (tensor) shape = (batch_size x state_dims),
                The batched tensor of next states collected during
                training (i.e. s')
        :param weights: (tensor) shape = (batch_size,)
                Weights for each batch item w.r.t. prioritized experience replay buffer
        :return: loss: (torch tensor) shape = (1),
                 new_priorities: (numpy array) shape = (batch_size,)
        '''
        ## create batches
        ## forward pass through target actor network
        next_action = self.target_actor_net(next_state_batch, training=False)
        q_target = self.target_critic_net(next_state_batch, next_action, training=False)
        ## y or target value that needs to be retreived
        y = reward_batch + self.config.gamma * q_target
        ## get q values from the current state
        q_vals = self.critic_net(state_batch, action_batch, training=True)
    
        ## calculate loss
        loss = tf.convert_to_tensor(y - q_vals)
        ## because loss is tensor shape
        ## we can extract the numpy value
        loss = tf.pow(loss, 2)
        weights_ten = tf.stop_gradient(weights)
        loss = tf.reshape(loss, (self.config.batch_size,)) * weights_ten
        ## stop the weights to be gradiented
        weights_ten = tf.stop_gradient(weights_ten)
        ## calculate new priorities
        new_priorities = tf.stop_gradient(loss).numpy() + 1e-5
        loss = tf.convert_to_tensor(tf.math.reduce_mean(loss))
        
        return loss, new_priorities
    
    def load_parameters(self):
        self.actor_net.load_weights('trained/actor_weights/actor_150')
        self.target_actor_net.load_weights('trained/actor_weights/actor_150')
        self.critic_net.load_weights('trained/critic_weights/critic_150')
        self.target_critic_net.load_weights('trained/critic_weights/critic_150')
        self.state_rep_net.load_weights('trained/state_rep_weights/state_rep_150')
    
    def offline_evaluate(self, T):
        ## loading the parameters
        self.load_parameters()
        
        history_buffer = HistoryBuffer(self.config.history_buffer_size)
        
        timesteps = 0
        epoch = 0
        rewards = []
        epi_precisions = []
        e_arr = []
        
        ## get users
        user_idxs = np.array(list(self.users.values()))
        np.random.shuffle(user_idxs)
        
        for step, e in enumerate(user_idxs):
            
            print("User Index: {}, step: {}".format(e, step))
            if len(e_arr) > self.config.max_epochs_offline:
                break
            
            ## extracting positive user reviews
            ## e variable is an element right now
            user_reviews = self.train_data[self.train_data[:, self.u] == e]
            pos_user_reviews = user_reviews[user_reviews[:, self.r] > 0]
            
            ## check if the user ratings doesn't have enough positive review
            ## in this case history_buffer_size is 4
            ## get the shape object and 0 denote the row index
            if pos_user_reviews.shape[0] < T or pos_user_reviews.shape[0] < self.config.history_buffer_size:
                continue
                
            candidate_items = tf.identity(tf.stop_gradient(self.item_embeddings))
            
            int_user_reviews_idx = user_reviews[:, self.i].numpy().astype(int)
            user_candidate_items = tf.identity(tf.stop_gradient(tf.gather(self.item_embeddings, indices=int_user_reviews_idx)))
            
            ## extracting user embedding tensors
            user_emb = self.user_embeddings[e]
            
            ## fill history buffer with positive item embeddings
            ## and remove item embeddings from candidate item sets
            ignored_items = []
            
            ## history_buffer_size has size of n items
            ## in this case 5
            for i in range(self.config.history_buffer_size):
                emb = candidate_items[int(pos_user_reviews[i, self.i].numpy())]
                history_buffer.push(tf.identity(tf.stop_gradient(emb)))
                
            ## initialize rewards list
            rewards = []
            
            ## starting item index
            t = 0
            
            ## declaring the needed variable
            state = None
            action = None
            reward = None
            next_state = None
            
            while t < self.config.episode_length:
                ## observing the current state
                ## choose action according to actor network explorations                
                ## state is the result of DRRAve model inference
                ## history_buffer.to_list get the list of previous items
                ## state representaton has the size (300, )
                state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
                if np.random.uniform(0, 1) < self.config.eps_eval:
                    action = tf.convert_to_tensor(np.random.rand(1, self.action_shape), dtype='float32') 
                else:
                    action = self.actor_net(tf.stop_gradient(state), training=False)
                    
                ranking_scores = candidate_items @ tf.transpose(action)
                ranking_scores = tf.reshape(ranking_scores, (ranking_scores.shape[0],)).numpy()
                ## calculating ranking scores accross items, discard ignored items
                
                if len(ignored_items) > 0:
                    rec_items = tf.stack(ignored_items).numpy().astype(int)
                else:
                    rec_items = []
                
                ranking_scores[rec_items[:, self.i] if len(ignored_items) > 0 else []] = -float("inf")
                
                ## get the recommended items
                ## first get the maximum value index
                ## then get the items by index from candidate items
                user_ranking_scores = tf.gather(ranking_scores, indices=user_reviews[:, self.i].numpy().astype(int))
                
                rec_item_idx = tf.math.argmax(user_ranking_scores).numpy()
                rec_item_emb = user_candidate_items[rec_item_idx]
                
                reward = user_reviews[rec_item_idx, self.r]

                rewards.append(reward.numpy())
                
                if reward > 0:
                    history_buffer.push(tf.identity(tf.stop_gradient(rec_item_emb)))
                    next_state = self.state_rep_net(user_emb, tf.stack(history_buffer.to_list()), training=False)
                else:
                    next_state = tf.stop_gradient(state)
                
                ignored_items.append(user_reviews[rec_item_idx])
                print(rec_item_idx)
                ## housekeeping
                t += 1
                timesteps += 1
                
                ## end of timesteps
            ## end of episodes
            
            rec_items = tf.stack(ignored_items)
            rel_pred = rec_items[rec_items[:, self.r] > 0]
            precision_T = len(rel_pred) / len(rec_items)
            
            epoch += 1
            e_arr.append(epoch)
            epi_precisions.append(precision_T)
            
            if timesteps % self.config.log_freq == 0:
                if len(rewards) > 0:
                    print(f'Episode {epoch} | '
                          f'Precision@{T} {precision_T} | '
                          f'Avg Precision@{T} {np.mean(epi_precisions):.4f} | '
                          )
                    sys.stdout.flush()
            
        print('Offline Evaluation Finished')
        print(f'Average Precision@{T}: {np.mean(epi_precisions):.4f} | ')
        plt.plot(e_arr, epi_precisions, label=f'Precision@{T}')
        plt.legend()
        plt.xlabel('Episode (t)')
        plt.ylabel('Precesion@T')
        plt.title('Precision@T (Offline Evaluation)')
        plt.minorticks_on()
        
        self.load_parameters()
            
        return np.mean(epi_precisions)