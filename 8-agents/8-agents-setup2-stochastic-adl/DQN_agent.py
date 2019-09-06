import tensorflow as tf
import collections
import random
import numpy as np
import math
import copy

np.random.seed(7)
random.seed(7)

class DQN_Agent:

    def __init__(self, name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl, grid_price, total_iterations, current_iteration, lam):
        self.name = name
        self.state_size = state_size
        self.max_battery = max_battery
        self.max_energy_generated = max_energy_generated
        self.max_received = max_received
        self.grid_price = grid_price
        self.action_size_pricing = (max_battery + max_energy_generated) * 6 + max_received + 1  # 6 for grid price to (grid price - 5) and 1 for the zeroth state.
        self.action_size_adl = 8
        self.total_iterations = total_iterations
        self.current_iteration = current_iteration
        self.memory = collections.deque(maxlen = 10000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0
        self.regularizer_loss = 50  # Loss for penalising the impossible actions
        self.regularizer_factor = 0.07
        self.pricing_model = self._build_model_pricing()
        self.adl_model = self._build_model_adl()
        self.non_adl = [3,4,5,6]
        self.prob_non_adl = [[0.4,0.3,0.2,0.1],[0.1,0.4,0.3,0.2],[0.1,0.3,0.4,0.2],[0.2,0.3,0.1,0.4]]
        self.adl_value = [[[1, 2], [2, 2], [1, 2]], [[1, 3], [2, 3], [1, 3]], [[2, 4], [2, 4], [1, 4]]]
        self.lam = lam
        self.adl_state = 7
        self.adl_state_second = 0 #  it can go from 0 to 26, this will denote which item to select from each set example 26 means 222(we have used ternary encoding) this means second element from each set is the adl demand  #Changed
    
    def convert_decimal_ternary(self,adl_state_sec): # Changed
        ans = []
        for i in range(3):
            ans.append(adl_state_sec % 3)
            adl_state_sec = adl_state_sec // 3
        return(ans)     # 0th index give the value for the first set 1st gives the value for the second set and ..

    #done
    def get_renewable(self,time):
        energy = np.random.poisson(lam=self.lam[time-1],size=1)
        energy = min([10, energy]) # clipping the value so that it can't exceede 8
        energy = int(math.floor(energy))
        return energy
    
    #done
    def get_non_adl_demand(self,time):
        demand = np.random.choice(self.non_adl,p=self.prob_non_adl[time-1])
        return int(demand)

    #done
    def custom_loss(self, y_true, y_pred):
        loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=1)) # need to put axis
        return loss

    #done
    def _build_model_pricing(self):

        # nd, d, adl_action , t , gp
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32,input_shape=(5,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(self.action_size_pricing))
        optim = tf.keras.optimizers.Adam(lr = 0.00007)
        model.compile(loss = self.custom_loss, optimizer = optim)
        return model

    #done
    def _build_model_adl(self):

    	# nd, d, adl_state, t, gp
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(16,input_shape=(6,), activation = tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(16, activation = tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(self.action_size_adl))
        optim = tf.keras.optimizers.Adam(lr = 0.00007)
        model.compile(loss = self.custom_loss, optimizer = optim)
        return model

    #done
    def summary(self):
        self.pricing_model.summary()
        print('-------------------')
        self.adl_model.summary()
        print('-------------------')

    #done
    def pricing_convert_constraint_values_to_allowed_action_indices(self, state):

        # State contains nd, d, adl, t, gp
        nd = state[0]
        d = state[1]
        adl = self.adl_convert_allowed_indices_to_values(state[2])
        lower_bound = max(-1*self.max_received, nd - self.max_battery - adl)
        upper_bound = nd + d - adl

        if (lower_bound <= 0):
            lower_bound_index = lower_bound + self.max_received
        else:
            lower_bound_index = self.max_received + (lower_bound - 1) * 6 + 1  # In order to account for the zero state

        if (upper_bound <= 0):
            upper_bound_index = upper_bound + self.max_received
        else:
            upper_bound_index = self.max_received + upper_bound * 6

        return lower_bound_index, upper_bound_index

    #done
    def pricing_convert_allowed_indices_to_values(self, action):
        # return price and ut 
        if (action <= self.max_received):
            return([0, action -self.max_received]);
        else :
            action = action - self.max_received - 1 
            return([action % 6 + self.grid_price - 5 , action // 6 + 1])

     

    def pricing_action(self, state):

        lower_bound_index, upper_bound_index = self.pricing_convert_constraint_values_to_allowed_action_indices(state)
        check = random.uniform(0, 1)
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) != 7:
                possible_actions.append(i)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            output = self.pricing_model.predict(state)
            index = 0
            max_ = - 50000000
            for i in possible_actions:
                if max_ < output[0][i]:
                    max_ = output[0][i]
                    index = i

            return index

    def price_max_Q(self,state,li,ui):
        lower_bound_index, upper_bound_index = li,ui
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) != 7:
                possible_actions.append(i)
        output = self.pricing_model.predict(state)
        max_ = -50000000
        for i in possible_actions:
            if max_ < output[0][i]:
                max_ = output[0][i]
        return max_


    # done
    def adl_give_possible_actions(self, state):

        possible_actions = []
        possible_actions.append(0)
        for i in range(3):
            if (state & 2**i):
                temp = copy.deepcopy(possible_actions)
                for j in range(len(temp)):
                    temp[j] += 2**i

                possible_actions.extend(temp)
        return possible_actions

    # done
    def adl_convert_allowed_indices_to_values(self, action):  #  # Changed
        adl = 0
        temp = action
        second_state = self.convert_decimal_ternary(self.adl_state_second)
        for j in range(3):
            if (temp % 2 == 1):
                adl += self.adl_value[j][second_state[j]][0]
            temp = temp//2
        return adl
    
    #done
    def adl_action(self, state):

        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        check = random.uniform(0, 1)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            index = 0
            max_ = - 50000000
            output = self.adl_model.predict(np.asarray([state]))
            for i in possible_actions:
                if max_ < output[0][i]:
                    max_ = output[0][i]
                    index = i

            return index  

    # Give the action with max Q value 
    #done
    def adl_max_Q(self, state):

        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        index = 0
        max_ = - 50000000
        output = self.adl_model.predict(np.asarray([state]))
        for i in possible_actions:
            if max_ < output[0][i]:
                max_ = output[0][i]
                index = i

        return max_ 

    #done
    def update_adl(self,adl_action, time):  # Changed
        self.adl_state = self.adl_state & (~adl_action)
        second_state = self.convert_decimal_ternary(self.adl_state_second)
        penalty = 0
        if time==2 and (self.adl_state & 1):
            self.adl_state = self.adl_state & (~1)
            penalty = self.adl_value[0][second_state[0]][0]
        
        elif time==3 and (self.adl_state & 2):
            self.adl_state = self.adl_state & (~2)
            penalty = self.adl_value[1][second_state[1]][0]

        elif time==4 and (self.adl_state & 4):
            penalty = self.adl_value[2][second_state[2]][0]
            #self.adl_state = 7
        
        if time==4:
            values = [0, 1, 2, 3, 4, 5, 6, 7]
            prob = [0.05, 0.05, 0.05, 0.15, 0.05, 0.15, 0.15, 0.35]
            self.adl_state = np.random.choice(values, p=prob)     # randomise this  
            self.adl_state_second = random.randint(0, 26)
        new_adl_state = copy.deepcopy(self.adl_state)
        new_secondary_adl_state = copy.deepcopy(self.adl_state_second)
        return penalty, new_adl_state, new_secondary_adl_state

    def remember(self, state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price):
        self.memory.append((state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price))

    def load_model(self,path_pricing, path_adl):
        self.pricing_model.load_weights(path_pricing)
        self.adl_model.load_weights(path_adl)

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        states_batch_pricing, targets_batch_pricing, regularizer_batch = [], [], []
        states_batch_adl, targets_batch_adl = [], []
        y_true = []

        for state_adl, state_price, action_adl,action_price, reward, next_state_adl, next_state_price in minibatch:
            
            dl_state, dl_next_state = np.array([state_price]),np.array([next_state_price])
            lower_bound_index_ns, upper_bound_index_ns = self.pricing_convert_constraint_values_to_allowed_action_indices(next_state_price)
            target_price = reward / 180.0 + self.gamma * self.price_max_Q(next_state_price,lower_bound_index_ns,upper_bound_index_ns)
            target_array_price = self.pricing_model.predict(dl_state)
            target_array_price[0][action_price] = target_price

            dl_adl_state, dl_adl_next_state = np.array([state_adl]), np.array([next_state_adl])
            target_adl = reward / 180.0 + self.gamma * self.adl_max_Q(next_state_adl)
            target_array_adl = self.adl_model.predict(dl_adl_state)
            target_array_adl[0][action_adl] = target_adl
            
            states_batch_pricing.append(state_price)
            targets_batch_pricing.append(target_array_price[0])
            states_batch_adl.append(state_adl)
            targets_batch_adl.append(target_array_adl[0])
    

        history_pricing = self.pricing_model.fit(np.array(states_batch_pricing), np.array(targets_batch_pricing), epochs=1, verbose=0)
        history_adl = self.adl_model.fit(np.array(states_batch_adl), np.array(targets_batch_adl), epochs=1, verbose=0)
        loss_pricing = history_pricing.history['loss'][0]
        loss_adl = history_adl.history['loss'][0]
        self.epsilon = max(self.epsilon_min, (0.8 - self.current_iteration/self.total_iterations))
        
        self.current_iteration += 4

        return loss_adl,loss_pricing

    def save_model(self):
        self.pricing_model.save_weights('./saved/'  + self.name + '_pricing_model_adl_pricing' + '.h5')
        self.adl_model.save_weights('./saved/'  + self.name + '_adl_model_adl_pricing' + '.h5')

class DQN_Agent_Constant_Price:

    def __init__(self, name, state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl, grid_price, total_iterations, current_iteration, lam):
        self.name = name
        self.state_size = state_size
        self.max_battery = max_battery
        self.max_energy_generated = max_energy_generated
        self.max_received = max_received
        self.grid_price = grid_price
        self.action_size_pricing = (max_battery + max_energy_generated) * 6 + max_received + 1  # 6 for grid price to (grid price - 5) and 1 for the zeroth state.
        self.action_size_adl = 8
        self.total_iterations = total_iterations
        self.current_iteration = current_iteration
        self.memory = collections.deque(maxlen = 10000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0
        self.regularizer_loss = 50  # Loss for penalising the impossible actions
        self.regularizer_factor = 0.07
        self.pricing_model = self._build_model_pricing()
        self.adl_model = self._build_model_adl()
        self.non_adl = [3,4,5,6]
        self.prob_non_adl = [[0.4,0.3,0.2,0.1],[0.1,0.4,0.3,0.2],[0.1,0.3,0.4,0.2],[0.2,0.3,0.1,0.4]]
        self.adl_value = [[[1, 2], [2, 2], [1, 2]], [[1, 3], [2, 3], [1, 3]], [[2, 4], [2, 4], [1, 4]]]
        self.lam = lam
        self.adl_state = 7
        self.adl_state_second = 0 #  it can go from 0 to 26, this will denote which item to select from each set example 26 means 222(we have used ternary encoding) this means second element from each set is the adl demand  #Changed
    
    def convert_decimal_ternary(self,adl_state_sec): # Changed
        ans = []
        for i in range(3):
            ans.append(adl_state_sec % 3)
            adl_state_sec = adl_state_sec // 3
        return(ans)     # 0th index give the value for the first set 1st gives the value for the second set and ..

    #done
    def get_renewable(self,time):
        energy = np.random.poisson(lam=self.lam[time-1],size=1)
        energy = min([10, energy]) # clipping the value so that it can't exceede 8
        energy = int(math.floor(energy))
        return energy
    
    #done
    def get_non_adl_demand(self,time):
        demand = np.random.choice(self.non_adl,p=self.prob_non_adl[time-1])
        return int(demand)

    #done
    def custom_loss(self, y_true, y_pred):
        loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=1)) # need to put axis
        return loss

    #done
    def _build_model_pricing(self):

        # nd, d, adl_action , t , gp
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32,input_shape=(5,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(self.action_size_pricing))
        optim = tf.keras.optimizers.Adam(lr = 0.00007)
        model.compile(loss = self.custom_loss, optimizer = optim)
        return model

    #done
    def _build_model_adl(self):

        # nd, d, adl_state, t, gp
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(16,input_shape=(6,), activation = tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(16, activation = tf.keras.activations.relu, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
        model.add(tf.keras.layers.Dense(self.action_size_adl))
        optim = tf.keras.optimizers.Adam(lr = 0.00007)
        model.compile(loss = self.custom_loss, optimizer = optim)
        return model

    #done
    def summary(self):
        self.pricing_model.summary()
        print('-------------------')
        self.adl_model.summary()
        print('-------------------')

    #done
    def pricing_convert_constraint_values_to_allowed_action_indices(self, state):

        # State contains nd, d, adl, t, gp
        nd = state[0]
        d = state[1]
        adl = self.adl_convert_allowed_indices_to_values(state[2])
        lower_bound = max(-1*self.max_received, nd - self.max_battery - adl)
        upper_bound = nd + d - adl

        if (lower_bound <= 0):
            lower_bound_index = lower_bound + self.max_received
        else:
            lower_bound_index = self.max_received + (lower_bound - 1) * 6 + 1  # In order to account for the zero state

        if (upper_bound <= 0):
            upper_bound_index = upper_bound + self.max_received
        else:
            upper_bound_index = self.max_received + upper_bound * 6

        return lower_bound_index, upper_bound_index

    #done
    def pricing_convert_allowed_indices_to_values(self, action):
        # return price and ut 
        if (action <= self.max_received):
            return([0, action -self.max_received]);
        else :
            action = action - self.max_received - 1 
            return([action % 6 + self.grid_price - 5 , action // 6 + 1])

     

    def pricing_action(self, state):

        lower_bound_index, upper_bound_index = self.pricing_convert_constraint_values_to_allowed_action_indices(state)
        check = random.uniform(0, 1)
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) == 0:
                possible_actions.append(i)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            output = self.pricing_model.predict(state)
            index = 0
            max_ = - 50000000
            for i in possible_actions:
                if max_ < output[0][i]:
                    max_ = output[0][i]
                    index = i

            return index

    def price_max_Q(self,state,li,ui):
        lower_bound_index, upper_bound_index = li,ui
        state = np.array([state])
        possible_actions = []
        for i in range(lower_bound_index, upper_bound_index + 1):
            if i <= self.max_received:
                possible_actions.append(i)
            elif ((i - self.max_received) % 6) == 0:
                possible_actions.append(i)
        output = self.pricing_model.predict(state)
        max_ = -50000000
        for i in possible_actions:
            if max_ < output[0][i]:
                max_ = output[0][i]
        return max_


    # done
    def adl_give_possible_actions(self, state):

        possible_actions = []
        possible_actions.append(0)
        for i in range(3):
            if (state & 2**i):
                temp = copy.deepcopy(possible_actions)
                for j in range(len(temp)):
                    temp[j] += 2**i

                possible_actions.extend(temp)
        return possible_actions

    # done
    def adl_convert_allowed_indices_to_values(self, action):  #  # Changed
        adl = 0
        temp = action
        second_state = self.convert_decimal_ternary(self.adl_state_second)
        for j in range(3):
            if (temp % 2 == 1):
                adl += self.adl_value[j][second_state[j]][0]
            temp = temp//2
        return adl
    
    #done
    def adl_action(self, state):

        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        check = random.uniform(0, 1)
        if (check < self.epsilon):
            return np.random.choice(possible_actions)
        else:
            index = 0
            max_ = - 50000000
            output = self.adl_model.predict(np.asarray([state]))
            for i in possible_actions:
                if max_ < output[0][i]:
                    max_ = output[0][i]
                    index = i

            return index  

    # Give the action with max Q value 
    #done
    def adl_max_Q(self, state):

        possible_actions = sorted(self.adl_give_possible_actions(state[2]))
        index = 0
        max_ = - 50000000
        output = self.adl_model.predict(np.asarray([state]))
        for i in possible_actions:
            if max_ < output[0][i]:
                max_ = output[0][i]
                index = i

        return max_ 

    #done
    def update_adl(self,adl_action, time):  # Changed
        self.adl_state = self.adl_state & (~adl_action)
        second_state = self.convert_decimal_ternary(self.adl_state_second)
        penalty = 0
        if time==2 and (self.adl_state & 1):
            self.adl_state = self.adl_state & (~1)
            penalty = self.adl_value[0][second_state[0]][0]
        
        elif time==3 and (self.adl_state & 2):
            self.adl_state = self.adl_state & (~2)
            penalty = self.adl_value[1][second_state[1]][0]

        elif time==4 and (self.adl_state & 4):
            penalty = self.adl_value[2][second_state[2]][0]
            #self.adl_state = 7
        
        if time==4:
            values = [0, 1, 2, 3, 4, 5, 6, 7]
            prob = [0.05, 0.05, 0.05, 0.15, 0.05, 0.15, 0.15, 0.35]
            self.adl_state = np.random.choice(values, p=prob)     # randomise this  
            self.adl_state_second = random.randint(0, 26)
        new_adl_state = copy.deepcopy(self.adl_state)
        new_secondary_adl_state = copy.deepcopy(self.adl_state_second)
        return penalty, new_adl_state, new_secondary_adl_state

    def remember(self, state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price):
        self.memory.append((state_adl, state_price, action_adl, action_price, reward, next_state_adl, next_state_price))

    def load_model(self,path_pricing, path_adl):
        self.pricing_model.load_weights(path_pricing)
        self.adl_model.load_weights(path_adl)

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        states_batch_pricing, targets_batch_pricing, regularizer_batch = [], [], []
        states_batch_adl, targets_batch_adl = [], []
        y_true = []

        for state_adl, state_price, action_adl,action_price, reward, next_state_adl, next_state_price in minibatch:
            
            dl_state, dl_next_state = np.array([state_price]),np.array([next_state_price])
            lower_bound_index_ns, upper_bound_index_ns = self.pricing_convert_constraint_values_to_allowed_action_indices(next_state_price)
            target_price = reward / 180.0 + self.gamma * self.price_max_Q(next_state_price,lower_bound_index_ns,upper_bound_index_ns)
            target_array_price = self.pricing_model.predict(dl_state)
            target_array_price[0][action_price] = target_price

            dl_adl_state, dl_adl_next_state = np.array([state_adl]), np.array([next_state_adl])
            target_adl = reward / 180.0 + self.gamma * self.adl_max_Q(next_state_adl)
            target_array_adl = self.adl_model.predict(dl_adl_state)
            target_array_adl[0][action_adl] = target_adl

            
            
            states_batch_pricing.append(state_price)
            targets_batch_pricing.append(target_array_price[0])
            states_batch_adl.append(state_adl)
            targets_batch_adl.append(target_array_adl[0])
    

        history_pricing = self.pricing_model.fit(np.array(states_batch_pricing), np.array(targets_batch_pricing), epochs=1, verbose=0)
        history_adl = self.adl_model.fit(np.array(states_batch_adl), np.array(targets_batch_adl), epochs=1, verbose=0)
        loss_pricing = history_pricing.history['loss'][0]
        loss_adl = history_adl.history['loss'][0]
        self.epsilon = max(self.epsilon_min, (0.8 - self.current_iteration/self.total_iterations))
        
        self.current_iteration += 4

        return loss_adl,loss_pricing

    def save_model(self):
        self.pricing_model.save_weights('./saved/'  + self.name + '_pricing_model_adl_pricing' + '.h5')
        self.adl_model.save_weights('./saved/'  + self.name + '_adl_model_adl_pricing' + '.h5')
