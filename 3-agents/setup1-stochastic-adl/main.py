import tensorflow as tf
import collections
import random
import numpy as np
from DQN_agent import *
import logging


# seeds
np.random.seed(7)
random.seed(7)

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def mean(var):
    return sum(var)/len(var)

formatter = logging.Formatter('%(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logger_updates = setup_logger('logger_updates','./logs/logger_updates.txt')
logger_loss1 = setup_logger('logger_loss','./logs/logger_loss1.txt')
logger_loss2 = setup_logger('logger_loss','./logs/logger_loss2.txt')
adl_logger = setup_logger('adl_logger','./logs/adl_logger.txt')
reward_logger = setup_logger('reward_logger','./logs/reward_logger.txt')
price_logger = setup_logger('price_logger','./logs/price_logger.txt')
renewable_logger = setup_logger('renewable_logger','./logs/renewable_logger.txt')
battery_logger = setup_logger('battery_logger','./logs/battery_logger.txt')
nd_logger = setup_logger('nd_logger','./logs/nd_logger.txt')
transmission_logger = setup_logger('transmission_logger','./logs/transmission_logger.txt')

def transaction(actions):

    # we will divide the agents as buyers and sellers
    rewards = []
    for i in range(len(actions)):
        rewards.append(0)
    buyers = [] 
    sellers = []
    prices_dict = {}
    # actions : index, price , ut
    for i in range(len(actions)):
        if actions[i][2] < 0:
            buyers.append(actions[i])
        else :
            sellers.append(actions[i])
            if prices_dict.get(actions[i][1]) == None:
                prices_dict[actions[i][1]] = 1
            else:
                prices_dict[actions[i][1]] +=1

    sellers = sorted(sellers ,key=lambda x: x[1])  # sort the sellers according to the prices they have quoted




    total_demand = 0
    total_supply = 0
    for i in range(len(sellers)):
        total_supply += sellers[i][2]

    for i in range(len(buyers)):
        total_demand += abs(buyers[i][2])

    sellers_earning = 0
    buyers_spending = 0



    if (total_demand >= total_supply):
        # you have to meet the extra demand using the external supply
        temp = total_demand
        for i in range(len(sellers)):
            buyers_spending += sellers[i][1] * sellers[i][2]
            temp -= sellers[i][2]

        buyers_spending += temp * grid_price
        # now we have both sellers_earning and buyers_spending
        for i in range(len(sellers)):
            rewards[sellers[i][0]] += sellers[i][1] * sellers[i][2]
        for i in range(len(buyers)):
            rewards[buyers[i][0]] += 1 * buyers_spending/total_demand * buyers[i][2] # buyers[i][2] is neagtive thats why not subtracting

    else:

        temp = total_demand
        i=0
        while(i<len(sellers)):
            p = sellers[i][1]
            val = prices_dict.get(p)
            if val == 1:
                if (sellers[i][2] <= temp):
                    rewards[sellers[i][0]] += sellers[i][1] * sellers[i][2]
                    buyers_spending += sellers[i][2] * sellers[i][1]
                    temp -= sellers[i][2]
                elif (temp < sellers[i][2] and temp > 0):
                    rewards[sellers[i][0]] += temp * sellers[i][1]
                    buyers_spending += sellers[i][1] * temp
                    rewards[sellers[i][0]] += (sellers[i][2] - temp) * (grid_price - 5)
                    temp -= temp
                else:
                    rewards[sellers[i][0]] += sellers[i][2] * (grid_price - 5)
                i+=1

            elif val>1:
                store = 0
                reward_common = 0
                for k in range(i,i+val):
                    store += sellers[k][2]
                if store != 0:
                    if (store<=temp):
                        reward_common = sellers[i][1]*store
                        buyers_spending += store * sellers[i][1]
                        temp -= store
                    elif (temp < store and temp > 0):
                        reward_common = temp*sellers[i][1]
                        buyers_spending += sellers[i][1] * temp
                        reward_common += (store-temp)*(grid_price-5)
                        temp -= temp
                    else:
                        reward_common = store*(grid_price-5)
                    for k in range(i,i+val):
                        rewards[sellers[k][0]] += (reward_common*sellers[k][2])/store
                i = i+val


        for i in range(len(buyers)):
            rewards[buyers[i][0]] += 1 * buyers_spending/total_demand * buyers[i][2] # buyers[i][2] is negative 

    return(rewards)


state_size = 5
max_battery = 10
max_energy_generated = 10
max_received = 10
min_non_adl = 3
max_non_adl = 6 
grid_price = 20

c_values = [10,20,30]
c = c_values[2]

total_iterations = 1200000
num_of_agents = 3
lamb = [[2.667e-07, 0.541, 6.5965, 4.3712],[8.8281, 10.2997, 9.8301, 9.7514],[8.8281, 10.2997, 9.8301, 9.7514]] # guess we have to put the lambda values

agents = []
names = ['g1','g2','g3']
for i in range(num_of_agents-1):
	agents.append(DQN_Agent(names[i],state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl, grid_price, total_iterations-200000, 0, lamb[i]))
agents.append(DQN_Agent_Constant_Price(names[2],state_size, max_battery, max_energy_generated, max_received, min_non_adl, max_non_adl, grid_price, total_iterations-200000, 0, lamb[1]))


states_price = []
states_adl = []

rewards = []
battery = []
renewable = []

total_reward_for_display = [[],[],[]]
total_prices_for_display = [[],[],[]]


for i in range(num_of_agents):
	battery.append(0)
	renewable.append(agents[i].get_renewable(1))


for i in range(num_of_agents):
	states_adl.append([renewable[i] + battery[i] - min(agents[i].non_adl), min(agents[i].non_adl), 7, 1, grid_price, 0]) #Changed  #Annanya Replied   # I think intially the adl_state_second is 0 for all 

adl_actions = []
adl_values = []

for i in range(num_of_agents):
	act = agents[i].adl_action(states_adl[i])
	adl_actions.append(act)
	adl_values.append(agents[i].adl_convert_allowed_indices_to_values(act))

for i in range(num_of_agents):
	states_price.append([states_adl[i][0], states_adl[i][1], adl_actions[i], states_adl[i][3], states_adl[i][4]])


for Iter in range(total_iterations):
	
	pricing_actions = []
	pricing_values = []
	for i in range(num_of_agents):
		act = agents[i].pricing_action(states_price[i])
		pricing_actions.append(act)
		pricing_values.append([i] + agents[i].pricing_convert_allowed_indices_to_values(act))

	adl_logger.info("{} {} {}".format(adl_actions[0],adl_actions[1],adl_actions[2]))
	renewable_logger.info("{} {} {}".format(renewable[0],renewable[1],renewable[2]))
	transmission_logger.info("{} {} {}".format(pricing_values[0][2],pricing_values[1][2],pricing_values[2][2])) 
	battery_logger.info("{} {} {}".format(battery[0],battery[1],battery[2]))
	price_logger.info("{} {} {}".format(pricing_values[0][1],pricing_values[1][1],pricing_values[2][1]))
	nd_logger.info("{} {} {}".format(states_price[0][0],states_price[1][0],states_price[2][0]))


	rewards = transaction(pricing_values)

	for i in range(num_of_agents):
		if (states_price[i][0] -adl_values[i]- pricing_values[i][2] < 0):
			rewards[i] += c * (states_price[i][0] -adl_values[i]- pricing_values[i][2])
			battery[i] = 0
		else:
			battery[i] = states_price[i][0] -adl_values[i]- pricing_values[i][2]
			battery[i] = min(battery[i], max_battery)
		
	adl_states =[]
	adl_secondary_states = [] # Changed

	for i in range(num_of_agents):
		penalty, updated_adl_state, updated_adl_secondary_state = agents[i].update_adl(adl_actions[i], states_adl[i][3])
		rewards[i] += c * -1 * penalty
		adl_states.append(updated_adl_state)
		adl_secondary_states.append(updated_adl_secondary_state) #Changed
		total_reward_for_display[i].append(rewards[i])
		total_prices_for_display[i].append(pricing_values[i][1])
	reward_logger.info("{} {} {}".format(adl_secondary_states[0],adl_secondary_states[1],adl_secondary_states[2]))

	temp_states_adl = []

	for i in range(num_of_agents):
		renewable[i] = agents[i].get_renewable((Iter+ 1)%4 + 1)
		temp = agents[i].get_non_adl_demand((Iter+1)%4 + 1)
		temp_states_adl.append([renewable[i] + battery[i] - temp, temp,adl_states[i] ,(Iter+1)%4 + 1, grid_price, adl_secondary_states[i]]) # Changed

	temp_adl_actions = []
	temp_adl_values = [] 
	temp_states_price = []

	for i in range(num_of_agents):
		act = agents[i].adl_action(temp_states_adl[i])
		temp_adl_actions.append(act)
		temp_adl_values.append(agents[i].adl_convert_allowed_indices_to_values(act))

	for i in range(num_of_agents):
		temp_states_price.append([temp_states_adl[i][0], temp_states_adl[i][1], temp_adl_actions[i], temp_states_adl[i][3], temp_states_adl[i][4]])

	for i in range(num_of_agents):
		agents[i].remember(states_adl[i], states_price[i], adl_actions[i], pricing_actions[i], rewards[i] , temp_states_adl[i], temp_states_price[i])



	if (Iter+1)%4 == 0:
		loss = [(0,0),(0,0),(0,0)]
		for i in range(num_of_agents):
			if Iter>7000:
				loss[i] = agents[i].replay(16)
		logger_loss.info("{} {} {}".format(loss[0],loss[1],loss[2]))
		logger_loss1.info("{} {} {}".format(loss[0][0],loss[1][0],loss[2][0]))
		logger_loss2.info("{} {} {}".format(loss[0][1],loss[1][1],loss[2][1]))

	if ((Iter + 1) % 500000 == 0) :
		for i in range(num_of_agents):
			agents[i].save_model()

	adl_values = temp_adl_values
	states_adl = temp_states_adl
	states_price = temp_states_price
	adl_actions = temp_adl_actions

	if Iter%10000 ==9999:
		print("Number of iterations completed = {} and reaming = {}".format(Iter+1,total_iterations-Iter-1))
		logger_updates.info("Iteration number {}".format(Iter+1))
		logger_updates.info('The average reward for agent 1 after is {}'.format(mean(total_reward_for_display[0])))
		logger_updates.info('The average reward for agent 2 after is {}'.format(mean(total_reward_for_display[1])))
		logger_updates.info('The average reward for agent 3 after is {}'.format(mean(total_reward_for_display[2])))
		logger_updates.info('The average prices for agent 1 after is {}'.format(mean(total_prices_for_display[0])))
		logger_updates.info('The average prices for agent 2 after is {}'.format(mean(total_prices_for_display[1])))
		logger_updates.info('The average prices for agent 3 after is {}'.format(mean(total_prices_for_display[2])))
		total_reward_for_display = [[],[],[]]
		total_prices_for_display = [[],[],[]]

for i in range(num_of_agents):
	agents[i].save_model()
