import yaml
import pickle
import random
from envs.letter_env import LetterEnv
import pandas as pd
import numpy as np

def get_variables(config_path):  # Gets all the variables that are not x, y or f from the config file
    with open(config_path, "r") as stream:
        try:    
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    rml_variables = config_dict['variables']
    list_of_variables = ['none']
    for i in rml_variables:
        if i['name'] != 'x' and i['name'] != 'yy' and i['name'] != 'f':
            list_of_variables.append(i['name'])
    return list_of_variables


def get_valid_actions(state, forbidden_transitions_dict,actions): # Used to avoid actions that are invalid
    x, y = state[0], state[1]
    forbidden_actions = forbidden_transitions_dict.get((x, y), [])
    return [action for action in actions if action not in forbidden_actions]


def save_q_table_and_params(q_table, params, filename):
    with open(filename, 'wb') as f:
        pickle.dump((q_table, params), f)

def load_q_table_and_params(filename):
    with open(filename, 'rb') as f:
        q_table, params = pickle.load(f)
    return q_table, params

def save_q_table_and_params_and_encoding(q_table, params, initial_encoding, event_index, filename):
    with open(filename, 'wb') as f:
        pickle.dump((q_table, params, initial_encoding, event_index), f)

def load_q_table_and_params_and_encoding(filename):
    with open(filename, 'rb') as f:
        q_table, params, initial_encoding, event_index = pickle.load(f)
    return q_table, params, initial_encoding, event_index


def get_forbidden_transitions_dict(env):
    forbidden_transitions_dict = {}

    for item in env.forbidden_transitions:
        x = item[0]
        y = item[1]
        z = item[2]
        
        # Use (x, y) as the key
        if (x, y) not in forbidden_transitions_dict:
            forbidden_transitions_dict[(x, y)] = []
        
        # Append the action z to the list of forbidden actions for this (x, y)
        forbidden_transitions_dict[(x, y)].append(z.value)
    return forbidden_transitions_dict

def learning_episode(env, forbidden_transitions_dict, q_table, actions, alpha, gamma, epsilon, total_steps):
    state , _ = env.reset()
    done = False
    while not done:
        valid_actions = get_valid_actions(state, forbidden_transitions_dict,actions)

        if tuple(state) not in q_table:         # Add state to the q table
            q_table[tuple(state)] = {action: 0 for action in actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            max_value = max(q_table[tuple(state)].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in actions if a in valid_actions and q_table[tuple(state)][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, _, __ = env.step(action)

        if tuple(next_state) not in q_table:    # Add next state to the q table
            q_table[tuple(next_state)] = {action: 0 for action in actions}

        # Q-learning update
        old_value = q_table[tuple(state)][action]
        next_max = max(q_table[tuple(next_state)].values())
        q_table[tuple(state)][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_steps += 1
    # Decay epsilon
    epsilon *= 0.999
    
    return q_table, state, epsilon, total_steps


def evaluation_episode(env, q_table, forbidden_transitions_dict, actions):
    env.env.evaluation_start()
    state = env.env.reset()
    done = False
    total_reward = 0

    while not done:
        valid_actions = get_valid_actions(state, forbidden_transitions_dict, actions)
        if tuple(state) in q_table:
            max_value = max(q_table[tuple(state)].values())
            best_actions = [a for a in actions if a in valid_actions and q_table[tuple(state)][a] == max_value]
            action = random.choice(best_actions)
        else:
            #action = random.choice(valid_actions)
            succesful_policy = False
            env.env.evaluation_end()
            return succesful_policy, total_reward

        # Take action
        next_state, reward, done, _, __ = env.step(action)
        total_reward += reward

        state = next_state
    final_state = env.monitor_states[state[-1]]
    if final_state == '1':
        succesful_policy = True
    else:
        succesful_policy = False
    
    env.env.evaluation_end()
    return succesful_policy, total_reward


def evaluation_episode5(env, q_table, forbidden_transitions_dict, actions, 
                        remaining_n, total_episodes, total_steps, result_table):
    """
    Code is used to evaluate when the environment n = 5, how long it takes to get to eaach n value.
    Needs each n value as input as well as total training episodes and steps (as well as other relevant items)

    Needs to iterate through the n. If the model is succesful for an n, the n, number of episodes and the total steps 
    needs to be recorded. Additionally, the n value needs to be striked from the list of remaining ns so it is no longer
    tested
    """
    succesful_policy = False
    for n_val in remaining_n[:]:
        state = env.env.reset()
        env.env.set_n(n_val)
        done = False
        total_reward = 0

        while not done:
            valid_actions = get_valid_actions(state, forbidden_transitions_dict, actions)
            if tuple(state) in q_table:
                max_value = max(q_table[tuple(state)].values())
                best_actions = [a for a in actions if a in valid_actions and q_table[tuple(state)][a] == max_value]
                action = random.choice(best_actions)
                # Take action
                next_state, reward, done, _, __ = env.step(action)
                total_reward += reward

                state = next_state
            else:
                #action = random.choice(valid_actions)
                done = True

        final_state = env.monitor_states[state[-1]]
        if final_state == '1':
            remaining_n.remove(n_val)
            new_row = pd.DataFrame([{'n value': n_val, 'episodes': total_episodes, 'steps': total_steps}])
            result_table = pd.concat([result_table, new_row])

            if len(remaining_n) == 0:    # Terminate loop condition when no remianing_n
                succesful_policy = True

        
    return succesful_policy, remaining_n, result_table


def evaluation_episode5_encoding(env, q_table, forbidden_transitions_dict, actions, 
                        remaining_n, total_episodes, total_steps, result_table, event_index):
    """
    Code is used to evaluate when the environment n = 5, how long it takes to get to eaach n value.
    Needs each n value as input as well as total training episodes and steps (as well as other relevant items)

    Needs to iterate through the n. If the model is succesful for an n, the n, number of episodes and the total steps 
    needs to be recorded. Additionally, the n value needs to be striked from the list of remaining ns so it is no longer
    tested
    """
    succesful_policy = False
    for n_val in remaining_n[:]:
        state = env.env.reset()
        env.env.set_n(n_val)
        done = False
        total_reward = 0

        while not done:
            valid_actions = get_valid_actions(state, forbidden_transitions_dict, actions)
            if tuple(state) in q_table:
                max_value = max(q_table[tuple(state)].values())
                best_actions = [a for a in actions if a in valid_actions and q_table[tuple(state)][a] == max_value]
                action = random.choice(best_actions)
                # Take action
                next_state, reward, done, _, __ = env.step(action)
                total_reward += reward

                state = next_state
            else:
                #action = random.choice(valid_actions)
                done = True

        # Retrieving the final monitor state to check if it corresponds to the value for success
        final_monitor_state = state.tolist()
        final_monitor_state = final_monitor_state[6:]    
        index_of_first_one = final_monitor_state.index(1)   # Find the index of the first occurrence of the value 1
        
        final_monitor_state_string = None      # Getting the string representing the monitor state
        for key, value in event_index.items():
            if value == index_of_first_one:
                final_monitor_state_string = key
                break
        if final_monitor_state_string == '1':   # If succesful this code adds data to the table
            print('n val - ', n_val)
            remaining_n.remove(n_val)
            new_row = pd.DataFrame([{'n value': n_val, 'episodes': total_episodes, 'steps': total_steps}])
            result_table = pd.concat([result_table, new_row])
            print('remaining - ', remaining_n)

            if len(remaining_n) == 0:    # Terminate loop condition when no remianing_n
                succesful_policy = True

        
    return succesful_policy, remaining_n, result_table

def evaluation_episode_encoding_rnn(env, model, remaining_n, total_steps, result_table, event_index):
    """
    Code is used to evaluate rnn model and how long it takes to get to each n value.
    Needs each n value as input as well as total training episodes and steps (as well as other relevant items)

    Needs to iterate through the n. If the model is successful for an n, the n, number of episodes and the total steps 
    need to be recorded. Additionally, the n value needs to be striked from the list of remaining ns so it is no longer
    tested.
    """
    successful_policy = False
    for n_val in remaining_n[:]:
        state, _ = env.reset()
        env.env.set_n(n_val)
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, done, _, __ = env.step(action)
            total_reward += reward
            state = next_state


        # Retrieving the final monitor state to check if it corresponds to the value for success
        final_monitor_state = state['monitor'].tolist()
        final_monitor_state = final_monitor_state[0][6:]    
        index_of_first_one = final_monitor_state.index(1)   # Find the index of the first occurrence of the value 1
        
        final_monitor_state_string = None      # Getting the string representing the monitor state
        for key, value in event_index.items():
            if value == index_of_first_one:
                final_monitor_state_string = key
                break
        print('final monitor state string - ', final_monitor_state_string)
        if final_monitor_state_string == '1':   # If successful this code adds data to the table
            print('n val - ', n_val)
            remaining_n.remove(n_val)
            new_row = pd.DataFrame([{'n value': n_val, 'steps': total_steps}])
            result_table = pd.concat([result_table, new_row])
            print('remaining - ', remaining_n)

            if len(remaining_n) == 0:    # Terminate loop condition when no remaining_n
                successful_policy = True

    return successful_policy, remaining_n, result_table
