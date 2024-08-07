import yaml
import pickle
import random
from envs.letter_env import LetterEnv
import pandas as pd
import numpy as np

def learning_episode(env, q_table, actions, alpha, gamma, epsilon, total_steps):
    state , _ = env.reset()
    state_tuple = tuple([state['position'],tuple(state['monitor'])])
    done = False
    total_reward = 0
    while not done:

        if state_tuple not in q_table:         # Add state to the q table
            q_table[state_tuple] = {action: 0 for action in actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            max_value = max(q_table[state_tuple].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in actions if q_table[state_tuple][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, _, __ = env.step(action)
        next_state_tuple = tuple([next_state['position'],tuple(next_state['monitor'])])

        if next_state_tuple not in q_table:    # Add next state to the q table
            q_table[next_state_tuple] = {action: 0 for action in actions}

        # Q-learning update
        old_value = q_table[state_tuple][action]
        next_max = max(q_table[next_state_tuple].values())
        q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state_tuple = next_state_tuple
        total_steps += 1
        total_reward += reward
    # Decay epsilon
    epsilon *= 0.999
    
    return q_table, state, epsilon, total_steps

def evaluation_episode_encoding(env, q_table, actions, 
                        n, total_episodes, total_steps, result_table, event_index, reward_if_correct):
    """
    Code is used to evaluate when the environment, how long it takes to a succesful policy.
    Needs as input total training episodes and steps (as well as other relevant items)
    """
    
    succesful_policy = False
    env.env.set_n(n)
    state, _ = env.reset()
    state_tuple = tuple([state['position'],tuple(state['monitor'])])
    done = False
    total_reward = 0
    n_steps = 0

    while not done:
        if state_tuple in q_table:
            max_value = max(q_table[state_tuple].values())
            best_actions = [a for a in actions if q_table[state_tuple][a] == max_value]
            action = random.choice(best_actions)
            # Take action
            next_state, reward, done, _, __ = env.step(action)
            total_reward += reward

            state_tuple = tuple([next_state['position'],tuple(next_state['monitor'])])
            n_steps += 1
        else:
            #action = random.choice(valid_actions)
            done = True

    # Retrieving the final monitor state to check if it corresponds to the value for success
    """
    final_monitor_state = state_tuple[1]
    index_of_first_one = final_monitor_state.index(1)   # Find the index of the first occurrence of the value 1
    
    final_monitor_state_string = None      # Getting the string representing the monitor state
    for key, value in event_index.items():
        if value == index_of_first_one:
            final_monitor_state_string = key
            break
    #print(final_monitor_state_string, ' , reward - ', total_reward, ' , steps - ', n_steps, ' , last reward - ', reward)
    if final_monitor_state_string == '1':   # If succesful this code adds data to the table
    """
    if reward == reward_if_correct:
        print('n val - ', n)
        new_row = pd.DataFrame([{'n value': n, 'episodes': total_episodes, 'steps': total_steps}])
        result_table = pd.concat([result_table, new_row])
        succesful_policy = True

        
    return succesful_policy, result_table