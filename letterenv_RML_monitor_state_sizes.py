from envs.letterenv import LetterEnv, Actions
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym, RMLGym_One_Hot, RMLGym_Simple
from tqdm import tqdm
from utils.learning_functions import learning_episode_letter, evaluation_episode_encoding
from utils.train_rml import rml_training
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index, create_encoding


states_for_encoding = {0: '@(eps*(star(not_abcd:eps)*app(,[0])),[=gen([n],(a_match:eps)*(star(not_abcd:eps)*(app(,[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1])))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          1: '@(star(not_abcd:eps)*(app(gen([n],),[0+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[0+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          2: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[0+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[0+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          3: '@(star(not_abcd:eps)*(app(gen([n],),[1+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[1+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          4: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[1+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[1+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          5: '@(star(not_abcd:eps)*(app(gen([n],),[2+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[2+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          6: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[2+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[2+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          7: '@(star(not_abcd:eps)*(app(gen([n],),[3+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[3+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          8: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[3+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[3+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          9: 'false_verdict', 
          10: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[1]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          11: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          12: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[2]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          13: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[2]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          14: '@(app(gen([n],),[2]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          15: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[2-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          16: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[3]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          17: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[3]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          18: '@(app(gen([n],),[1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          19: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[1-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          20: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[4]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          21: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[4]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          22: '@(star(not_abcd:eps)*(app(gen([n],),[4+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[4+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          23: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[4+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[4+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          24: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[5]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
          25: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[5]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          26: '@(app(gen([n],),[4]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          27: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[4-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          28: '@(app(gen([n],),[3]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          29: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[3-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          30: '@(app(gen([n],),[4-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          31: '@(app(gen([n],),[3-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          32: '@(app(gen([n],),[2-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          33: '@(app(gen([n],),[1-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          34: '1', 
          35: '@(app(gen([n],),[5]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          36: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[5-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
          37: '@(app(gen([n],),[5-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])'}

def learning_loop(n, env, alpha=0.5, gamma=0.9, epsilon=0.35):
    
    succesful_policy = False
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    q_table = {}
    observed_monitor_states = []


    while not succesful_policy:
        steps = 0
        env.env.set_n(n)
        state , _ = env.reset()
        done = False


        if isinstance(state['monitor'], int):
            state_tuple = (state['position'], (state['monitor']))
        else:
            state_tuple = (state['position'], tuple(state['monitor']))

        while not done and steps < 200:
            if state_tuple[1] not in observed_monitor_states:
                observed_monitor_states.append(state_tuple[1])
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
            next_state, reward, done, _, __ = env.step(action)#
            if isinstance(next_state['monitor'], int):
                next_state_tuple = (next_state['position'], (next_state['monitor']))
            else:
                next_state_tuple = (next_state['position'], tuple(next_state['monitor']))

            if next_state_tuple not in q_table:    # Add next state to the q table
                q_table[next_state_tuple] = {action: 0 for action in actions}
                reward += 2

            # Q-learning update
            old_value = q_table[state_tuple][action]
            next_max = max(q_table[next_state_tuple].values())
            q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state_tuple = next_state_tuple
            steps += 1

            if reward == 110:
                succesful_policy = True


        # Decay epsilon
        epsilon *= 0.99

    observed_states_no = len(observed_monitor_states)
    print('success, no of states - ', observed_states_no, ', n - ', n)
    return observed_states_no

config_path = './examples/letter_env.yaml'


register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:RML_LetterEnv_5',
    max_episode_steps=200
)

unique_events, event_index = generate_events_and_index(states_for_encoding)
initial_encoding = create_encoding(states_for_encoding[0],event_index)

results_df = pd.DataFrame(columns=['env', 'n_value', 'no_mon_states'])
for env in [RMLGym, RMLGym_One_Hot]:
    env_name = env.__name__
    for n in [1,2,3,4,5]:
        env_inst = env(event_index, initial_encoding, config_path)
        result = learning_loop(n,env_inst)
        new_row = pd.DataFrame([{'env': env_name, 'n_value': n, 'no_mon_states': result}])
        results_df = pd.concat([results_df, new_row])

for env in [RMLGym_Simple]:
    env_name = env.__name__
    for n in [1,2,3,4,5]:
        env_inst = env(config_path)
        result = learning_loop(n,env_inst)
        new_row = pd.DataFrame([{'env': env_name, 'n_value': n, 'no_mon_states': result}])
        results_df = pd.concat([results_df, new_row])

print(results_df)

with open('results/machine_sizes_letterenv_RML.pkl', 'wb') as f:
    pickle.dump(results_df, f)