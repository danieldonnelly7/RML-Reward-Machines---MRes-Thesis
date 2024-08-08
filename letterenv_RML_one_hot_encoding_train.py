from envs.letterenv import LetterEnv, Actions
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym_One_Hot
from tqdm import tqdm
from utils.letterenv_functions import learning_episode, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index_one_hot, create_encoding_one_hot


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

unique_events, event_index = generate_events_and_index_one_hot(states_for_encoding)
initial_encoding = create_encoding_one_hot(states_for_encoding[0],event_index)


results_df_input = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])


def initialise_parameters():
    epsilon = 0.35
    alpha = 0.5
    gamma = 0.9
    correct_reward = 110
    return epsilon, alpha, gamma, correct_reward

def reset_environment():
    config_path = './examples/letter_env.yaml'
    epsilon, alpha, gamma, correct_reward = initialise_parameters()
    env = RMLGym_One_Hot(event_index, initial_encoding, config_path)
    q_table = {}
    succesful_policy = False
    num_episodes = 0
    total_steps = 0
    results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]

    return epsilon, alpha, gamma, correct_reward, env, q_table, succesful_policy, num_episodes, total_steps, results_df, actions 


def test_loop(event_index, initial_encoding,n):
    epsilon, alpha, gamma, correct_reward, env, q_table, succesful_policy, num_episodes, total_steps, results_df, actions = reset_environment()

    while succesful_policy == False:
        num_episodes += 1
        q_table, state, epsilon, total_steps = learning_episode(env, q_table, actions, alpha, gamma, epsilon, total_steps)
        succesful_policy, results_df = evaluation_episode_encoding(env, q_table, actions, 
                                                                        n, num_episodes, total_steps, results_df, correct_reward)
    return results_df

def get_test_statistics(event_index, initial_encoding, final_results_df,n):
    if final_results_df.empty:
        iteration = 0
    else:
        iteration = final_results_df['iteration'].iloc[-1]
    for i in tqdm(range(40)):
        j = 0
        iteration += 1
        while j < n:
            j += 1
            iteration_results_df = test_loop(event_index, initial_encoding,j)
            iteration_results_df['iteration'] = iteration
            final_results_df = pd.concat([final_results_df, iteration_results_df])

    return final_results_df



register(
    id='letter-env',
    entry_point='envs.new_letterenv_wrappers:RML_LetterEnv_5',
    max_episode_steps=200
)

the_test_results_df = get_test_statistics(event_index,initial_encoding, results_df_input,n=5)



print(the_test_results_df)


with open('results/results_LetterEnv_RML_one_hot_encoding.pkl', 'wb') as f:
    pickle.dump(the_test_results_df, f)

