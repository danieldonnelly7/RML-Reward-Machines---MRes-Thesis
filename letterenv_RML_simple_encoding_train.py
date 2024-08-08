from envs.letterenv import LetterEnv, Actions
import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym_Simple
from tqdm import tqdm
from utils.letterenv_functions import learning_episode, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle


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
    env = RMLGym_Simple(config_path)
    q_table = {}
    succesful_policy = False
    num_episodes = 0
    total_steps = 0
    results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]

    return epsilon, alpha, gamma, correct_reward, env, q_table, succesful_policy, num_episodes, total_steps, results_df, actions 


def test_loop(n):
    epsilon, alpha, gamma, correct_reward, env, q_table, succesful_policy, num_episodes, total_steps, results_df, actions = reset_environment()

    while succesful_policy == False:
        num_episodes += 1
        q_table, state, epsilon, total_steps = learning_episode(env, q_table, actions, alpha, gamma, epsilon, total_steps)
        succesful_policy, results_df = evaluation_episode_encoding(env, q_table, actions, 
                                                                        n, num_episodes, total_steps, results_df, correct_reward)
    return results_df

def get_test_statistics(final_results_df,n):
    if final_results_df.empty:
        iteration = 0
    else:
        iteration = final_results_df['iteration'].iloc[-1]
    for i in tqdm(range(40)):
        j = 0
        iteration += 1
        while j < n:
            j += 1
            iteration_results_df = test_loop(j)
            iteration_results_df['iteration'] = iteration
            final_results_df = pd.concat([final_results_df, iteration_results_df])

    return final_results_df



register(
    id='letter-env',
    entry_point='envs.new_letterenv_wrappers:RML_LetterEnv_5_Simple',
    max_episode_steps=200
)

the_test_results_df = get_test_statistics(results_df_input,n=5)



print(the_test_results_df)


with open('results/results_LetterEnv_RML_simple_encoding.pkl', 'wb') as f:
    pickle.dump(the_test_results_df, f)

