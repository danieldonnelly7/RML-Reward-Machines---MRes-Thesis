import gymnasium as gym
import numpy as np
import pandas as pd
import random
from enum import Enum
from gymnasium.envs.registration import registry
from gymnasium.envs.registration import load_env_creator
from gymnasium.envs.registration import register
from rml.rmlgym import RMLGym
from tqdm import tqdm
from utils.learning_functions import learning_episode_office, learning_episode_letter, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index, create_encoding
import copy


class rml_training():
    def __init__(self, learn_func, rmlgym, states_for_encoding, actions, config_path, n, epsilon=0.35, alpha=0.5, gamma=0.9, correct_reward=110):
        
        self.unique_events, self.event_index = generate_events_and_index(states_for_encoding)
        self.initial_encoding = create_encoding(states_for_encoding[0],self.event_index)

        self.results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func

    def initialise_parameters(self):
        epsilon = copy.deepcopy(self.epsilon)
        alpha = copy.deepcopy(self.alpha)
        gamma = copy.deepcopy(self.gamma)
        correct_reward = copy.deepcopy(self.correct_reward)
        return epsilon, alpha, gamma, correct_reward

    def test_loop(self,n):
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        env = self.rmlgym(self.event_index, self.initial_encoding, self.config_path)
        q_table = {}
        succesful_policy = False
        num_episodes = 0
        total_steps = 0
        results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])

        while succesful_policy == False:
            num_episodes += 1
            succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(env, q_table, self.actions, alpha, gamma, epsilon, total_steps,n)

            if succesful_policy:
                new_row = pd.DataFrame([{'n value': n, 'episodes': num_episodes, 'steps': total_steps}])
                results_df = pd.concat([results_df, new_row])
                print('(n val, steps) - ', (n, total_steps))
        return results_df

    def get_test_statistics(self):
        if self.results_df.empty:
            iteration = 0
        else:
            iteration = self.results_df['iteration'].iloc[-1]
        for i in tqdm(range(40)):
            j = 0
            iteration += 1
            while j < self.n:
                j += 1
                iteration_results_df = self.test_loop(j)
                iteration_results_df['iteration'] = iteration
                self.results_df = pd.concat([self.results_df, iteration_results_df])

        return self.results_df
    
class rml_training_simple(rml_training):
    def __init__(self, learn_func, rmlgym, actions, config_path, n, epsilon=0.35, alpha=0.5, gamma=0.9, correct_reward=110):

        self.results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps', 'iteration'])
        self.config_path = config_path
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.correct_reward = correct_reward
        self.rmlgym = rmlgym
        self.actions = actions
        self.learning_function = learn_func

    def test_loop(self,n):
        epsilon, alpha, gamma, correct_reward = self.initialise_parameters()
        env = self.rmlgym(self.config_path)
        q_table = {}
        succesful_policy = False
        num_episodes = 0
        total_steps = 0
        results_df = pd.DataFrame(columns=['n value', 'episodes', 'steps'])

        while succesful_policy == False:
            num_episodes += 1
            succesful_policy, q_table, state, epsilon, total_steps = self.learning_function(env, q_table, self.actions, alpha, gamma, epsilon, total_steps,n)

            if succesful_policy:
                new_row = pd.DataFrame([{'n value': n, 'episodes': num_episodes, 'steps': total_steps}])
                results_df = pd.concat([results_df, new_row])
                print('(n val, steps) - ', (n, total_steps))
        return results_df
