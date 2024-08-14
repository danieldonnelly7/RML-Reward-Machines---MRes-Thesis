from envs.office_world import OfficeWorld, Actions
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
from utils.learning_functions import learning_episode_office, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index, create_encoding
from utils.train_rml import rml_training_simple

config_path = './examples/office.yaml'

register(
    id='Office-v0',
    entry_point='envs.office_world_wrappers:OfficeRMLEnv',
    max_episode_steps=500
)

actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value]

training_class = rml_training_simple(learning_episode_office, RMLGym_Simple, actions, config_path, epsilon=0.2, gamma= 0.1, n=3)

results = training_class.get_test_statistics()

print(results)

with open('results/results_office_world_RML_simple_encoding.pkl', 'wb') as f:
    pickle.dump(results, f) 