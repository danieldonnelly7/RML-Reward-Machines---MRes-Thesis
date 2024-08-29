from envs.letterenv import LetterEnv, Actions
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

config_path = './examples/letter_env.yaml'


register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:RML_LetterEnv_5',
    max_episode_steps=200
)

actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]

training_class = rml_training(learning_episode_letter, RMLGym, states_for_encoding, actions, config_path, n=5)


results = training_class.get_test_statistics()

print(results)


with open('results/results_LetterEnv_RML_num_encoding.pkl', 'wb') as f:
    pickle.dump(results, f)

