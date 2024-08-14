from envs.office_world import OfficeWorld, Actions
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
from utils.learning_functions import learning_episode_office, evaluation_episode_encoding
import matplotlib.pyplot as plt
import pickle
from utils.encoding_functions import generate_events_and_index, create_encoding
from utils.train_rml import rml_training

states_for_encoding = {0: '@(eps*(star(not_efgon:eps)*app(,[1])),[=gen([n],(mail_pick_up:eps)*(star(not_efgon:eps)*(app(,[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)])))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1))])', 
                       1: '@(star(not_efgon:eps)*(app(gen([n],),[1+1])\\/(out_of_mail:eps)*app(gen([s,m],),[1,1])),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       2: '@(eps*(star(not_efgon:eps)*(app(gen([n],),[1+1])\\/(out_of_mail:eps)*app(gen([s,m],),[1,1]))),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       3: '@(star(not_efgon:eps)*(app(gen([n],),[2+1])\\/(out_of_mail:eps)*app(gen([s,m],),[2,2])),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       4: '@(eps*(star(not_efgon:eps)*(app(gen([n],),[2+1])\\/(out_of_mail:eps)*app(gen([s,m],),[2,2]))),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       5: '@(star(not_efgon:eps)*(app(gen([n],),[3+1])\\/(out_of_mail:eps)*app(gen([s,m],),[3,3])),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       6: '@(eps*(star(not_efgon:eps)*(app(gen([n],),[3+1])\\/(out_of_mail:eps)*app(gen([s,m],),[3,3]))),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       7: '@(star(not_efgon:eps)*(app(gen([n],),[4+1])\\/(out_of_mail:eps)*app(gen([s,m],),[4,4])),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       8: '@(eps*(star(not_efgon:eps)*(app(gen([n],),[4+1])\\/(out_of_mail:eps)*app(gen([s,m],),[4,4]))),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       9: '@(star(not_efgon:eps)*(app(gen([n],),[5+1])\\/(out_of_mail:eps)*app(gen([s,m],),[5,5])),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       10: '@(eps*(star(not_efgon:eps)*(app(gen([n],),[5+1])\\/(out_of_mail:eps)*app(gen([s,m],),[5,5]))),[=(mail_pick_up:eps)*(star(not_efgon:eps)*(app(gen([n],),[var(n)+1])\\/(out_of_mail:eps)*app(,[var(n),var(n)]))),=gen([s,m],guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(,[var(s)-1,var(m)])),app(,[var(m)]))),=gen([s],guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(,[var(s)-1])),1)),=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       11: '@(app(gen([s,m],),[5,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       12: '@(eps*(star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[5-1,5]))),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       13: 'false_verdict', 
                       14: '@(app(gen([s,m],),[5-1,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       15: '@(eps*(star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[4-1,5]))),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       16: '@(app(gen([s,m],),[4-1,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       17: '@(eps*(star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[3-1,5]))),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       18: '@(app(gen([s,m],),[3-1,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       19: '@(eps*(star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[2-1,5]))),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       20: '@(app(gen([s,m],),[2-1,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       21: '@(eps*(star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[1-1,5]))),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       22: '@(app(gen([s,m],),[1-1,5]),[=guarded(var(s)>0,star(not_efgon:eps)*((coffee_pick_up:eps)*app(gen([s,m],),[var(s)-1,var(m)])),app(gen([s],),[var(m)])),=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       23: '@(eps*(star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[5-1]))),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       24: '@(app(gen([s],),[5-1]),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       25: '@(eps*(star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[4-1]))),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       26: '@(app(gen([s],),[4-1]),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       27: '@(eps*(star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[3-1]))),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       28: '@(app(gen([s],),[3-1]),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       29: '@(eps*(star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[2-1]))),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       30: '@(app(gen([s],),[2-1]),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])', 
                       31: '@(eps*(star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[1-1]))),[=guarded(var(s)>0,star(not_efgon:eps)*((drop_off:eps)*app(gen([s],),[var(s)-1])),1)])'}
config_path = './examples/office.yaml'

register(
    id='Office-v0',
    entry_point='envs.office_world_wrappers:OfficeRMLEnv',
    max_episode_steps=500
)

actions = [Actions.up.value, Actions.right.value, Actions.left.value, Actions.down.value]

training_class = rml_training(learning_episode_office, RMLGym_One_Hot, states_for_encoding, actions, config_path, epsilon=0.2, gamma= 0.1, n=3)

results = training_class.get_test_statistics()

print(results)

with open('results/results_office_world_RML_one_hot_encoding.pkl', 'wb') as f:
    pickle.dump(results, f) 