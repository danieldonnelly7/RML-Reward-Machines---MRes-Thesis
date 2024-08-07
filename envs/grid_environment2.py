import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.office_world import OfficeWorld, OfficeWorld_Delivery, Actions
from envs.letter_env import LetterEnv
import yaml

class GridEnv(gym.Env):
    def __init__(self, env, config_path, observation_space_size = 2, render_mode = None):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        #super().__init__(config_path = './examples/office.yaml')
        self.env = env
        N,M      = self.env.map_height, self.env.map_width
        self.action_space = spaces.Discrete(4) # up, right, down, left

        self.observation_space = spaces.Box(low=0, high=max([N,M]), shape=(observation_space_size,), dtype=np.uint8)  
        self.observation_dict  = spaces.Dict({'features': self.observation_space})
        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0


    def get_events(self):
        return self.env.get_true_propositions()

    def step(self, action):
        self.step_num += 1
        self.env.execute_action(action)
        obs = self.env.get_features()
        if self.step_num < self.max_steps:
            truncated = False
        else:
            truncated = True
        done = truncated

        reward = 0  # reward done at monitor_reward level
        #reward = super().monitor_reward(done, truncated) # all the reward comes from the RM
        info = {}
        return obs, reward, done, truncated, info

    def reset(self):
        self.env.reset()
        self.step_num = 0
        return self.env.get_features()

    def show(self):
        self.env.show()

    def get_model(self):
        return self.env.get_model()
    
    def render(self, mode='agent'):
        if mode == 'human':
            # commands
            str_to_action = {"w":0,"d":1,"s":2,"a":3}

            # play the game!
            done = True
            while True:
                if done:
                    print("New episode --------------------------------")
                    obs = self.reset()
                    self.env.show()
                    print("Features:", obs)
                    print("Events:", self.get_events())

                print("\nAction? (WASD keys or q to quite) ", end="")
                a = input()
                print()
                if a == 'q':
                    break
                # Executing action
                if a in str_to_action:
                    obs, rew, done, _, __ = self.step(str_to_action[a])
                    self.env.show()
                    print("Features:", obs)
                    print("Reward:", rew)
                    print("Events:", self.get_events())
                else:
                    print("Forbidden action")
        elif mode == 'agent':
            return self._render_agent()
        else:
            raise NotImplementedError
        
    def _render_agent(self):
        self.env.show()

class GridEnv_RNN(GridEnv):
    def __init__(self, env, config_path, observation_space_size = 2, render_mode = None, max_monitor_length=12):   #  observation_space_siz Default is 2 as it's (x,y). Else it adds the number of monitor states as well
        #super().__init__(config_path = './examples/office.yaml')
        self.env = env
        N,M, L      = self.env.map_height, self.env.map_width, len(self.env.object_list) + 2  # + 2 for x,y axis
        self.action_space = spaces.Discrete(4) # up, right, down, left

        # Observation spaces
        self.position_space = spaces.Box(low=0, high=max([N, M]), shape=(L,), dtype=np.uint8)  # (x, y) + objects
        self.monitor_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_monitor_length,20), dtype=np.float32)  # Monitor state of size 20

        self.observation_space = spaces.Dict({
            'position': self.position_space,
            'monitor': self.monitor_space
        })        
        
        #self.observation_dict  = spaces.Dict({'features': self.observation_space})
        
        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.max_steps = config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified
        self.step_num = 0  

    def reset(self):
        self.env.reset()
        self.step_num = 0
        self.monitor_state = self.env.monitor_state
        return {
            'position': self.env.get_features(),
            'monitor': self.monitor_state
        }    

    def step(self, action):
        self.env.execute_action(action)
        obs = self._get_observation()
        self.step_num += 1
        if self.step_num < self.max_steps:
            truncated = False
        else:
            truncated = True
        done = truncated

        reward = 0  # reward done at monitor_reward level
        info = {}
        return obs, reward, done, truncated, info

    def _get_observation(self):
        position = self.env.get_features()
        monitor = self.env.monitor_state
        return {
            'position': position,
            'monitor': monitor
        }


class OfficeRMLEnv(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = OfficeWorld()
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/office_new.yaml'
        super().__init__(self.env, config_path, render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

class OfficeRMLEnv_Delivery(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = OfficeWorld_Delivery()
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/office_new.yaml'
        super().__init__(self.env, config_path, render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

class OfficeRMLEnv_Delivery_PPO(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = OfficeWorld_Delivery()
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/office_new.yaml'
        super().__init__(self.env, config_path, observation_space_size=27, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

class RML_LetterEnv(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv()
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

