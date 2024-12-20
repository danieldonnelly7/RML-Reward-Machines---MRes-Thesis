import json
import re

import websocket
import numpy as np
import gymnasium as gym
import yaml
from typing import TypeVar, Tuple

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

def make(config_path: str, env=None) -> "RMLGym_Old":
    return RMLGym_Old(config_path, env)

class RMLGym_Old(gym.core.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    
    Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    """
    def __init__(self, config_path: str, env=None, render_mode = None):
        """
        TODO: description
        """
        # Read the config YAML file
        with open(config_path, "r") as stream:
            try:    
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # print(config_dict)
        # Make the environment if it is not already provided

        if env is not None:
            self.env = env
        else:
            self.env = gym.make(config_dict['env_name'], render_mode = render_mode)
        self.config_dict = config_dict
        self._action_space = None
        self._observation_space = None
        self._reward_range = None
        self._metadata = None
        self.data = dict()
        #self.json_data = dict() #Json data for rmlgym
        self.data['time'] = []
        self.step_num = 0
        # Create a WebSocket object.
        

        # Pull time information if it is provided
        self.timestep = 1 if 'timestep' not in config_dict.keys() else config_dict['timestep']


        # Sort through specified variables that will be tracked
        self.rml_variables = config_dict['variables']
        self.rewards = config_dict['reward']
        for i in self.rml_variables:
            self.data[i['name']] = []
        self.data['action'] = []

        self.max_steps = self.config_dict.get('max_episode_steps', 200)  # Default to 200 if not specified

        
    def __getattr__(self, name):
        print(self)
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute {name}")
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def action_space(self):
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        self._observation_space = space

    @property
    def reward_range(self):
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value):
        self._reward_range = value

    @property
    def metadata(self):
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        #o, reward, done, truncated, info = self.env.step(action)
        o, reward, done, truncated, info = self.env.step(action)
        # Record and increment the time
        self.step_num += 1
        if self.step_num >= self.max_steps:
            truncated = True
            done = True

        observations = dict()
        # observations['action'] = action
        # Add variables to their lists
        for i in self.rml_variables:
            if i['location'] == 'obs':
                observations['location'] = i['location']
                observations[i['name']] = float(o[i['identifier']])
            elif i['location'] == 'info':
                observations[i['name']] = float(info[i['identifier']])
            elif i['location'] == 'state':
                observations[i['name']] = float(self.__getattr__(i['identifier']))
            else:
                # make an error for this
                print('ERROR ERROR')
        
        self.data = observations

        # Calculate the reward
        reward, reward_info = self.monitor_reward(done, truncated)
        if self.monitor_state_unencoded == '1' or self.monitor_state_unencoded == 'false_verdict':
            done = True
        #info.update(reward_info)
        #print(self.data)
        if done or truncated:
            self.reset()
        return o, reward, done, truncated ,info
        #return o, reward, done, info

    def reset(self, seed=None, options=None, **kwargs):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        # Reset the RML variable data
        self.step_num = 0
        self.reset_monitor()
        for key in self.data.keys():
            self.data[key] = []
        return 1, self.env.reset(**kwargs)    # If encountering the office world error making this 1, self.env.reset(**kwargs) works

    def reset_monitor(self):
        self.data['terminate'] = True
        json_string = json.dumps(self.data)
        ws = websocket.WebSocket()
        host = f'ws://{self.config_dict["host"]}:{self.config_dict["port"]}'
        # Connect the WebSocket object to the server.
        ws.connect(host)
        ws.send(json_string)
        # Receive response from the server
        response = ws.recv()
        # Convert the JSON string to a Python dictionary
        #monitor_rew = bool(response[1:])
        ws.close()
    
    def render(self, mode="human", **kwargs):
        #return self.env.render(mode, **kwargs)
        return self.env.render(**kwargs)

    def close(self):
        #self.ws.close()
        #self.ws.close(1000, "Normal closure")
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
    def monitor_reward(self, done: bool, truncated) -> Tuple[float, dict]:
        #print(data)
        #print(self.step_num == self.max_steps)
        #info = dict(self.data)

        if truncated == True:
            self.data['terminate'] = True
        else:
            self.data['terminate'] = False
        
        json_string = json.dumps(self.data)
        ws = websocket.WebSocket()
        host = f'ws://{self.config_dict["host"]}:{self.config_dict["port"]}'
        # Connect the WebSocket object to the server.
        ws.connect(host)
        #print('send', json_string)
        ws.send(json_string)
        # Receive response from the server
        response = ws.recv()
        # Convert the JSON string to a Python dictionary
        response = json.loads(response)
        #if response['a'] == float(1):
        self.monitor_state_unencoded = response['monitor_state']

        #print('response', response)
        #monitor_rew = bool(response[1:])
        ws.close()

        #response = json.loads(response)
        # Check if the response is valid
        reward = self.rewards[response['verdict']]
        #info[self.rewards['name']] = reward
        return reward, {}

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped