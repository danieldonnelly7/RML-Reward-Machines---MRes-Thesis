from envs.grid_environment2 import GridEnv, GridEnv_RNN
from envs.letter_env import LetterEnv, LetterEnv_RNN
import random

class RML_LetterEnv_n1(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=1)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False

class RML_LetterEnv_n2(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=2)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)
    
    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False

class RML_LetterEnv_n3(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=3)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False

class RML_LetterEnv_n4(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=4)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False

class RML_LetterEnv_n5(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=5)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.env.n = n

class RML_LetterEnv_n6(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv(n=6)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.env.n = n


class RML_LetterEnv_n3_RNN(GridEnv_RNN):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):
        self.env = LetterEnv_RNN(n=3)
        self.forbidden_transitions = self.env.forbidden_transitions
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, observation_space_size=26, render_mode = render_mode)

    def get_monitor_state(self, state):
        self.env.get_monitor_state(state)

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False

    def set_n(self, n):
        self.env.n = n