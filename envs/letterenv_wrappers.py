from envs.grid_environment import GridEnv, GridEnv_RNN
from envs.letterenv import LetterEnv

class RML_LetterEnv_5(GridEnv):
    metadata = {'render_modes': [22]}
    def __init__(self, number_of_monitor_states = 20, render_mode = None):
        self.N = 5
        self._initialize_env()
        config_path = './examples/letter_env.yaml'
        super().__init__(self.env, config_path, monitor_states=number_of_monitor_states, render_mode = render_mode)
    
    def _initialize_env(self):
        self.env = LetterEnv(
            n_rows=6,
            n_cols=6,
            locations={
                "A": (1, 1),
                "B": (1, 4),
                "C": (4, 1),
            },
            max_observation_counts={
                "A": self.N,
                "B": None,
                "E": None,
                "C": None,
            },
            replacement_mapping={"A": "E"},
            task_string="A" * self.N + "E" + "B" + "C" * self.N
        )

    def evaluation_start(self):
        self.env.evaluation_n = True
    
    def evaluation_end(self):
        self.env.evaluation_n = False
    
    def set_n(self, n):
        self.N = n
        self._initialize_env()

class RML_LetterEnv_5_Simple(RML_LetterEnv_5):
    metadata = {'render_modes': [22]}
    def __init__(self, render_mode = None):  # Only change from normal is monitor_states = 1 instead of 20
        super().__init__(number_of_monitor_states=1, render_mode = render_mode)
