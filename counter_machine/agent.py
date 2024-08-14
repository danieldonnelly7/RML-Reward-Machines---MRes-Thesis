from collections import defaultdict
import numpy as np


class CounterMachineAgent:
    def __init__(
        self,
        machine,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
        action_space,
    ):
        self.machine = machine
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.reset_training()

    def reset_training(self):
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.epsilon = self.initial_epsilon

    def reset(self):
        self.u = self.machine.u_0
        self.counters = tuple(0 for _ in list(self.machine.delta_u.keys())[0][2])

    def get_action(self, obs):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)

        if np.random.random() < self.epsilon or np.all(self.Q[machine_state] == 0):
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[machine_state])

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        # Remove propositions
        o = obs[:2]
        next_o = next_obs[:2]

        machine_state = o + (self.u,) + (self.counters,)

        next_u, next_counters, reward = self.machine.transition(
            props, self.u, self.counters
        )
        if next_u in self.machine.F:
            self.Q[machine_state][action] += self.learning_rate * (
                reward - self.Q[machine_state][action]
            )
        else:
            next_machine_state = next_o + (next_u,) + (next_counters,)

            self.Q[machine_state][action] += self.learning_rate * (
                reward
                + self.discount_factor * np.max(self.Q[next_machine_state])
                - self.Q[machine_state][action]
            )

        self.u = next_u
        self.counters = next_counters

    def get_greedy_action(self, obs):
        # Remove propositions
        o = obs[:2]
        machine_state = o + (self.u,) + (self.counters,)
        return np.argmax(self.Q[machine_state])

    def step(self, next_obs):
        props = next_obs[3]
        next_u, next_counters, _ = self.machine.transition(props, self.u, self.counters)
        self.u = next_u
        self.counters = next_counters

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def terminated(self):
        return self.u in self.machine.F


class CounterMachineCRMAgent(CounterMachineAgent):
    def reset_training(self):
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.observed_counters = defaultdict(lambda: set())
        self.epsilon = self.initial_epsilon

    def update(self, obs, action, next_obs, reward, terminated):
        props = next_obs[3]
        o = obs[:2]
        next_o = next_obs[:2]
        # Store observed counter states
        self.observed_counters[self.u].add(self.counters)

        for u_i in self.machine.U:
            for c_j in self.observed_counters[u_i]:
                counterfactual_state = o + (u_i,) + (c_j,)
                u_k, c_k, r_k = self.machine.transition(props, u_i, c_j)

                if u_k in self.machine.F:
                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k - self.Q[counterfactual_state][action]
                    )
                else:
                    next_counterfactual_state = next_o + (u_k,) + (c_k,)

                    self.Q[counterfactual_state][action] += self.learning_rate * (
                        r_k
                        + self.discount_factor
                        * np.max(self.Q[next_counterfactual_state])
                        - self.Q[counterfactual_state][action]
                    )

        next_u, next_counters, _ = self.machine.transition(props, self.u, self.counters)
        self.u = next_u
        self.counters = next_counters

