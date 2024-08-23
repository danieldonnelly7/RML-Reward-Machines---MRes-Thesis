import os
import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from envs.letterenv import Actions

import numpy as np
from agents.counter_machine.agent import (
    CounterMachineCRMAgent_NewAction
)

from agents.counter_machine.context_free.config_extra_reward import (
    ContextFreeCounterMachine_Rewards
)

from environments import (
    create_context_free_env_labelled,
    create_context_free_env_mdp
)
from environments.context_free import (
    labelled_action_space,
    mdp_action_space,
)
from utils.train import train_till_conv_repeat_letter

SEED = 0

N_EPISODES = [
    25000,
    50000,
    100000,
    1000000,
    1000000,
]


agent_kwargs = {
    "initial_epsilon": 0.1,
    "final_epsilon": 0.01,
    "epsilon_decay": 1.0 / 100000,
    "learning_rate": 0.5,
    "discount_factor": 0.99,
}
train_conv_kwargs = {
    "n_repeats": 1,
}



def create_counter_crm_agent():
    return CounterMachineCRMAgent_NewAction(
        machine=ContextFreeCounterMachine_Rewards(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )


def get_agent_env_pairs(n):
    return (
        (
            "CQL",
            create_counter_crm_agent(),
            create_context_free_env_labelled(n),
        )
    )


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")

    N = 5
    convergence_results = {}

    for n in range(1, N + 1):
        print(f"Training started for N = {n}")
        pairs = get_agent_env_pairs(n)
        name, agent, env = pairs[0], pairs[1], pairs[2]

        agent.epsilon_decay = 1.0 / (0.5 * N_EPISODES[n - 1])
        train_conv_kwargs["max_samples"] = 100 * N_EPISODES[n - 1]

        convergence_results[n] = []
        for i in range(1, 21):
            conv_results = train_till_conv_repeat_letter(Actions, agent, env, **train_conv_kwargs)
            print(conv_results)
            convergence_results[n].append(conv_results)

    print(convergence_results)
    
    with open(f"results/convergence_results_CF_CQL_improved.pkl", "wb") as f:
        pickle.dump(dict(convergence_results), f)

    with open(f"results/convergence_results_CF_CQL_improved.pkl", "rb") as f:
        convergence_results = pickle.load(f)


