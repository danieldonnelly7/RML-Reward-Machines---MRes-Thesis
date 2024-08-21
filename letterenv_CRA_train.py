import os
import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
from agents.counter_machine import (
    ContextFreeCounterMachine,
    CounterMachineCRMAgent,
)

from agents.reward_machine import (
    ContextFreeRewardMachine,
    RewardMachineRSCRMAgent
)

from environments import (
    create_context_free_env_labelled,
    create_context_free_env_mdp
)
from environments.context_free import (
    labelled_action_space,
    mdp_action_space,
)
from utils.train import train_till_conv_repeat

SEED = 0

N_EPISODES = [
    25000,
    50000,
    100000,
    1000000,
    1000000,
]


agent_kwargs = {
    "initial_epsilon": 1.0,
    "final_epsilon": 0.01,
    "epsilon_decay": 1.0 / 1000000,
    "learning_rate": 0.01,
    "discount_factor": 0.99,
}
train_conv_kwargs = {
    "n_repeats": 1,
}



def create_counter_crm_agent():
    return CounterMachineCRMAgent(
        machine=ContextFreeCounterMachine(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )

def create_reward_machine_rs_crm_agent(n):
    return RewardMachineRSCRMAgent(
        machine=ContextFreeRewardMachine(N=n),
        action_space=labelled_action_space,
        **agent_kwargs,
    )

def get_agent_env_pairs(n):
    return (
        (
            "CQL",
            create_counter_crm_agent(),
            create_context_free_env_labelled(n),
        ),
        (
            "CRM+RS",
            create_reward_machine_rs_crm_agent(n),
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

        for name, agent, env in get_agent_env_pairs(n):
            print(f"Training started for {name}")
            agent.epsilon_decay = 1.0 / (0.5 * N_EPISODES[n - 1])
            train_conv_kwargs["max_samples"] = 100 * N_EPISODES[n - 1]

            if "CQL" in name:
                    if n > 1:
                        # Copy results from n = 1
                        convergence_results[name][n] = convergence_results[name][1]
                    else:
                        convergence_results[name] = {1 :{'iteration':[],'result':[]}}
                        for i in range(1, 21):
                            conv_results = train_till_conv_repeat(agent, env, **train_conv_kwargs)
                            print(conv_results)
                            convergence_results[name][1]['iteration'].append(i)
                            convergence_results[name][1]['result'].append(conv_results)
            else:
                if n == 1:
                    np.random.seed(int(SEED))
                conv_results = train_till_conv_repeat(agent, env, **train_conv_kwargs)
                if name not in convergence_results:
                    convergence_results[name] = []
                convergence_results[name].append(conv_results)
    print(convergence_results)
    
    with open(f"results/convergence_results_CF-{SEED}.pkl", "wb") as f:
        pickle.dump(dict(convergence_results), f)

    with open(f"results/convergence_results_CF-{SEED}.pkl", "rb") as f:
        convergence_results = pickle.load(f)


