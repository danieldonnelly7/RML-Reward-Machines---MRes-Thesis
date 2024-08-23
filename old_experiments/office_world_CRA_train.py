import os
import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from agents.counter_machine.agent import (
    CounterMachineCRMAgent,
)

from agents.counter_machine.office_world.config import OfficeWorldCounterMachine

from environments.office_world_CRA_wrappers import (
    _create_office_world_env, create_office_world_labelled
)
from environments.office_world_CRA_wrappers import (
    labelled_action_space,
)
from utils.train import train_till_conv_repeat_office


N_EPISODES = [
    50000,
    100000,
    200000
]


agent_kwargs = {
    "initial_epsilon": 0.3, 
    "final_epsilon": 0.01,
    "epsilon_decay": 1/1000000, 
    "learning_rate": 0.5,  
    "discount_factor": 0.1,
}
train_conv_kwargs = {
    "n_repeats": 1,
}


def create_counter_crm_agent():
    return CounterMachineCRMAgent(
        machine=OfficeWorldCounterMachine(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )



def get_agent_env_pairs(n):
    return "CQL", create_counter_crm_agent(),create_office_world_labelled(n)



def load_progress(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            convergence_results = pickle.load(f)
        last_iter = convergence_results['iteration'].max()
        last_n = convergence_results.loc[convergence_results['iteration'] == last_iter, 'n'].max()        
        if last_n == 3:   # allows the loop to start at a new iteration
            last_n = 0
            last_iter += 1
        return convergence_results, last_n, last_iter
    else:
        return pd.DataFrame(columns=['n','no of samples','iteration']), 0, 1


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")


    N = 3
    number_of_repeats = 20
    results_file = "results/office_world_success_results_CQL.pkl"

    # Load progress if it exists
    convergence_results, last_n, last_iter = load_progress(results_file)

    for iter in range(last_iter,number_of_repeats+1):
        for n in range(1, N + 1):
            if n <= last_n:   # Resets training from where it left off
                continue
            print(f"Training started for N = {n}, iteration = {iter}")

            name, agent, env = get_agent_env_pairs(n)
            print(f"Training started for {name}")

            train_conv_kwargs["max_samples"] = 500 * N_EPISODES[n - 1]

            conv_results = train_till_conv_repeat_office(agent, env, **train_conv_kwargs)
            result_df = pd.DataFrame({'n' : [n], 'no of samples' : [conv_results], 'iteration' : [iter]})
            print('result')
            print(result_df)
            convergence_results = pd.concat([convergence_results,result_df],ignore_index=True)

            # Save results
            
            with open(results_file, "wb") as f:
                pickle.dump(convergence_results, f)
            
        last_n = 0   # Restting last_n so it doesn't intervene in the loop


    print(convergence_results)


