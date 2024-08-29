import numpy as np
from envs.letterenv import Actions
from utils.new_vs_old_functions import learning_episode_new_vector, learning_episode_new_simple, learning_episode_old
from rml.rmlgym import RMLGym, RMLGym_One_Hot, RMLGym_Simple
from rml.rmlgym_old import RMLGym_Old
import pickle


# Define the number of episodes
episodes = 10000
config_path = './examples/letter_env.yaml'

alpha = 0.5
gamma = 0.9
epsilon = 1
epsilon_decay = 0.9999
hyperparameters = [alpha, gamma, epsilon, epsilon_decay]

# Initialize data structures to record results
results = {
    "RMLGym_Old": {"total_rewards": [], "successes": []},
    "simple_encoding": {"total_rewards": [], "successes": []},
    "one_hot_encoding": {"total_rewards": [], "successes": []},
    "num_vector_encoding": {"total_rewards": [], "successes": []},
}

# List of approaches (learning functions) to compare
approaches = [
    ("RMLGym_Old", RMLGym_Old ,learning_episode_old),
    ("simple_encoding", RMLGym_Simple ,learning_episode_new_simple),
    ("one_hot_encoding", RMLGym_One_Hot, learning_episode_new_vector),
    ("num_vector_encoding", RMLGym ,learning_episode_new_vector),
]

# Loop through each approach
for name, env, learning_function in approaches:
    print(f"Evaluating {name}...")

        # Apply the learning function for the current approach and episode
    total_reward, success = learning_function(env, config_path, Actions, episodes, hyperparameters)
        
        # Record the results
    results[name]["total_rewards"] = total_reward
    results[name]["successes"] = success
    print(success)


# Define the filename for the pickle file
pickle_filename = "results/new_vs_old_results.pkl"


# Save the results dictionary to a pickle file
with open(pickle_filename, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {pickle_filename}")


