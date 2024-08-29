import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the results from the pickle file
pickle_filename = "results/new_vs_old_results.pkl"

with open(pickle_filename, "rb") as file:
    results = pickle.load(file)

# Prepare the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 8))

labels = {'RMLGym_Old': 'RMLGym', 'simple_encoding': 'Simple Encoding', 
          'one_hot_encoding': 'One Hot Encoding', 'num_vector_encoding': 'Numerical Vector Encoding'}
# Iterate over each strategy and calculate the rolling average normalized reward
for name in results.keys():
    total_rewards = results[name]["total_rewards"]
    
    # Calculate the maximum reward achieved in any episode
    max_reward = max(total_rewards) if total_rewards else 1  # Avoid division by zero
    
    # Calculate rolling average of the last 100 episodes (or fewer if less than 100)
    rolling_avg_normalized_rewards = [
        np.mean(total_rewards[max(0, i-99):i+1]) / max_reward for i in range(len(total_rewards))
    ]
    
    # Plot the rolling average normalized rewards
    sns.lineplot(x=range(len(rolling_avg_normalized_rewards)), y=rolling_avg_normalized_rewards, label=labels[name])

# Customize the plot
#plt.title("Average Normalized Reward per Episode (Rolling Average over Last 100 Episodes)")
plt.xlabel("Episode")
plt.ylabel("Normalized Average Reward")
plt.legend(title="Approach")
plt.show()


# Print a table that shows the success rate per approach for 10,000 episodes
print("Success Rate per Approach (10,000 episodes):\n")
for name in results.keys():
    successes = results[name]["successes"]
    
    # Calculate success rate
    success_rate = len(successes) / 10000  # Assuming total episodes is 10,000
    print(f"{name}: Success Rate = {success_rate:.2%}")
