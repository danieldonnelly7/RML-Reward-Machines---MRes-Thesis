import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style
sns.set_theme(style="whitegrid")

# Set font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Load the results from the pickle files
with open('results/results_LetterEnv_RML_num_encoding.pkl', 'rb') as f:
    num_encoding_results = pickle.load(f)

with open('results/results_LetterEnv_RML_one_hot_encoding.pkl', 'rb') as f:
    one_hot_results = pickle.load(f)

with open('results/results_LetterEnv_RML_simple_encoding.pkl', 'rb') as f:
    simple_results = pickle.load(f)

with open('results/convergence_results_CF-0.pkl', 'rb') as f:
    CRA_MDP_results = pickle.load(f)

# Extracting results from CRA_MDP_results
MDP_results = CRA_MDP_results['MDP Q-Learning']
CRA_results = CRA_MDP_results['CQL']

# Calculate means and standard deviations for num, one hot and simple
means_num = {n: np.mean(num_encoding_results.loc[num_encoding_results['n value'] == n]['steps']) for n in num_encoding_results['n value']}
std_devs_num = {n: np.std(num_encoding_results.loc[num_encoding_results['n value'] == n]['steps']) for n in num_encoding_results['n value']}

means_one_hot = {n: np.mean(one_hot_results.loc[one_hot_results['n value'] == n]['steps']) for n in one_hot_results['n value']}
std_devs_one_hot = {n: np.std(one_hot_results.loc[one_hot_results['n value'] == n]['steps']) for n in one_hot_results['n value']}

means_simple = {n: np.mean(simple_results.loc[simple_results['n value'] == n]['steps']) for n in simple_results['n value']}
std_devs_simple = {n: np.std(simple_results.loc[simple_results['n value'] == n]['steps']) for n in simple_results['n value']}


# Prepare data for plotting
n_values = list(means_one_hot.keys())

# Plotting Steps to Success for one hot and vector encoding
plt.figure(figsize=(10, 6))
plt.plot(n_values, list(means_num.values()), marker='o', label='Num Encoding Mean Steps', color='orange')
plt.fill_between(n_values, np.array(list(means_num.values())) - np.array(list(std_devs_num.values())), np.array(list(means_num.values())) + np.array(list(std_devs_num.values())), color='orange', alpha=0.1, label='Num Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_one_hot.values()), marker='o', label='One-Hot Encoding Mean Steps', color='blue')
plt.fill_between(n_values, np.array(list(means_one_hot.values())) - np.array(list(std_devs_one_hot.values())), np.array(list(means_one_hot.values())) + np.array(list(std_devs_one_hot.values())), color='blue', alpha=0.1, label='One-Hot Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_simple.values()), marker='o', label='Simple Encoding Mean Steps', color='green')
plt.fill_between(n_values, np.array(list(means_simple.values())) - np.array(list(std_devs_simple.values())), np.array(list(means_simple.values())) + np.array(list(std_devs_simple.values())), color='green', alpha=0.1, label='Simple Encoding ±1 Standard Deviation')

plt.xlabel('N value')
plt.ylabel('Steps required to obtain a solution')
plt.xticks(np.arange(min(n_values), max(n_values) + 1, 1))
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Comparison of Steps to Success for Different Encodings')
plt.show()


# Plotting Steps to Success on a log scale
plt.figure(figsize=(10, 6))
plt.plot(n_values, list(means_num.values()), marker='o', label='Num Encoding Mean Steps', color='orange')
plt.plot(n_values, list(means_one_hot.values()), marker='o', label='One-Hot Encoding Mean Steps', color='blue')
plt.plot(n_values, list(means_simple.values()), marker='o', label='Simple Encoding Mean Steps', color='grey')


# Plot MDP and CRA results
plt.plot(n_values, MDP_results, marker='o', label='MDP Q-Learning', color='green')
plt.plot(n_values, CRA_results, marker='o', label='CQL', color='red')

plt.xlabel('N value')
plt.ylabel('Steps required to obtain a solution (log scale)')
plt.yscale('log')
plt.xticks(np.arange(min(n_values), max(n_values) + 1, 1))
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Comparison of Steps to Success for Different Encodings and MDP/CRA')
plt.show()
