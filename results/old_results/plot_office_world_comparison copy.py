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
plt.rcParams['legend.fontsize'] = 8

# Load the results from the pickle files
with open('results/results_office_world_RML_num_encoding.pkl', 'rb') as f:
    num_encoding_results = pickle.load(f)
with open('results/results_office_world_RML_one_hot_encoding.pkl', 'rb') as f:
    one_hot_results = pickle.load(f)

with open('results/results_office_world_RML_simple_encoding.pkl', 'rb') as f:
    simple_results = pickle.load(f)

with open('results/office_world_success_results_CQL.pkl', 'rb') as f:
    CRA_results = pickle.load(f)

with open('results/office_world_success_results_CQL_improved.pkl', 'rb') as f:
    CRA_action_results = pickle.load(f)

# Calculate means and standard deviations for num, one hot and simple
means_num = {n: np.mean(num_encoding_results.loc[num_encoding_results['n value'] == n]['steps']) for n in num_encoding_results['n value']}
std_devs_num = {n: np.std(num_encoding_results.loc[num_encoding_results['n value'] == n]['steps']) for n in num_encoding_results['n value']}

means_one_hot = {n: np.mean(one_hot_results.loc[one_hot_results['n value'] == n]['steps']) for n in one_hot_results['n value']}
std_devs_one_hot = {n: np.std(one_hot_results.loc[one_hot_results['n value'] == n]['steps']) for n in one_hot_results['n value']}

means_simple = {n: np.mean(simple_results.loc[simple_results['n value'] == n]['steps']) for n in simple_results['n value']}
std_devs_simple = {n: np.std(simple_results.loc[simple_results['n value'] == n]['steps']) for n in simple_results['n value']}

#means_CRA = {n: np.mean(CRA_results.loc[CRA_results['n'] == n]['no of samples']) for n in CRA_results['n']}
#std_devs_CRA = {n: np.std(CRA_results.loc[CRA_results['n'] == n]['no of samples']) for n in CRA_results['n']}

means_CRA_IR = {n: np.mean(CRA_action_results.loc[CRA_action_results['n'] == n]['no of samples']) for n in CRA_action_results['n']}
std_devs_CRA_IR = {n: np.std(CRA_action_results.loc[CRA_action_results['n'] == n]['no of samples']) for n in CRA_action_results['n']}

# Prepare data for plotting
n_values = list(means_one_hot.keys())

# Plotting Steps to Success for one hot and vector encoding
plt.figure(figsize=(10, 6))
plt.plot(n_values, list(means_num.values()), marker='o', label='Num Encoding Mean Steps', color='orange')
plt.fill_between(n_values, np.array(list(means_num.values())) - np.array(list(std_devs_num.values())), np.array(list(means_num.values())) + np.array(list(std_devs_num.values())), color='orange', alpha=0.07, label='Num Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_one_hot.values()), marker='o', label='One-Hot Encoding Mean Steps', color='blue')
plt.fill_between(n_values, np.array(list(means_one_hot.values())) - np.array(list(std_devs_one_hot.values())), np.array(list(means_one_hot.values())) + np.array(list(std_devs_one_hot.values())), color='blue', alpha=0.07, label='One-Hot Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_simple.values()), marker='o', label='Simple Encoding Mean Steps', color='green')
plt.fill_between(n_values, np.array(list(means_simple.values())) - np.array(list(std_devs_simple.values())), np.array(list(means_simple.values())) + np.array(list(std_devs_simple.values())), color='green', alpha=0.07, label='Simple Encoding ±1 Standard Deviation')

plt.xlabel('N value')
plt.ylabel('Steps required to obtain a solution')
plt.xticks(np.arange(min(n_values), max(n_values) + 1, 1))
plt.legend(loc='upper left',ncol=1)
plt.grid(True)
plt.ylim(bottom=0)
#plt.title('Comparison of Steps to Success for Different Encodings')
plt.show()


# With CQL Approach
plt.figure(figsize=(10, 6))
plt.plot(n_values, list(means_num.values()), marker='o', label='Num Encoding Mean Steps', color='orange')
plt.fill_between(n_values, np.array(list(means_num.values())) - np.array(list(std_devs_num.values())), np.array(list(means_num.values())) + np.array(list(std_devs_num.values())), color='orange', alpha=0.07, label='Num Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_one_hot.values()), marker='o', label='One-Hot Encoding Mean Steps', color='blue')
plt.fill_between(n_values, np.array(list(means_one_hot.values())) - np.array(list(std_devs_one_hot.values())), np.array(list(means_one_hot.values())) + np.array(list(std_devs_one_hot.values())), color='blue', alpha=0.07, label='One-Hot Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_simple.values()), marker='o', label='Simple Encoding Mean Steps', color='green')
plt.fill_between(n_values, np.array(list(means_simple.values())) - np.array(list(std_devs_simple.values())), np.array(list(means_simple.values())) + np.array(list(std_devs_simple.values())), color='green', alpha=0.07, label='Simple Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_CRA_IR.values()), marker='o', label='CRA+IR Mean Steps', color='red')
plt.fill_between(n_values, np.array(list(means_CRA_IR.values())) - np.array(list(std_devs_CRA_IR.values())), np.array(list(means_CRA_IR.values())) + np.array(list(std_devs_CRA_IR.values())), color='red', alpha=0.07, label='CRA+IR ±1 Standard Deviation')

plt.xlabel('N value')
plt.ylabel('Steps required to obtain a solution')
plt.xticks(np.arange(min(n_values), max(n_values) + 1, 1))
plt.legend(loc='upper left',ncol=1)
plt.grid(True)
plt.ylim(bottom=0)
#plt.title('Comparison of Steps to Success for Different Encodings')
plt.show()
"""
# Plotting Steps to Success on a log scale
plt.figure(figsize=(10, 6))
plt.plot(n_values, list(means_num.values()), marker='o', label='Num Encoding Mean Steps', color='orange')
plt.fill_between(n_values, np.array(list(means_num.values())) - np.array(list(std_devs_num.values())), np.array(list(means_num.values())) + np.array(list(std_devs_num.values())), color='orange', alpha=0.07, label='Num Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_one_hot.values()), marker='o', label='One-Hot Encoding Mean Steps', color='blue')
plt.fill_between(n_values, np.array(list(means_one_hot.values())) - np.array(list(std_devs_one_hot.values())), np.array(list(means_one_hot.values())) + np.array(list(std_devs_one_hot.values())), color='blue', alpha=0.07, label='One-Hot Encoding ±1 Standard Deviation')

plt.plot(n_values, list(means_simple.values()), marker='o', label='Simple Encoding Mean Steps', color='grey')
plt.fill_between(n_values, np.array(list(means_simple.values())) - np.array(list(std_devs_simple.values())), np.array(list(means_simple.values())) + np.array(list(std_devs_simple.values())), color='grey', alpha=0.07, label='Simple Encoding ±1 Standard Deviation')



# Plot CRA results
plt.plot(n_values, list(means_CRA.values()), marker='o', label='CQL', color='red')
plt.fill_between(n_values, np.array(list(means_CRA.values())) - np.array(list(std_devs_CRA.values())), np.array(list(means_CRA.values())) + np.array(list(std_devs_CRA.values())), color='red', alpha=0.07, label='CQL ±1 Standard Deviation')

plt.xlabel('N value')
plt.ylabel('Steps required to obtain a solution (log scale)')
plt.yscale('log')
plt.xticks(np.arange(min(n_values), max(n_values) + 1, 1))
plt.legend(loc='lower right',fontsize=6)
plt.grid(True)
plt.ylim(bottom=1000)
#plt.title('Comparison of Steps to Success for Different Encodings and MDP/CRA')
plt.show()"""
