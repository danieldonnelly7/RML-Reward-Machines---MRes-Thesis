import numpy as np

from .eval import eval_agent

import time 
import copy

def train_repeat(
    agent,
    env,
    n_episodes,
    n_repeats=1,
):
    rewards = []

    for _ in range(n_repeats):
        agent.reset_training()
        reward = train(
            agent,
            env,
            n_episodes,
        )
        rewards.append(reward)

    rewards = np.array(rewards)
    return np.mean(rewards, axis=0)


def train_till_conv_repeat(
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_2(
            agent,
            env,
            max_samples // 100,
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)

def train_till_conv_repeat_office(
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_3(
            agent,
            env,
            max_samples // 100,
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)


def train(
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []

    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(100):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)

            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            # if reward == 1:
            #     print("HERE")

            # if terminated:
            #     break
            if agent.terminated():
                # print(obs, next_obs, last_u, agent.u)
                break

        agent.decay_epsilon()
        rewards.append(reward)

        # if episode % 1000 == 0:
        #     pbar.set_description(
        #         f"Episode {episode} | Reward {np.mean(rewards[-1000:]):.2f}"
        #     )
    return np.array(rewards)


def train_till_conv(
    agent,
    env,
    max_samples,
):
    sample_counter = 0
    done = False

    while not done:
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(200):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)

            sample_counter += 1

            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            if agent.terminated():
                break
            if sample_counter == max_samples:
                break

        agent.decay_epsilon()
        done = eval_agent(agent, env) == 1 or sample_counter == max_samples
    return sample_counter


def train_conv_2(
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(100):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break

        agent.decay_epsilon()
        rewards.append(reward)

        if episode >= 20:
            ave = np.mean(rewards[-20:])
            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                break
    return sample_count

def train_conv_3(
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(500):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs


            sample_count += 1


            if agent.terminated() or terminated:
                break

        agent.decay_epsilon()
        rewards.append(reward)

        if rewards.count(1) > 0:
            # Succesful policy true when deterministic policy is succesful, false otherwise
            succesful_policy = eval_office_world(env, agent)

            if succesful_policy:
                break

    return sample_count

def eval_office_world(env,agent):
    agent.reset()
    obs, _ = env.reset()
    obs = env.observation(obs)
    success = False
    steps = 0
    for t in range(500):
        action, flag = agent.get_evaluation_action(obs)
        if flag:   # Breaks if the agent has 0 in q table for all possible next actions
            if steps == 0:
                reward = 0
            break

        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.observation(next_obs)
        agent.evaluation_update(next_obs)
        obs = next_obs
        
        steps += 1
        if agent.terminated() or terminated:
            break
    if reward == 1:
        success = True
    return success
