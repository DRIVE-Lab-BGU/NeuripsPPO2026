import random
import numpy as np
import matplotlib.pyplot as plt


def evaluate(agent, Env, episodes=1, verbose=False, seed=42):
    returns = []
    for eps in range(episodes):
        total_reward = 0
        state, _ = Env.reset(seed=seed+eps)
        for step in range(Env.horizon):
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = Env.step(action)
            done = terminated or truncated
            total_reward += reward
            if verbose:
                print()
                print(f'step       = {step}')
                print(f'state      = {state}')
                print(f'action     = {action}')
                print(f'next state = {next_state}')
                print(f'reward     = {reward}')
            state = next_state
            if done:
                break
        if verbose:
            print(f'episode ended with reward {total_reward}')
        returns.append(total_reward)
    return {'mean': np.mean(returns), 'std': np.std(returns)}


def displayResults(returns):
    means = []
    stds = []
    x = np.arange(len(returns))
    for i in range(len(returns)):
        means.append(returns[i]['mean'])
        stds.append(returns[i]['std'])
    means = np.array(means)
    stds = np.array(stds)
    upper_bound = means + stds
    lower_bound = means - stds

    plt.figure(figsize=(8, 5))
    plt.plot(x, means, color='b', marker='o', label="Mean")  # Mean curve
    plt.fill_between(x, lower_bound, upper_bound, color='b', alpha=0.2, label="±1 Std Dev")  # Shaded std envelope

    # Add labels and title
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("results")
    plt.legend()

    # Show the figure
    plt.show()
