# normal DDPG

import numpy as np
import gym
from ddpg import DDPG
import matplotlib.pyplot as plt
import pickle

env = gym.make('BipedalWalker-v2')

agent = DDPG(a_dim=4, s_dim=24, a_bound=1)

total_reward = []
for episode in range(1000):
    state = env.reset()
    var = 1
    cum_reward = 0
    terminate = False
    count = 0
    while not terminate:
        action = np.reshape(np.clip(agent.choose_action(state) + np.random.normal(0, var, size=[4, ]), -1, 1), [4, ])
        next_state, reward, done, _ = env.step(action)
        # print(action)
        cum_reward += reward
        bd = 0
        if done:
            bd = 1
        agent.store_transition(state, action, reward, next_state, bd)
        state = next_state
        count += 1
        agent.learn()
        if done:
            terminate = True
            print('Episode', episode, ' Complete at reward ', cum_reward, '!!!')
            break
    total_reward.append(cum_reward)
    if var > 0.1:
        var *= 0.995

plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('MountainCar continuous')
plt.savefig('ddpg')
pickle.dump(total_reward, open('DDPG'))
