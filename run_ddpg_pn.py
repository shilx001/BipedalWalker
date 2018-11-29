# normal DDPG with parameter noise

import numpy as np
import gym
from ddpg_pn import DDPG
import matplotlib.pyplot as plt
import pickle

env = gym.make('BipedalWalker-v2')

agent = DDPG(a_dim=4, s_dim=24, a_bound=1)

total_reward = []
for episode in range(10000):
    state = env.reset()
    cum_reward = 0
    for step in range(1600):
        action = np.reshape(agent.choose_action(state), [4, ])
        next_state, reward, done, _ = env.step(action)
        # print(action)
        cum_reward += reward
        agent.store_transition(state, action, reward, next_state)
        state = next_state
        agent.learn()
        if done:
            print('Episode', episode, ' Complete at reward ', cum_reward, '!!!')
            break
        if step == 1600 - 1:
            print('Episode', episode, ' finished at reward ', cum_reward)
    total_reward.append(cum_reward)

plt.plot(total_reward)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.title('MountainCar continuous')
plt.savefig('ddpg_pn')
pickle.dump(total_reward, open('DDPG_pn', 'wb'))
