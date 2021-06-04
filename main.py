from unityagents import UnityEnvironment
import numpy as np
import torch
import random
from model import QNetwork
from Replay_buffer import ReplayBuffer
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import matplotlib.pyplot as plt


#intialize Paramters, buffer and Network

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the local network
C = 10000               # how often to update the target Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#intialize unity Enviroment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#Q-Network
state_size = brain.vector_observation_space_size
action_size = brain.vector_action_space_size
qnetwork_local = QNetwork(state_size, action_size, seed=0).to(device)
qnetwork_target = QNetwork(state_size, action_size, seed=0).to(device)
optimizer = optim.Adam(qnetwork_local.parameters(), lr=LR)

# Replay memory
memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=0)
# Initialize time step (for updating every UPDATE_EVERY steps)
global t_step
t_step = 0

#intialize time step for updating target qnetwork
global q_step
q_step =0

#action function
def act(state, eps=.0):
    """Returns actions for given state as per current policy.

    Params
    ======
        state (array_like): current state
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    qnetwork_local.eval()
    with torch.no_grad():
        action_values = qnetwork_local(state)
    qnetwork_local.train()
    # Epsilon-greedy action selection
    if random.random() > eps:
        return np.int32(np.argmax(action_values.cpu().data.numpy()))
    else:
        return np.int32(random.choice(np.arange(action_size)))


def step(state, action, reward, next_state, done):
    # Save experience in replay memory
    memory.add(state, action, reward, next_state, done)

    # Learn every UPDATE_EVERY time steps.
    global t_step
    t_step = (t_step + 1) % UPDATE_EVERY
    if t_step == 0:
        # If enough samples are available in memory, get random subset and learn
        if len(memory) > BATCH_SIZE:
            experiences = memory.sample()
            learn(experiences, GAMMA)


def learn(experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    optimizer.zero_grad()

    action_values = qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    y = rewards + (gamma * action_values * (1 - dones))
    local_action_values = qnetwork_local(states)
    labels = local_action_values.gather(1, actions)

    loss = F.mse_loss(labels, y)
    loss.backward()
    optimizer.step()

    global q_step
    q_step += 1
    if q_step % C == 0:
        for target_param, local_param in zip(qnetwork_target.parameters(), qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)

# reset the environment
def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]
            step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps*eps_decay, eps_end)
        scores_window.append(score)
        scores.append(score)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(1) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
            torch.save(qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

env.close()

# plot the scores
fig = plt.figure()
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()