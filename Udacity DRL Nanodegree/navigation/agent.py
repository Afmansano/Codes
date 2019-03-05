import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


class ReplayMemory:

    def __init__(self, maxlen, batch_size, device, seed=32):
        '''
        It implements experience replay memory

        Params
        ======
        maxlen (int): maxmium memory size
        batch_size (int): size of selected memory to replay

        '''
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0
        self.device = device

        
    def add(self, state, action, reward, next_state, done):
        '''
        add the event to the memory buffer
        '''
        self.buf[self.index] = [state, action, reward, next_state, done]
        self.length = min(self.length + 1, self.maxlen) 
        self.index = (self.index + 1) % self.maxlen #replace the oldest when it is full
        
    
    def sample(self, with_replacement=True):    
        '''
        randomly sample events from memory
        '''
        def to_torch(x):
            return torch.from_numpy(np.vstack(x))
        def to_torch_uint8(x):
            return torch.from_numpy(np.vstack(x).astype(np.uint8))
        
        if with_replacement:
            indices = np.random.randint(self.length, size=self.batch_size)
        else:
            indices = np.random.permutation(self.length)[:self.batch_size]
        
        experiences = [[], [], [], [], []] # state, action, reward, next_state, continue
        for memory in indices:
            for e, value in zip(experiences, self.buf[memory]):
                e.append(value)
        experiences = [np.array(experience) for experience in experiences]
        
        states = to_torch([s for s in experiences[0]]).float()
        actions = to_torch([a for a in experiences[1]]).long()
        rewards = to_torch([r for r in experiences[2]]).float()
        next_states = to_torch([n for n in experiences[3]]).float()
        dones = to_torch_uint8([d for d in experiences[4]]).float()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        return (states, actions, next_states, rewards, dones)



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=42, hidden_layers=[32, 8]):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
            
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # detect GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE, self.device, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if self.memory.length > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, next_states, rewards, dones = experiences
        
        self.qnetwork_target.eval()
        with torch.no_grad():
            # get the max expected q-values 
            Q_expected = self.qnetwork_local(next_states) # gather = multiindex selector, dim=1 indices = actions
            action_argmax = torch.max(Q_expected, dim=1, keepdim=True)[1]
            Q_max_expected = Q_expected.gather(1, action_argmax)

            # get max predicted q-values for next states from target model (action with max value per state)
            # detach gets the tensor value, unsqueeze makes a matrix with one column
            Q_targets_next = self.qnetwork_target(next_states)
            # q-target for current state
            targets = rewards + gamma * Q_max_expected * (1-dones) #consider only not dones
        self.qnetwork_target.train()
        
        expected = self.qnetwork_local(states).gather(1, actions)
        loss = torch.sum((expected - targets)**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network 
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


    def train(self, env, brain_name, n_episodes=2000, timesteps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        '''
        train the model network applying experience replay
        Params
        ======
            agent (Agent): agent that interacts with the enviroment
            n_episodes (int): number of games played
            timesteps (int): max number os steps to be played in the game
            eps_start (floaßt): initial proportion os random actions on epsilon-greedy action
            eps_end (float): final proportion os random actions on epsilon-greedy action
            eps_decay (float): epsilon decay rate 
        '''
        scores = []
        last_scores = deque(maxlen=100)
        eps = eps_start
        for i_episode in range(n_episodes):
            env_status = env.reset(train_mode=True)[brain_name]
            state = env_status.vector_observations[0] #get state
            score = 0
            for _ in range(timesteps):
                action = self.act(state, eps).astype(int)
                env_status = env.step(action)[brain_name]
                next_state = env_status.vector_observations[0]
                reward = env_status.rewards[0]
                done = env_status.local_done[0]
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            last_scores.append(score)
            eps = max(eps_end, eps*eps_decay) #decreases epsilon
            print('\rEpisode {}\tScores mean: {:.2f}'.format(i_episode, np.mean(last_scores)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tLast 100 scores mean: {:.2f}'.format(i_episode, np.mean(last_scores)))
            if np.mean(last_scores)>= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tScores mean: {:.2f}'.format(i_episode-100, np.mean(last_scores)))
                torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        return scores 
        