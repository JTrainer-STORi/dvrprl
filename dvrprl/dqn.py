import numpy as np
from tqdm import tqdm

import gymnasium as gym
import torch
import copy
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    """
    A replay memory class that stores the transitions that the agent observes during deep Q-learning

    """

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.memory_counter = 0
        self.state_memory = [None for _ in range(capacity)]
        self.action_memory = [None for _ in range(capacity)]
        self.reward_memory = [None for _ in range(capacity)]
        self.next_state_memory = [None for _ in range(capacity)]
        self.terminal_memory = [None for _ in range(capacity)]
        self.rng = np.random.default_rng(seed)

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory."""
        index = self.memory_counter % self.capacity
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        """
        Randomly sample batch_size transitions from the replay memory. Transitions are sampled without replacement.
        """
        max_memory = min(self.memory_counter, self.capacity)
        batch = self.rng.choice(max_memory, batch_size, replace=False)
        transitions = [None for _ in range(batch_size)]

        for i, j in enumerate(batch):
            transitions[i] = (
                self.state_memory[j],
                self.action_memory[j],
                self.reward_memory[j],
                self.next_state_memory[j],
                self.terminal_memory[j],
            )

        padded_transitions = (
            torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(t[0]).to(torch.float32) for t in transitions], batch_first=True, padding_value=-1
            ),
            torch.tensor([t[1] for t in transitions]).to(torch.int32),
            torch.tensor([t[2] for t in transitions]).to(torch.float32),
            torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t[3]).to(torch.float32) for t in transitions], batch_first=True, padding_value=-1
            ),
            torch.tensor([t[4] for t in transitions]).to(torch.bool),
        )

        return padded_transitions
    
    def __len__(self):
        """Return the number of transitions stored in the replay memory."""
        return min(self.memory_counter, self.capacity)
    
class DQNAgent:
    def __init__(self, replay_memory_size: int = 50_000, value_network: torch.nn.Module = None, seed: int = 0) -> None:
        
        # Main model
        self.model = value_network

        # Target model
        self.target_model = copy.deepcopy(self.model) # get a new instance
        self.target_model.load_state_dict(self.model.state_dict()) # copy weights and stuff

        self.replay_memory = ReplayMemory(replay_memory_size, seed=seed)
        self.target_update_counter = 0
        self.default_rng = np.random.default_rng(seed)

    def update_replay_memory(self, transition):
        """Add a transition to the replay memory."""
        self.replay_memory.store_transition(*transition)
   
    def train(self, env: gym.Env, episodes: int = 20_000, min_replay_memory_size: int =1_000, update_target_every: int = 5, batch_size: int = 64, discount: float = 0.99, epsilon: float = 1, epsilon_min: float = 0.001, learning_rate: float = 1e-3, logdir=None, save_freq: int = 1000):

        epsilon_decay = (epsilon - epsilon_min) / episodes

        tb_writer = SummaryWriter(logdir) if logdir is not None else None
        history = {"episode_rewards": np.zeros(episodes), "epsilons": np.zeros(episodes), "episode_steps": np.zeros(episodes)}

        for episode in tqdm(range(episodes), ascii=True, unit="episode"):
            # Reset environment and get initial state
            current_state = env.reset()
            done = False
            episode_reward = 0
            step = 1

            while not done:
                # Decide action
                if self.default_rng.random() > epsilon:
                    with torch.no_grad():
                        action = torch.argmax(self.model(torch.from_numpy(current_state).to(torch.float32).unsqueeze(0))).item()
                else:
                    action = self.default_rng.integers(0, current_state.shape[0])

                # Perform action
                next_state, reward, done, _ = env.step(action)

                episode_reward += reward
                step += 1

                # Update replay memory
                loss = self.update_replay_memory((current_state, action, reward, next_state, done))

                self.update_networks(min_replay_memory_size=min_replay_memory_size, update_target_every=update_target_every, batch_size=batch_size, discount=discount, learning_rate=learning_rate)

                current_state = next_state

            # Update target network counter every episode
            self.target_update_counter += 1

            history["episode_rewards"][episode] = episode_reward
            history["epsilons"][episode] = epsilon
            history["episode_steps"][episode] = step

            # Update logs
            if logdir is not None and (episode+1) % save_freq == 0:
                self.save_value_weights(f'{logdir}/value_weights_{i+1}.pth')
            if tb_writer is not None:
                tb_writer.add_scalar("episode_reward", episode_reward, episode)
                tb_writer.add_scalar("epsilon", epsilon, episode)
                tb_writer.add_scalar("episode_steps", step, episode)
                tb_writer.flush()

            if epsilon > epsilon_min:
                epsilon -= epsilon_decay
                epsilon = max(epsilon_min, epsilon)

        return history

    def update_networks(self, min_replay_memory_size, batch_size, discount, update_target_every, learning_rate,):
        
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < min_replay_memory_size:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = self.replay_memory.sample(batch_size)

        # Get current states from minibatch, then query NN model for Q values
        with torch.no_grad():
            current_qs_list = self.model(minibatch[0])

        # Get future states from minibatch, then query NN target model for Q values
        with torch.no_grad():
            future_qs_list = self.target_model(minibatch[3])
        
        X = []
        y = []

        # Now we need to enumerate our batches

        for index in range(batch_size):
            current_state = minibatch[0][index]
            action = minibatch[1][index]
            reward = minibatch[2][index]
            next_state = minibatch[3][index]
            done = minibatch[4][index]

            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward
            
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


        # Fit on all samples as one batch, log only on terminal state
        for X, y in zip(X, y):
            optimizer.zero_grad()
            y_pred = self.model(X.unsqueeze(0))
            loss = loss_fn(y_pred, y.unsqueeze(0))
            loss.backward()
            optimizer.step()

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > update_target_every:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        return loss.item()
    
    def load_value_weights(self, path: str):
        """
        Load the weights of the value network from the given path.

        Args:
            path (str): Path to the weights.
        """
        if self.value_network is not None:
            self.model.load_state_dict(torch.load(path))
    
    def save_value_weights(self, path: str):
        """
        Save the weights of the value network to the given path.

        Args:
            path (str): Path to save the weights.
        """

        torch.save(self.model.state_dict(), path)