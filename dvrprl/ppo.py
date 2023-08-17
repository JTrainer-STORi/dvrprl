import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

torch.autograd.set_detect_anomaly(True)

def discount_rewards(rewards: list[float], gam: float) -> list[float]:
    """
    Helper function that returns dicounted rewards-to-go from a list of rewards

    Args:
        rewards (list[float]): List of rewards
        gam (float): Discount factor

    Returns:
        list[float]: List of discounted rewards-to-go
    """
    cumulative_reward = 0
    discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
    for i in reversed(range(len(rewards))):
        cumulative_reward = cumulative_reward * gam + rewards[i]
        discounted_rewards[i] = cumulative_reward

    return discounted_rewards


def compute_advantages(
    rewards: list[float], values: list[float], gam: float, lam: float
) -> list[float]:
    """
    Helper function that returns generalized advantage estimates from a list
    of rewards and values

    Args:
        rewards (list[float]): List of rewards
        values (list[float]): List of values
        gam (float): Discount factor
        lam (float): GAE-Lambda parameter

    Returns:
        list[float]: List of generalized advantage estimates
    """
    rewards = np.array(rewards, dtype=np.float64)
    values = np.array(values, dtype=np.float64)
    delta = rewards - values
    delta[:-1] += gam * values[1:]
    return discount_rewards(delta, gam * lam)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    Args:
        gam (float, optional): Discount factor. Defaults to 0.99.
        lam (float, optional): GAE-Lambda parameter. Defaults to 0.95.
    """

    def __init__(self, gam: float = 0.99, lam: float = 0.95) -> None:
        self.obs = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.gam, self.lam = gam, lam
        self.ptr, self.path_start_idx = 0, 0

    def store(
        self, obs: np.ndarray, action: int, reward: float, logprob: float, value: float
    ) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.ptr += 1

    def finish(self) -> None:
        """
        Finish an episode and compute advantages and discounted rewards-to-go.

        Advantages are stored in place of 'values' and discounted rewards-to-go
        are stored in place of 'rewards' for the current trajectory.
        """
        tau = slice(self.path_start_idx, self.ptr)
        rewards = discount_rewards(self.rewards[tau], self.gam)
        values = compute_advantages(
            self.rewards[tau], self.values[tau], self.gam, self.lam
        )
        self.rewards[tau] = rewards
        self.values[tau] = values
        self.path_start_idx = self.ptr

    def clear(self) -> None:
        """
        Reset the buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.ptr, self.path_start_idx = 0, 0

    def get(
        self,
        batch_size: int = None,
        normalize_advantages: bool = True,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Return a dataset of training data from this buffer
        """
        actions = np.array(self.actions[: self.path_start_idx], dtype=np.int32)
        logprobs = np.array(self.logprobs[: self.path_start_idx], dtype=np.float32)
        advantages = np.array(self.values[: self.path_start_idx], dtype=np.float32)
        values = np.array(self.rewards[: self.path_start_idx], dtype=np.float32)

        if normalize_advantages:
            advantages -= np.mean(advantages)
            advantages /= np.std(advantages) + 1e-8

        # if self.states and self.states[0].ndim == 2:
        # Filter out states with only one action available
        indices = [
            i
            for i in range(len(self.states[: self.path_start_idx]))
            if self.states[i].shape[0] != 1
        ]
        states = [
            torch.from_numpy(self.states[i]).to(torch.float32) for i in indices
        ]
        actions = actions[indices]
        logprobs = logprobs[indices]
        advantages = advantages[indices]
        values = values[indices]

        # Pad states to the same length
        padded_states = torch.nn.utils.rnn.pad_sequence(
            states, batch_first=True, padding_value=-1
        )

        dataset = torch.utils.data.TensorDataset(
            padded_states,
            torch.from_numpy(actions),
            torch.from_numpy(logprobs),
            torch.from_numpy(advantages),
            torch.from_numpy(values),
        )

        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
        )

    def __len__(self) -> int:
        return len(self.states)


class Agent:
    """
    Abstract base class for policy gradient agents

    """

    def __init__(
        self,
        policy_network: torch.nn.Module,
        policy_lr: float = 3e-4,
        policy_updates: int = 1,
        value_lr: float = 1e-3,
        value_network: torch.nn.Module = None,
        value_updates: int = 25,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_advantages: bool = True,
        kld_limit: float = 1e-2,
        ent_bonus: float = 0.0,
    ) -> None:
        """
        Args:
            env (gym.Env): Environment to interact with.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lam (float, optional): GAE-Lambda parameter. Defaults to 0.95.
            policy_lr (float, optional): Learning rate for the policy. Defaults to 3e-4.
            value_lr (float, optional): Learning rate for the value function. Defaults to 1e-3.
            device (str, optional): Device to run the agent on. Defaults to "cpu".
        """
        self.policy_network = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = torch.optim.Adam(
            params=self.policy_network.parameters(), lr=policy_lr
        )
        self.policy_updates = policy_updates

        self.value_network = value_network
        self.value_loss = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(
            params=self.value_network.parameters(), lr=value_lr
        )
        self.value_updates = value_updates

        self.gamma = gamma
        self.lam = lam
        self.buffer = PPOBuffer(gam=gamma, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.kld_limit = kld_limit
        self.ent_bonus = ent_bonus

    @torch.no_grad()
    def get_action(
        self, obs: np.ndarray, return_logprob: bool = False
    ) -> tuple[int, float]:
        """
        Get an action from the policy given an observation.

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            int: Action to take.
            float: Log probability of the action.
            float: Value of the state.
        """
        logpi = torch.squeeze(
            self.policy_network(torch.from_numpy(obs).to(torch.float32).unsqueeze(0))
        )
        action = torch.distributions.Categorical(logits=logpi).sample()

        if return_logprob:
            return action.item(), logpi[action].item()
        else:
            return action.item()

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """
        Return the predicted value for the given obs/state using the value model

        Args:
            obs (np.ndarray): Observation from the environment.

        Returns:
            float: Value of the state.
        """

        return (
            self.value_network(torch.from_numpy(obs).to(torch.float32).unsqueeze(0))
            .squeeze()
            .item()
        )

    def train(
        self,
        env: gym.Env,
        episodes: int = 100,
        epochs: int = 1,
        verbose: int = 0,
        save_freq: int = 100,
        logdir: str = None,
        batch_size: int = 64,
    ) -> dict:
        tb_writer = SummaryWriter(logdir) if logdir is not None else None
        history = {
            "mean_returns": np.zeros(epochs),
            "min_returns": np.zeros(epochs),
            "max_returns": np.zeros(epochs),
            "std_returns": np.zeros(epochs),
            "mean_ep_lens": np.zeros(epochs),
            "min_ep_lens": np.zeros(epochs),
            "max_ep_lens": np.zeros(epochs),
            "std_ep_lens": np.zeros(epochs),
            "policy_updates": np.zeros(epochs),
            "delta_policy_loss": np.zeros(epochs),
            "policy_ent": np.zeros(epochs),
            "policy_kld": np.zeros(epochs),
            "mean_mse": np.zeros(epochs),
        }

        for i in tqdm(range(epochs), unit="epoch", position=0, leave=True):
            self.buffer.clear()
            return_history = self.run_episodes(env, episodes, buffer=self.buffer)
            dataloader = self.buffer.get(
                batch_size=batch_size, normalize_advantages=self.normalize_advantages
            )
            policy_history = self._fit_policy_model(
                dataloader, epochs=self.policy_updates
            )
            if not self.value_network is None:
                value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
                history["mean_mse"][i] = np.mean(value_history["loss"])

            history["mean_returns"][i] = np.mean(return_history["returns"])
            history["min_returns"][i] = np.min(return_history["returns"])
            history["max_returns"][i] = np.max(return_history["returns"])
            history["std_returns"][i] = np.std(return_history["returns"])

            history["mean_ep_lens"][i] = np.mean(return_history["lengths"])
            history["min_ep_lens"][i] = np.min(return_history["lengths"])
            history["max_ep_lens"][i] = np.max(return_history["lengths"])
            history["std_ep_lens"][i] = np.std(return_history["lengths"])

            history["policy_updates"][i] = len(policy_history["loss"])
            history["delta_policy_loss"][i] = (
                policy_history["loss"][-1] - policy_history["loss"][0]
            )
            history["policy_ent"][i] = policy_history["ent"][-1]
            history["policy_kld"][i] = policy_history["kld"][-1]

            if logdir is not None and (i + 1) % save_freq == 0:
                self.save_policy_weights(f"{logdir}/policy_weights_{i+1}.pth")
                self.save_value_weights(f"{logdir}/value_weights_{i+1}.pth")
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "returns/mean_returns", history["mean_returns"][i], i
                )
                tb_writer.add_scalar(
                    "returns/min_returns", history["min_returns"][i], i
                )
                tb_writer.add_scalar(
                    "returns/max_returns", history["max_returns"][i], i
                )
                tb_writer.add_scalar(
                    "returns/std_returns", history["std_returns"][i], i
                )

                tb_writer.add_scalar(
                    "episode_lengths/mean_ep_lens", history["mean_ep_lens"][i], i
                )
                tb_writer.add_scalar(
                    "episode_lengths/min_ep_lens", history["min_ep_lens"][i], i
                )
                tb_writer.add_scalar(
                    "episode_lengths/max_ep_lens", history["max_ep_lens"][i], i
                )
                tb_writer.add_scalar(
                    "episode_lengths/std_ep_lens", history["std_ep_lens"][i], i
                )

                tb_writer.add_scalar(
                    "policy/policy_updates", history["policy_updates"][i], i
                )
                tb_writer.add_scalar(
                    "policy/delta_policy_loss", history["delta_policy_loss"][i], i
                )
                tb_writer.add_scalar("policy/policy_ent", history["policy_ent"][i], i)
                tb_writer.add_scalar("policy/policy_kld", history["policy_kld"][i], i)

                if not self.value_network is None:
                    tb_writer.add_scalar("value/mse", history["mean_mse"][i], i)
                    tb_writer.add_histogram("value/mse_distribution", value_history["loss"], i)

                tb_writer.add_histogram("returns/returns", return_history["returns"], i)
                tb_writer.add_histogram(
                    "episode_lengths/ep_lens", return_history["lengths"], i
                )
                tb_writer.flush()

        return history

    def run_episode(
        self, env: gym.Env, buffer: PPOBuffer = None, render: bool = False
    ) -> tuple[float, int]:
        """
        Run an episode in the environment and store the transitions in the buffer

        Args:
            env (gym.env): Environment to interact with.
            buffer (PPOBuffer, optional): Buffer to store the transitions in. Defaults to None.

        Returns:
            int: Length of the episode.
            float: Total reward of the episode.
        """

        obs, _ = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            action, logprob = self.get_action(obs, return_logprob=True)
            if self.value_network is None:
                value = 0.0
            else:
                value = self.get_value(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if render:
                env.render()
            episode_length += 1
            total_reward += reward
            if buffer is not None:
                buffer.store(obs, action, reward, logprob, value)
            obs = next_obs
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length

    def run_episodes(
        self, env: gym.Env, episodes: int = 100, buffer: PPOBuffer = None
    ) -> dict:
        """
        Run multiple episodes in the environment and store the transitions in the buffer

        Args:
            env (gym.env): Environment to interact with.
            episodes (int, optional): Number of episodes to run. Defaults to 100.
            buffer (PPOBuffer, optional): Buffer to store the transitions in. Defaults to None.

        Returns:
            int: Length of the episode.
            float: Total reward of the episode.
        """

        history = {"returns": np.zeros(episodes), "lengths": np.zeros(episodes)}
        for i in range(episodes):
            R, L = self.run_episode(env, buffer=buffer)
            history["returns"][i] = R
            history["lengths"][i] = L
        return history

    def _fit_policy_model(self, dataloader, epochs=1) -> dict:
        """
        Fit the policy model on the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to train on.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
        """

        history = {"loss": [], "kld": [], "ent": []}
        for i in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            for states, actions, logprobs, advantages, _ in dataloader:
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(
                    states, actions, logprobs, advantages
                )
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1
            history["loss"].append(loss / batches)
            history["kld"].append(kld / batches)
            history["ent"].append(ent / batches)
            if self.kld_limit is not None and kld > self.kld_limit:
                break
        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_model_step(
        self, states, actions, logprobs, advantages
    ) -> tuple[float, float, float]:
        """
        Perform a single training step on the policy model.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            logprobs (torch.Tensor): Batch of log probabilities of the actions.
            advantages (torch.Tensor): Batch of advantages.

        Returns:
            float: Loss of the batch.
            float: KL divergence of the batch.
            float: Entropy of the batch.
        """

        self.policy_optimizer.zero_grad()
        logpi = self.policy_network(states)
        new_logprobs = torch.sum(torch.nn.functional.one_hot(actions.type(torch.LongTensor), num_classes=logpi.shape[-1]) * logpi, dim=-1)
        #print(torch.nn.functional.one_hot(actions.type(torch.LongTensor), num_classes=logpi.shape[-1]) * logpi)
        #print(new_logprobs)
        #print(logprobs)
        #print(torch.exp(new_logprobs - logprobs))
        loss = self.policy_loss(new_logprobs, logprobs, advantages)
        loss.mean().backward()
        self.policy_optimizer.step()
        kld = (logprobs - new_logprobs).mean()
        ent = -new_logprobs.mean()
        return loss.mean().item(), kld.item(), ent.item()

    def load_policy_weights(self, path: str) -> None:
        """
        Load the weights of the policy network from the given path.

        Args:
            path (str): Path to the weights.
        """

        self.policy_network.load_state_dict(torch.load(path))

    def save_policy_weights(self, path: str) -> None:
        """
        Save the weights of the policy network to the given path.

        Args:
            path (str): Path to save the weights.
        """

        torch.save(self.policy_network.state_dict(), path)

    def _fit_value_model(self, dataloader, epochs=1) -> dict:
        """
        fit value model using data from dataset
        """
        if self.value_network is None:
            epochs = 0
        history = {"loss": []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for states, _, _, _, returns in dataloader:
                batch_loss = self._fit_value_model_step(states, returns)
                loss += batch_loss
                batches += 1
            history["loss"].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    def _fit_value_model_step(self, states, returns) -> float:
        """
        perform a single training step on the value model
        """
        self.value_optimizer.zero_grad()
        values = torch.squeeze(self.value_network(states), dim=1)
        loss = self.value_loss(values, returns)
        loss.backward()
        self.value_optimizer.step()
        return loss.item()

    def load_value_weights(self, path: str) -> None:
        """
        Load the weights of the value network from the given path.

        Args:
            path (str): Path to the weights.
        """
        if self.value_model is not None:
            self.value_network.load_state_dict(torch.load(path))

    def save_value_weights(self, path: str) -> None:
        """
        Save the weights of the value network to the given path.

        Args:
            path (str): Path to save the weights.
        """

        torch.save(self.value_network.state_dict(), path)


def pg_surrogate_loss(logprobs, old_logprobs, advantages) -> float:
    return -logprobs * advantages


def ppo_surrogate_loss(method="clip", eps=0.2, c=0.01):
    if method == "clip":

        def loss_fn(logprobs, old_logprobs, advantages):
            min_advantages = torch.where(
                advantages > 0, (1 + eps) * advantages, (1 - eps) * advantages
            )
            return -torch.minimum(
                torch.exp(logprobs - old_logprobs) * advantages, min_advantages
            )

        return loss_fn
    elif method == "penalty":

        def loss_fn(logprobs, old_logprobs, advantages):
            return -(
                torch.exp(logprobs - old_logprobs) * advantages
                - c * (logprobs - old_logprobs)
            )

        return loss_fn


class PGAgent(Agent):
    def __init__(self, policy_network: torch.nn.Module, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


class PPOAgent(Agent):
    def __init__(
        self,
        policy_network: torch.nn.Module,
        method: str = "clip",
        eps: float = 0.2,
        c: float = 0.01,
        **kwargs,
    ):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = ppo_surrogate_loss(method, eps)
