import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, input_dim, output_dim, env):
        self.policy = PolicyNetwork(input_dim, output_dim).to(device)
        self.value = ValueNetwork(input_dim).to(device)
        self.env = env
        
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=0.005)

    def select_action(self, state):
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def train(self, n_episodes=500, gamma=0.99):
        loss_graph = []
        reward_graph = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            
            done = False
            total_reward = 0

            while not done:
  
                action, log_prob = self.select_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
                
                curr_value = self.value(state)
                next_value = self.value(next_state_t)
                
                if done:
                    target_value = reward_t
                else:
                    target_value = reward_t + gamma * next_value
                
                # Delta: TD-Error
                delta = target_value - curr_value

                # (target - V(s))^2 -> V(s) * (target - V(s)) = V(s) * delta
                loss_value = F.mse_loss(curr_value, target_value.detach())
                
                self.optimizer_value.zero_grad()
                loss_value.backward()
                self.optimizer_value.step()

                
                # log_prob -> autograd
                loss_policy = -log_prob * delta.detach()
                
                self.optimizer_policy.zero_grad()
                loss_policy.backward()
                self.optimizer_policy.step()

                state = next_state_t

            reward_graph.append(total_reward)
            
            if episode % 20 == 0:
                print(f"Episode {episode}, Reward: {total_reward}")

        return reward_graph
