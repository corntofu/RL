class REINFORCE(nn.Module):
    def __init__(self, input_size, output_size):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

model = REINFORCE(4, 2).to(device)
opt = optim.Adam(model.parameters(), lr = 1e-3)
epoch = 500
gamma = 0.99

for e in range(1, epoch + 1):
    state, _ = env.reset()
    episode_log_probs = []
    episode_rewards = []
    total_episode_reward = 0

    while True:
        action, log_prob = model.select_action(state)
        episode_log_probs.append(log_prob)
        state, reward, done, _, _ = env.step(action)
        episode_rewards.append(reward)
        total_episode_reward += reward
        if done:
            break

    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    policy_loss = []
    for log_prob, G_t in zip(episode_log_probs, returns):
        policy_loss.append(-log_prob * G_t)

    opt.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    opt.step()

    if e % 10 == 0:
        print(f'Epoch: {e}\tReward: {total_episode_reward}')
