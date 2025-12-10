import copy

class DQN:
    def __init__(self, network, env):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = network.to(self.device)
        self.target_net = copy.deepcopy(network).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.env = env

        self.optim = None
        self.buffer = None
        self.step_done = 0

    def select_action(self, state):
        # if noisy -> not epsilon-greedy!
        if self.noisy:
            with torch.no_grad():
                q_distributions = self.policy_net(state.unsqueeze(0))
                q_means = q_distributions.mean(dim=2)
                action = q_means.argmax(dim=1).unsqueeze(0)
                return action
        eps = self.eps_ed + (self.eps_st - self.eps_ed) * math.exp(-1 * self.step_done / self.eps_decay)
        self.step_done += 1

        if random.random() < eps:
            return torch.tensor([[self.env.action_space.sample()]], device = self.device, dtype = torch.long)

        else:
            with torch.no_grad():
                q_distributions = self.policy_net(state.unsqueeze(0))
                q_means = q_distributions.mean(dim=2)
                action = q_means.argmax(dim=1).unsqueeze(0)
                return action

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return

        isPER = isinstance(self.buffer, PriortizedReplayBuffer)
        idxs = is_weights = None

        if not isPER:
            transitions = self.buffer.sample(self.batch_size)
        else:
            transitions, idxs, is_weights = self.buffer.sample(self.batch_size)
            is_weights = torch.tensor(is_weights, dtype=torch.float, device=self.device)

        batch = Transition(*zip(*transitions))

        state_batch = torch.stack([torch.tensor(s, device=self.device, dtype=torch.float) for s in batch.state])
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        k_batch = torch.tensor(batch.k, device=self.device, dtype=torch.float)

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device, dtype=torch.bool
        )

        if non_final_mask.any():
            non_final_next_states = torch.stack([
                torch.tensor(s, device=self.device, dtype=torch.float)
                for s in batch.next_state if s is not None
            ])
        else:
            non_final_next_states = None

        dist = self.policy_net(state_batch)
        dist = dist.gather(
            1, action_batch.unsqueeze(-1).expand(-1, -1, N_ATOMS)
        ).squeeze(1)

        target_dist = torch.zeros_like(dist)

        with torch.no_grad():
            if non_final_next_states is not None:
                next_q_dist = self.target_net(non_final_next_states)

                next_online_dist = self.policy_net(non_final_next_states)
                next_actions = next_online_dist.mean(-1).argmax(1)

                next_selected_dist = next_q_dist[
                    range(len(next_actions)), next_actions
                ]

            target_dist[non_final_mask] = next_selected_dist

        gamma_k = (self.gamma ** k_batch).unsqueeze(1)
        target_dist = reward_batch.unsqueeze(1) + gamma_k * target_dist

        e_loss = quantile_huber_loss(dist, target_dist, self.device)

        loss = (e_loss * is_weights).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if isPER:
            td_error = e_loss.detach().cpu().numpy()
            self.buffer.update(idxs, td_error)

        return loss.item()

    # Soft update: update slowly each step, not batch at one time
    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, n_episodes=500, eps_st=0.9, eps_ed=0.1, eps_decay=2500,
          gamma=0.99, lr=3e-4, capacity=10000, batch_size=256, tau=0.005, buffer = OriginalReplayBuffer, n_step = 3, noisy = False, converge = 450):
        '''
        n_episodes: number of episodes
        eps_st: start epsilon for epsilon-greedy
        eps_ed: end epsilon for epsilon-greedy
        eps_decay: decay rate for epsilon
        gamma: discount factor
        lr: learning rate
        capacity: capacity of replay buffer
        tau: update rate for soft-update
        buffer: type of replay buffer(Original, PER, etc...)
        n_step: n-step return
        noisy: whether to use noisy net
        optim: optimizer
        '''
        self.n_episodes = n_episodes
        self.eps_st = eps_st
        self.eps_ed = eps_ed
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_step = n_step
        self.noisy = noisy

        self.buffer = buffer(capacity)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.step_done = 0

        reward_graph = []
        loss_graph = []
        real_graph = []

        state_deque = deque()
        action_deque = deque()
        next_state_deque = deque()
        reward_deque = deque()

        gamma_n_minus_1 = self.gamma ** (self.n_step - 1)

        for i_episode in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            running_reward = 0
            real_reward = 0
            done = False

            state_deque.clear()
            action_deque.clear()
            next_state_deque.clear()
            reward_deque.clear()

            idx = -1

            rw = 0.0

            while not done:
                if self.noisy:
                    self.policy_net.reset_noise()
                    self.target_net.reset_noise()

                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                state_deque.append(state.cpu().numpy())
                action_deque.append(action.item())
                reward_deque.append(reward)
                next_state_deque.append(None if done else next_state)

                idx += 1
                running_reward += reward
                real_reward += -1
                L = len(reward_deque)
                if L <= self.n_step:
                    rw += reward_deque[-1] * (self.gamma ** (L - 1))
                else:
                    r_old = reward_deque[-(self.n_step + 1)]
                    rw = (rw - r_old) / self.gamma + gamma_n_minus_1 * reward_deque[-1]

                if idx >= self.n_step - 1:
                    start_state = state_deque[-self.n_step]
                    start_action = action_deque[-self.n_step]
                    next_state_for_oldest = next_state_deque[-1]

                    actual_k = self.n_step
                    push_G = rw
                    for i in range(self.n_step):
                        if next_state_deque[-self.n_step + i] is None:
                            actual_k = i + 1
                            G_tmp = 0.0
                            for j in range(actual_k):
                                G_tmp += reward_deque[-self.n_step + j] * (self.gamma ** j)
                            push_G = G_tmp
                            next_state_for_oldest = None
                            break

                    self.buffer.push(start_state, start_action, next_state_for_oldest, push_G, actual_k)

                if done:
                    L_remain = len(reward_deque)
                    while len(reward_deque) > 0:
                        G_tmp = 0.0
                        for j, r in enumerate(list(reward_deque)[:self.n_step]):
                            G_tmp += r * (self.gamma ** j)
                        s0 = state_deque.popleft()
                        a0 = action_deque.popleft()
                        ns0 = next_state_deque.popleft()
                        reward_deque.popleft()
                        k0 = min(self.n_step, L_remain)
                        self.buffer.push(s0, a0, ns0, G_tmp, k0)
                        L_remain -= 1
                    state_deque.clear(); action_deque.clear(); next_state_deque.clear(); reward_deque.clear()

                state = torch.tensor(next_state, device=self.device, dtype=torch.float) if not done else None

                loss = self.optimize_model()
                if loss is not None:
                    loss_graph.append(loss)

                self.soft_update()

            real_graph.append(real_reward)
            reward_graph.append(running_reward)
            print(f"Episode {i_episode+1} Reward: {running_reward}, Real Reward: {real_reward}")
            if np.mean(real_graph[-100:]) >= converge:
                break

        return reward_graph, loss_graph
