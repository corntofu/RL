from torch.nn.modules.dropout import AlphaDropout

# Abstract Class for buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity

    def push(self, *args):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

# Original Replay Buffer(PER's alpha = 0)
class OriginalReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# fenwick tree for efficient calculation, not segment tree
# bit operation & non-recursive is fast!
class Tree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(capacity + 1, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _update_bit_internal(self, idx_1_based, val_change):
        val_change = np.array(val_change)
        if val_change.size > 1:
            val_change = val_change.item()
        while idx_1_based <= self.capacity:
            self.tree[idx_1_based] += val_change
            idx_1_based += idx_1_based & (-idx_1_based)

    def _query_bit_internal(self, idx_1_based):
        s = 0
        while idx_1_based > 0:
            s += self.tree[idx_1_based]
            idx_1_based -= idx_1_based & (-idx_1_based)
        return s

    def total(self):
        return self._query_bit_internal(self.capacity)

    def update(self, data_idx_0_based, new_priority):
        if not (0 <= data_idx_0_based < self.capacity):
            raise IndexError(f"data_idx_0_based ({data_idx_0_based}) out of bounds [0, {self.capacity-1}]")

        if isinstance(new_priority, np.ndarray):
            new_priority = new_priority.item()

        old_priority = self.priorities[data_idx_0_based]
        change = new_priority - old_priority
        if isinstance(change, np.ndarray):
            change = change.item()
        self.priorities[data_idx_0_based] = new_priority
        self._update_bit_internal(data_idx_0_based + 1, change)

    def retrieve(self, s):
        low = 0
        high = self.capacity - 1
        ans_data_idx = -1

        while low <= high:
            mid = (low + high) // 2
            current_prefix_sum = self._query_bit_internal(mid + 1)
            if current_prefix_sum >= s:
                ans_data_idx = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans_data_idx

    def get(self, s):

        data_idx = self.retrieve(s)
        if data_idx == -1 or data_idx >= self.n_entries:
            if self.n_entries == 0:
                raise ValueError("Cannot sample from an empty FenwickTree buffer.")
            pass

        return data_idx, self.priorities[data_idx], self.data[data_idx]

    def add(self, data_item, initial_priority):
        current_data_idx = self.write
        self.data[current_data_idx] = data_item
        self.update(current_data_idx, initial_priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def __len__(self):
        return self.n_entries

class PriortizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_inc=1e-4, eps=1e-6):
        super().__init__(capacity)
        self.tree = Tree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.eps = eps

        # max priority used when inserting a new sample (so new samples get high chance)
        self.max_priority = 1.0

    def push(self, *args):
        # Store Transition object in the tree with priority = max_priority ** alpha
        priority = (self.max_priority if self.max_priority is not None else 1.0) ** self.alpha
        self.tree.add(Transition(*args), priority)

    def sample(self, batch_size, max_retries_per_pick=3):
        total = self.tree.total()
        N = self.tree.n_entries

        if N == 0 or total <= 0:
            raise ValueError("Cannot sample from an empty PrioritizedReplayBuffer")

        actual_batch = min(batch_size, N)
        segment = total / actual_batch

        self.beta = min(1.0, self.beta + self.beta_inc)

        batch = []
        idxs = []
        priorities = []

        for i in range(actual_batch):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = min(max(s, 1e-8), total)

            pick_attempts = 0
            while pick_attempts < max_retries_per_pick:
                try:
                    idx, p, data = self.tree.get(s)
                except Exception:
                    s = random.uniform(1e-8, total)
                    pick_attempts += 1
                    continue

                if data is None:
                    s = random.uniform(1e-8, total)
                    pick_attempts += 1
                    continue

                batch.append(data)
                idxs.append(idx)
                priorities.append(p)
                break

            else:
                fallback_idx = random.randrange(0, N)
                batch.append(self.tree.data[fallback_idx])
                idxs.append(fallback_idx)
                priorities.append(self.tree.priorities[fallback_idx])

        priorities = np.array(priorities, dtype=np.float64)
        sampling_probabilities = priorities / total
        is_weights = np.power(N * sampling_probabilities + 1e-12, -self.beta)
        is_weights = is_weights / (is_weights.max() + 1e-12)

        return batch, idxs, is_weights

    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.eps) ** self.alpha
            if not (0 <= idx < self.tree.capacity):
                continue
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.tree)
