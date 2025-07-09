import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

from common import evaluate_returns, preprocess, plot_rewards, save_model, evaluate_agent, save_metrics
GPU_ID = sys.argv[1]  # 사용하고 싶은 GPU 번호
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

global_path = 'Model/Research/Research_5_QLearning/'
model_name = 'DoubleDQN'

# -------------------- 전역 상수 --------------------
LEVERAGE = 15  # 💡 선물 트레이딩용 레버리지 적용 계수
SEED = 2025       # 랜덤 시드 고정

# -------------------- 랜덤 시드 고정 --------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- 환경 클래스 정의 --------------------
class LongShortHedgeEnv:
    def __init__(self, features, returns, actions):
        self.features = features
        self.returns = returns
        self.actions = actions
        self.t = 0

    def reset(self):
        self.t = 0
        return self.features[self.t]

    def step(self, action_idx):
        nasdaq_w = self.actions[action_idx]
        sp500_w = -(1 - nasdaq_w)
        reward = (nasdaq_w * self.returns[self.t + 1, 0] + sp500_w * self.returns[self.t + 1, 1]) * LEVERAGE
        self.t += 1
        done = self.t >= len(self.features) - 1
        return self.features[self.t], reward, done

# -------------------- DQN 정의 --------------------
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=100_000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.batch_size = 64

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.model.net[-1].out_features)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s).to(device)
        s_ = torch.FloatTensor(s_).to(device)
        r = torch.FloatTensor(r).to(device)
        a = torch.LongTensor(a).to(device)
        d = torch.BoolTensor(d).to(device)

        q = self.model(s).gather(1, a.view(-1, 1)).squeeze()

        next_q_values = self.model(s_)
        next_actions = torch.argmax(next_q_values, dim=1)
        next_q_target = self.target_model(s_).gather(1, next_actions.unsqueeze(1)).squeeze().detach()

        q_target = r + self.gamma * next_q_target * (~d)

        loss = nn.MSELoss()(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())



# -------------------- 훈련 및 테스트 --------------------
def train_and_test(df, max_episodes=100):
    actions = np.round(np.arange(0.1, 0.91, 0.05), 2).tolist()
    train_f, train_r, test_f, test_r = preprocess(df)

    agent = DQNAgent(train_f.shape[1], len(actions))
    train_env = LongShortHedgeEnv(train_f, train_r, actions)
    train_rewards = []

    patience, window = np.inf, 10
    best_avg_reward = -np.inf
    action_counts = np.zeros(len(actions))

    for ep in range(max_episodes):
        state = train_env.reset()
        total_reward = 0
        done = False
        daily_returns = []
        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done = train_env.step(action)
            daily_returns.append(reward)

            cumulative_returns = np.cumprod(1 + np.array(daily_returns)) - 1
            if cumulative_returns[-1] <= -1:
                print(f"❌ Episode {ep+1} early stopped: Cumulative return reached -100%")
                done = True

            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        train_rewards.append(total_reward)

        print(f"Episode {ep+1:3d} | Total Train Reward: {total_reward:.6f} | Epsilon: {agent.epsilon:.3f}")

        if len(train_rewards) >= window:
            avg_recent = np.mean(train_rewards[-window:])
            if avg_recent > best_avg_reward:
                best_avg_reward = avg_recent
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("✅ Early stopping: Converged based on moving average")
                break

    save_model(agent, path=f"{global_path}/Models/{model_name}.pth")
    test_env = LongShortHedgeEnv(test_f, test_r, actions)
    test_rewards, test_actions = evaluate_agent(test_env, agent, actions)

    metrics = evaluate_returns(test_rewards)
    save_metrics(metrics)
    plot_rewards(train_rewards, test_rewards)

    pd.DataFrame({"Test Action Weight": [actions[a] for a in test_actions], "Test Reward": test_rewards}).to_csv(f"{global_path}/Results/{model_name}_test_actions_rewards.csv", index=False)
    pd.DataFrame({"Train Episode Reward": train_rewards}).to_csv(f"{global_path}/Results/{model_name}_train_episode_rewards.csv", index=False)
    pd.DataFrame({"Action Index": list(range(len(actions))), "Action Weight": actions, "Action Count": action_counts}).to_csv(f"{global_path}/Results/{model_name}_action_counts.csv", index=False)

    print("\n📈 Test Performance:")
    for k, v in metrics.items():
        try : print(f"{k}: {v:.4f}")
        except : pass
        
# -------------------- 실행 --------------------
if __name__ == "__main__":
    df = pd.read_csv(f"{global_path}/data.csv")
    train_and_test(df, max_episodes=10000)