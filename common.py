import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def evaluate_returns(daily_returns):
    cumulative_returns = np.cumprod(1 + daily_returns) - 1
    cumulative_return = cumulative_returns[-1]
    annualized_return = (1 + cumulative_return) ** (252 / len(daily_returns)) - 1
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
    sortino_ratio = annualized_return / (np.std([r for r in daily_returns if r < 0]) * np.sqrt(252)) if any(r < 0 for r in daily_returns) else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
    avg_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    skewness = pd.Series(daily_returns).skew()
    kurtosis = pd.Series(daily_returns).kurt()

    return {
        "Cumulative Return": float(cumulative_return),
        "Annualized Return": float(annualized_return),
        "Annualized Volatility": float(annualized_volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "Sortino Ratio": float(sortino_ratio),
        "Calmar Ratio": float(calmar_ratio),
        "Average Daily Return": float(avg_daily_return),
        "Std Daily Return": float(std_daily_return),
        "Skewness": float(skewness),
        "Kurtosis": float(kurtosis),
        "Cumulative Return Series": cumulative_returns
    }



# -------------------- 데이터 전처리 --------------------
def preprocess(df):
    df['date'] = pd.to_datetime(df.iloc[:, 0], format="%Y_%m_%d")
    date_col = df['date']
    df = df.drop(columns=[df.columns[0]])

    df[['NASDAQ', 'SP500']] = df[['NASDAQ', 'SP500']].apply(pd.to_numeric, errors='coerce')

    feature_cols = df.columns.difference(['date', 'NASDAQ', 'SP500'])
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    varying_cols = [col for col in numeric_cols if df[col].nunique() > 1]
    if not varying_cols:
        raise ValueError("❌ No feature columns with variance found.")

    features = df[varying_cols].pct_change().fillna(0)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    features = features.clip(-1e6, 1e6)

    returns = df[['NASDAQ', 'SP500']].pct_change().shift(-1).dropna()
    min_len = min(len(features), len(returns))
    features = features.iloc[:min_len]
    returns = returns.iloc[:min_len]
    dates = date_col.iloc[-min_len:]

    train_mask = dates < pd.Timestamp("2021-01-01")
    test_mask = dates >= pd.Timestamp("2022-01-01")
    if train_mask.sum() == 0:
        raise ValueError("❌ No training data before 2021-01-01.")

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    return (
        scaled_features[train_mask.values], returns[train_mask.values].values,
        scaled_features[test_mask.values], returns[test_mask.values].values
    )

def plot_rewards(train_rewards, test_rewards):
    plt.plot(train_rewards, label="Train Episode Reward")
    plt.axhline(np.mean(test_rewards), color="r", linestyle="--", label="Avg Test Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig("train_test_rewards.png")
    plt.close()

def save_model(agent, path):
    torch.save(agent.model.state_dict(), path)

def evaluate_agent(env, agent, actions):
    state = env.reset()
    rewards, actions_taken = [], []
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        rewards.append(reward)
        actions_taken.append(action)
        state = next_state
    return np.array(rewards), actions_taken

def save_metrics(metrics, path="performance_metrics.csv"):
    if "Cumulative Return Series" in metrics:
        metrics = {k: v for k, v in metrics.items() if k != "Cumulative Return Series"}
    pd.DataFrame([metrics]).to_csv(path, index=False)
