# Benchmarking Deep Reinforcement Learning Algorithms for Portfolio Optimization

> A controlled benchmark of four deep reinforcement learning algorithms — **PPO, A2C, SAC, and TD3** — against classical baselines (equal-weight 1/N and the Markowitz max-Sharpe portfolio) for long-only daily portfolio allocation over seven assets plus cash.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Stable--Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)
![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

Course project for **CISC 856 — Reinforcement Learning** (Group 7), School of Computing, Queen's University.

---

## Overview

Portfolio optimization asks how to split capital across assets to maximize return for an acceptable level of risk. The classical answer, Markowitz mean–variance optimization, assumes stationary and predictable markets — assumptions that real markets violate. Deep Reinforcement Learning (DRL) offers a model-free alternative that learns an allocation policy directly from market experience.

Rather than proposing a single agent, this project runs a **fair, controlled comparison**: the trading environment, reward function, feature set, and training budget are held identical across all four algorithms, so any performance difference is attributable to the learning algorithm alone. The agents are also compared against three non-learning baselines to see whether DRL actually adds value.

## Highlights

- 🧪 **Controlled 72-run benchmark**: 4 algorithms × 3 risk-aversion levels × 2 look-back windows × 3 random seeds.
- 🏗️ **Custom [Gymnasium](https://gymnasium.farama.org/) environment** for long-only daily allocation with a softmax action (including an explicit cash position) and a risk-adjusted, transaction-cost-aware reward.
- 🤖 **Four DRL agents** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): PPO, A2C (on-policy) and SAC, TD3 (off-policy).
- 📊 **Three classical baselines**: equal-weight (1/N), the Markowitz max-Sharpe (tangency) portfolio, and passive buy-and-hold.
- 📈 **Seven risk/return metrics** with genuine across-seed error bars, plus a concrete $10,000 one-year investment simulation.

## Methodology

The task is modelled as a Markov Decision Process. The agent acts once per trading day using only information available up to that day (fully causal).

**State / observation (36-dim, compact).** For each of the 7 risky assets, four features — 1-, 5-, and 21-day log-returns and 21-day rolling volatility — are standardized with a causal trailing-window z-score, then concatenated with the current portfolio weights (7 risky assets + cash).

**Action (8-dim).** The policy outputs an unconstrained vector whose last entry is cash. A softmax maps it to a long-only, fully-invested portfolio (weights ≥ 0, summing to 1), enforcing the no-short-selling and budget constraints for every algorithm.

**Reward.** A risk-adjusted return net of trading cost:

```
R_t = portfolio_return_t  -  λ · volatility_t  -  κ · turnover_t
```

where `λ ∈ {0, 0.5, 1}` controls risk aversion and `κ = 0.1%` is a transaction cost on risky-asset turnover.

**Baselines.** Equal-weight (daily rebalance), Markowitz max-Sharpe (trailing 252-day estimate, monthly rebalance), and buy-and-hold of the S&P 500 (SPY).

**Metrics.** Total return, CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown (MDD), and Calmar ratio.

## Dataset

Daily adjusted prices downloaded via [yfinance](https://github.com/ranaroussi/yfinance) for **seven assets** — AAPL, MSFT, NVDA, AMZN (tech), GLD (gold), USO (oil), SPY (market) — plus a cash sleeve.

- **Period:** 2015–2024 (2,494 trading days after feature computation)
- **Split:** chronological at 2022-01-01 → 1,742 training days / 752 out-of-sample test days

## Experimental Setup

| Setting | Value |
|---|---|
| Algorithms | PPO, A2C (on-policy); SAC, TD3 (off-policy) |
| Policy network | MLP (`MlpPolicy`), SB3 defaults |
| Timesteps per run | 50,000 (CPU) |
| Risk aversion λ | {0.0, 0.5, 1.0} |
| Look-back window L | {30, 60} days |
| Seeds | {0, 1, 2} |
| Transaction cost κ | 0.1% of risky turnover |
| **Total runs** | **72** |

## Results

Out-of-sample performance over the full test period (2022–2024). Agents show the best configuration per algorithm (mean over 3 seeds); Sharpe includes across-seed standard deviation. **Bold** = best in column.

| Strategy | Config | Total Ret. | CAGR | Volatility | Sharpe | Max DD | Calmar |
|:--|:--|--:|--:|--:|:--:|--:|--:|
| Equal-weight (1/N) | — | 74.8% | 20.6% | 21.4% | **0.98** | −28.2% | 0.73 |
| Markowitz (max-Sharpe) | 252d, monthly | **83.9%** | **22.7%** | 24.8% | 0.95 | −29.4% | **0.77** |
| **PPO** | λ=0.0, L=60 | 55.7% | 16.0% | 19.1% | 0.87 ± 0.10 | **−27.2%** | 0.59 |
| A2C | λ=0.0, L=60 | 47.0% | 13.8% | 19.8% | 0.75 ± 0.06 | −28.7% | 0.49 |
| SAC | λ=1.0, L=60 | 24.8% | 7.6% | **19.3%** | 0.47 ± 0.23 | −29.3% | 0.27 |
| TD3 | λ=1.0, L=30 | 23.0% | 7.1% | 21.8% | 0.42 ± 0.14 | −28.5% | 0.25 |

### Key findings

- **PPO is the strongest DRL agent** (test Sharpe 0.87), followed by A2C, then SAC and TD3.
- **On-policy methods (PPO, A2C) clearly beat off-policy methods (SAC, TD3)** within the 50k-step budget — the off-policy critics struggle to learn from the noisy, non-stationary daily reward signal.
- **The classical baselines remain hard to beat**: over the full test period no agent surpasses equal-weight (Sharpe 0.98) or Markowitz (0.95) on a risk-adjusted basis. A simple 1/N rule is a remarkably strong baseline.
- In a **$10,000 one-year simulation** over the 2022 bear market, every strategy loses money; the Markowitz portfolio ends highest (−18.3%), with the PPO agent mid-pack.
- **PPO is also the most robust** to its risk-aversion and look-back settings, while TD3 is the most sensitive.

> This is framed as a benchmarking study: the value is the controlled, honest comparison — including the finding that DRL does not automatically outperform simple classical allocation on this task.

## Repository Structure

```
.
├── rl_portfolio_project.ipynb   # main notebook: environment, training, evaluation, plots
├── report/                      # IEEE-format report (PDF / Word) and LaTeX source
├── figures/                     # generated plots (equity curves, heatmaps, etc.)
├── requirements.txt
└── README.md
```
*(Adjust to match your actual file layout.)*

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

**Main dependencies:** `stable-baselines3`, `gymnasium`, `yfinance`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `torch`.

<details>
<summary>Example <code>requirements.txt</code></summary>

```
stable-baselines3
gymnasium
yfinance
numpy
pandas
matplotlib
seaborn
scipy
torch
```
</details>

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook rl_portfolio_project.ipynb
```

The notebook downloads the price data, builds the Gymnasium environment, trains the full grid of agents, evaluates them against the baselines, and produces all figures and result tables. Finished runs are checkpointed so the grid is resumable, and training is logged to TensorBoard.

## Report

A full IEEE-format write-up (abstract, methodology with equations, experiments, results, and discussion) is available in the `report/` folder as both PDF and Word.

## Authors

**Group 7 - CISC 856, Queen's University**

- Osama Adel Saad
- Abdelrahman Ibrahim
- Aya Said Abdallah Noah
- Esraa Mohammed Elsayed

## Acknowledgements

Built with the open-source [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) libraries. Thanks to Dr. Sidney Givigi and the CISC 856 teaching team.

## License

Released under the MIT License — see `LICENSE` for details. *(Add a LICENSE file if you want this to apply.)*
