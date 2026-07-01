# Benchmarking Deep Reinforcement Learning Algorithms for Portfolio Optimization

> A controlled benchmark of four deep reinforcement learning algorithms — **PPO, A2C, SAC, and TD3** — against classical baselines (equal-weight 1/N and the Markowitz max-Sharpe portfolio) for long-only daily portfolio allocation over seven assets plus cash.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Stable--Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)
![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-orange)
![Report](https://img.shields.io/badge/report-IEEE%20format-8A2BE2)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Course project for **CISC 856 — Reinforcement Learning** (Group 7), School of Computing, Queen's University.

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Figures](#figures)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Report](#report)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Portfolio optimization asks how to split capital across assets to maximize return for an acceptable level of risk. The classical answer, Markowitz mean–variance optimization, assumes stationary and predictable markets — assumptions that real markets violate. Deep Reinforcement Learning (DRL) offers a model-free alternative that learns an allocation policy directly from market experience.

Rather than proposing a single agent, this project runs a **fair, controlled comparison**: the trading environment, reward function, feature set, and training budget are held identical across all four algorithms, so any performance difference is attributable to the learning algorithm alone. The agents are also compared against three non-learning baselines to see whether DRL actually adds value.

## Highlights

- 🧪 **Controlled 72-run benchmark**: 4 algorithms × 3 risk-aversion levels × 2 look-back windows × 3 random seeds.
- 🏗️ **Custom [Gymnasium](https://gymnasium.farama.org/) environment** for long-only daily allocation with a softmax action (including an explicit cash position) and a risk-adjusted, transaction-cost-aware reward.
- 🤖 **Four DRL agents** from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): PPO, A2C (on-policy) and SAC, TD3 (off-policy).
- 📊 **Three classical baselines**: equal-weight (1/N), the Markowitz max-Sharpe (tangency) portfolio, and passive buy-and-hold.
- 📈 **Seven risk/return metrics** with genuine across-seed error bars, plus a concrete \$10,000 one-year investment simulation.
- 🔁 **Reproducible**: fixed seeds, cached/resumable runs, TensorBoard logging, and an IEEE-format write-up with derivations.

## Methodology

The task is modelled as a Markov Decision Process. The agent acts once per trading day using only information available up to that day (**fully causal**: it acts at day *t*, and the return is realized at *t + 1*).

### State / observation (36-dim, compact)

For each of the $N = 7$ risky assets, four features 1-, 5-, and 21-day log-returns and 21-day rolling volatility are standardized with a **causal trailing-window z-score** (so no future information leaks into training):

$$
z(x_t) = \frac{x_t - \mu_{t-W:t}}{\sigma_{t-W:t} + \epsilon}, \qquad W = 252, \;\; \epsilon = 10^{-8}
$$

Stacking the $F = 4$ features across the $N$ assets and concatenating the current portfolio weights (risky assets **+ cash**) gives the observation:

$$
s_t = \big[\, z(\text{features}_t),\; w_t \,\big] \in \mathbb{R}^{FN + (N+1)} = \mathbb{R}^{36}
$$

### Action (8-dim)

The policy outputs an unconstrained vector whose last entry is cash. A **softmax** maps it to a long-only, fully-invested portfolio — enforcing the no-short-selling and budget constraints for every algorithm:

$$
w_t = \operatorname{softmax}(a_t), \qquad w_{t,i} \ge 0, \qquad \sum_{i=1}^{N+1} w_{t,i} = 1
$$

### Reward

A risk-adjusted return net of trading cost:

$$
R_t = \underbrace{\sum_{i=1}^{N} w_{t,i}\, \rho_{t+1,i} + w_{t,N+1}\, r_f}_{\text{portfolio return}} \; - \; \lambda\, \sigma^{p}_{t} \; - \; \kappa\, \lVert w^{\text{risky}}_{t} - w^{\text{risky}}_{t-1} \rVert_{1}
$$

where $\rho_{t+1}$ are next-day simple returns, $\sigma^{p}_{t}$ is the rolling portfolio volatility, $\lambda \in \{0, 0.5, 1\}$ controls risk aversion ($\lambda = 0$ pure profit-seeker, $\lambda = 1$ strongly risk-averse), and $\kappa = 0.1\%$ is a transaction cost on risky-asset turnover.

### Baselines

Equal-weight (daily rebalance), buy-and-hold of the S&P 500 (SPY), and the Markowitz max-Sharpe (tangency) portfolio, re-estimated from a trailing 252-day window and rebalanced monthly:

$$
w^{\star} = \operatorname*{arg\,max}_{w} \; \frac{w^{\top} \mu}{\sqrt{w^{\top} \Sigma\, w}} \qquad \text{s.t.} \;\; \sum_i w_i = 1, \;\; w_i \ge 0
$$

### Metrics

Total return, CAGR, annualized volatility, Sortino, maximum drawdown (MDD), plus the Sharpe and Calmar ratios:

$$
\text{Sharpe} = \frac{\bar{r}}{\sigma_r}\sqrt{252}, \qquad \text{Calmar} = \frac{\text{CAGR}}{\lvert \text{MDD} \rvert}
$$

## Dataset

Daily adjusted prices downloaded via [yfinance](https://github.com/ranaroussi/yfinance) for **seven assets** — AAPL, MSFT, NVDA, AMZN (tech), GLD (gold), USO (oil), SPY (market) — plus a cash sleeve.

- **Period:** 2015–2024 (2,494 trading days after feature computation)
- **Split:** chronological at `2022-01-01` → 1,742 training days / 752 out-of-sample test days (all test results are strictly out of sample)

## Experimental Setup

| Setting | Value |
|---|---|
| Algorithms | PPO, A2C (on-policy); SAC, TD3 (off-policy) |
| Policy network | MLP (`MlpPolicy`), SB3 defaults |
| Timesteps per run | 50,000 (CPU) |
| Off-policy extras | replay buffer 50,000; learning starts 1,000 |
| Risk aversion λ | {0.0, 0.5, 1.0} |
| Look-back window L | {30, 60} days |
| Seeds | {0, 1, 2} |
| Transaction cost κ | 0.1% of risky turnover |
| **Total runs** | **72** |

> **Sanity check.** Before the full grid, a short 20k-step PPO run reached a test Sharpe of **0.86** (vs. 0.98 for 1/N), confirming the data → environment → agent loop works end-to-end.

## Results

Out-of-sample performance over the full test period (**2022–2024**). Agents show the best configuration per algorithm (mean over 3 seeds); Sharpe includes across-seed standard deviation. **Bold** = best in column.

| Strategy | Config | Total Ret. | CAGR | Volatility | Sharpe | Max DD | Calmar |
|:--|:--|--:|--:|--:|:--:|--:|--:|
| Equal-weight (1/N) | — | 74.8% | 20.6% | 21.4% | **0.98** | −28.2% | 0.73 |
| Markowitz (max-Sharpe) | 252d, monthly | **83.9%** | **22.7%** | 24.8% | 0.95 | −29.4% | **0.77** |
| **PPO** | λ=0.0, L=60 | 55.7% | 16.0% | **19.1%** | 0.87 ± 0.10 | **−27.2%** | 0.59 |
| A2C | λ=0.0, L=60 | 47.0% | 13.8% | 19.8% | 0.75 ± 0.06 | −28.7% | 0.49 |
| SAC | λ=1.0, L=60 | 24.8% | 7.6% | 19.3% | 0.47 ± 0.23 | −29.3% | 0.27 |
| TD3 | λ=1.0, L=30 | 23.0% | 7.1% | 21.8% | 0.42 ± 0.14 | −28.5% | 0.25 |

### \$10,000 invested over one year (2022 bear market)

Ranked by final value. Returns are negative, so a **less-negative** number is a **smaller loss** (better). `Max DD` is the worst **peak-to-trough** dip during the year — a risk measure, not the final loss.

| Strategy | Final \$ | ROI | Max DD | Sharpe |
|:--|--:|--:|--:|--:|
| Markowitz (max-Sharpe) | **8,167** | **−18.3%** | −29.0% | **−0.51** |
| Buy & hold SPY | 8,164 | −18.4% | **−24.5%** | −0.72 |
| RL model (PPO) | 7,825 | −21.8% | −26.7% | −0.86 |
| Equal-weight (1/N) | 7,638 | −23.6% | −28.2% | −0.80 |

> **Reading this fairly.** 2022 was a broad market downturn in which essentially every asset fell; since all strategies are long-only and fully invested, the loss reflects the market regime, not a model failure. The same PPO agent that finishes third here returns **+55.7%** over the full 2022–2024 window once the recovery is included — so a single down-year snapshot understates the model.

### Key findings

- **PPO is the strongest DRL agent** (test Sharpe 0.87), followed by A2C, then SAC and TD3.
- **On-policy methods (PPO, A2C) clearly beat off-policy methods (SAC, TD3)** within the 50k-step budget — the off-policy critics struggle to bootstrap value from the noisy, non-stationary daily reward signal.
- **The classical baselines remain hard to beat**: over the full test period no agent surpasses equal-weight (Sharpe 0.98) or Markowitz (0.95) on a risk-adjusted basis. A simple 1/N rule is a remarkably strong baseline.
- **PPO is also the most robust** to its risk-aversion and look-back settings, while TD3 is the most sensitive (even going negative at λ=1, L=60).
- A **longer look-back (L=60)** helps the on-policy methods; both PPO and A2C peak there.

> This is framed as a **benchmarking study**: the value is the controlled, honest comparison — including the finding that DRL does not automatically outperform simple classical allocation on this task.

## Figures

All plots are generated by the notebook and saved to `figures/`:

| # | Figure | Shows |
|--:|:--|:--|
| 1 | Dataset | Normalized price paths (train/test split) + daily-return correlation matrix |
| 2 | Asset characteristics | Risk/return by asset + 63-day rolling annualized volatility |
| 3 | Sharpe by algorithm | Best config per algorithm (mean ± std) vs. the 1/N and Markowitz baselines |
| 4 | (λ, L) heat-maps | Mean test Sharpe for every risk-aversion / look-back pair, per algorithm |
| 5 | Training curves | Smoothed episode reward — on-policy stable, off-policy struggling |
| 6 | Out-of-sample equity | Growth of \$1 for each best agent vs. baselines, 2022–2024 |
| 7 | Best-agent behaviour | PPO portfolio weights (with cash sleeve) + drawdown over time |
| 8 | \$10,000 simulation | Portfolio value and cumulative ROI over the 2022 bear market |

## Repository Structure

```
.
├── rl_portfolio_project.ipynb   # main notebook: environment, training, evaluation, plots
├── report/                      # IEEE-format report (PDF + Word) and LaTeX source
├── figures/                     # generated plots (equity curves, heat-maps, etc.)
├── requirements.txt
├── LICENSE
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

The notebook downloads the price data, builds the Gymnasium environment, trains the full grid of agents, evaluates them against the baselines, and produces all figures and result tables. Finished runs are checkpointed so the grid is **resumable**, and training is logged to **TensorBoard**:

```bash
tensorboard --logdir tb/
```

## Report

A full **IEEE-format write-up** (abstract, methodology with equations, experiments, results, and discussion) is available in the [`report/`](report/) folder as both **PDF** and **Word**.

## Citation

If you use this work, please cite:

```bibtex
@misc{group7_2026_drl_portfolio,
  title  = {Benchmarking Deep Reinforcement Learning Algorithms for Portfolio Optimization},
  author = {Ibrahim Kamal Eldin, Abdelrahman and Moursy, Ossama Adel and
            Noah, Aya Said Abdallah and Elsayed, Esraa Mohammed},
  year   = {2026},
  note   = {CISC 856 — Reinforcement Learning, School of Computing, Queen's University}
}
```

## Authors

**Group 7 — CISC 856, School of Computing, Queen's University**

| Name | Student ID |
|:--|:--|
| Abdelrahman Ibrahim Kamal Eldin | 20596377 |
| Ossama Adel Moursy | 20595449 |
| Aya Said Abdallah Noah | 20596338 |
| Esraa Mohammed Elsayed | 20596301 |

Instructor: **Dr. Sidney Givigi**

## Acknowledgements

Built with the open-source [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) libraries. Thanks to Dr. Sidney Givigi and the CISC 856 teaching team for their guidance and feedback.

## License

Released under the MIT License — see [`LICENSE`](LICENSE) for details. *(Add a `LICENSE` file if you want this to apply.)*
