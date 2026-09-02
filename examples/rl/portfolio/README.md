# Portfolio DQN Research Project

This project adds a multi-asset portfolio-rebalancing experiment alongside
Qlib's existing single-asset order-execution RL implementation. The first
version is deliberately isolated from existing Qlib behavior: it adds a new
package, examples, and tests without changing the order-execution modules.

## Experiment artifact layout

RL output paths are derived from the selected workflow configuration. Given:

```text
<experiment>/configs/<workflow>.yaml
```

the benchmark, training, and evaluation commands automatically use:

```text
<experiment>/portfolio_dqn_rl/
├── baselines/
├── training/
└── evaluation/
```

For a future signal experiment, pass its workflow and prediction artifact:

```bash
python examples/rl/portfolio/run_benchmarks.py \
    --workflow-config <experiment>/configs/<workflow>.yaml

python examples/rl/portfolio/train.py \
    --workflow-config <experiment>/configs/<workflow>.yaml

python examples/rl/portfolio/evaluate.py \
    --workflow-config <experiment>/configs/<workflow>.yaml
```

An explicit `--output-dir` remains available for benchmark and training runs.
Evaluation also accepts explicit `--training-dir` and `--output-dir` overrides.

## Research objective

Train a DQN agent to choose a portfolio-rebalancing command every two trading
days. A deterministic action interpreter converts that discrete command into
valid target weights for the configured stock universe and cash.

The first milestone is simulator correctness, not profitability.

## Problem definition

### Decision interval

- The agent observes information available after close `t` and selects an
  action.
- The feasible target portfolio is executed at close `t + 1`.
- The resulting portfolio is held from close `t + 1` through close `t + 3`.
- The next action is selected two trading sessions after the previous action.
- Calendar-day offsets must never be used as a substitute for trading-session
  offsets.

### Observation

The observation at `t` may contain:

- configured Qlib features and model scores available at `t`;
- the date-specific tradable mask;
- current asset weights and cash weight;
- backward-looking market or portfolio context.

The observation must not contain the realized return, price, tradability, or
feature values from after `t`.

### Discrete action space

DQN selects one portfolio command. It does not assign a separate discrete
weight to every stock. The initial action vocabulary is:

| ID | Command | Target behavior |
|---:|---|---|
| 0 | Hold | Keep current weights, subject to forced non-tradable handling |
| 1 | Cash | Move tradable positions to cash |
| 2 | Equal weight | Equal-weight eligible positive-score stocks |
| 3 | Conservative score | Score-weight eligible stocks with a lower equity budget |
| 4 | Aggressive score | Score-weight eligible stocks with a higher equity budget |
| 5 | Volatility adjusted | Weight eligible stocks by score divided by volatility |
| 6 | Reduce exposure | Move halfway from current risky assets toward cash |
| 7 | Partial rebalance | Move halfway toward the standard score-weight portfolio |
| 8 | Increase exposure | Invest 25% of cash into positive-score existing holdings, or positive-score candidates when none are held |
| 9 | Top-score concentrated | Allocate 95% equity to at most five highest positive-score stocks |
| 10 | Rotate worst to best | Replace weak tradable holdings only when an equal number of positive-score replacements is available |
| 11 | Defensive volatility | Allocate 50% equity using score divided by volatility |

IDs 0 through 7 retain their original meanings. Adding IDs 8 through 11
changes both the observation and DQN output dimensions, so an 8-action
checkpoint cannot be reused; the DQN must be retrained.

```yaml
increase_exposure_ratio: 0.25
concentrated_holdings: 5
concentrated_equity: 0.95
rotation_count: 2
defensive_equity: 0.50
```

The action vocabulary is configuration-backed and must remain stable within a
trained model version. Changing its meaning invalidates existing checkpoints.

### Portfolio constraints

Version 1 is long-only:

- every asset weight is finite and non-negative;
- cash is always available;
- weights, including cash, sum to one within numerical tolerance;
- non-tradable assets cannot receive new allocation;
- position handling for assets that become non-tradable is explicit rather
  than silently discarding their value.

### Reward

The DQN learning reward over the two-session holding period is:

```text
portfolio_net_return_t = gross_return_t - transaction_cost_t
one_way_turnover_t = max(buy_weight_t, sell_weight_t)
turnover_penalty_t = turnover_penalty_rate * one_way_turnover_t
learning_reward_t = portfolio_net_return_t - turnover_penalty_t
```

Turnover and transaction cost must be calculated from executed weights, not
from an infeasible raw target. ``learning_reward`` trains DQN, while portfolio
value and performance use ``portfolio_net_return`` so the artificial penalty
is never deducted as real money. The active experiment configures
``turnover_penalty: 0.002``.

## Data contracts

Prepared inputs use a trading-date-by-instrument layout with explicit labels
for:

- observation features;
- prediction scores;
- asset returns used only after the action when calculating reward;
- tradability;
- current portfolio state.

All joins must be keyed explicitly by trading date and instrument. Missing
values must have documented behavior; they must not be filled using future
observations.

## Planned package boundaries

```text
qlib/rl/portfolio/
    action.py          discrete action definitions and target construction
    data.py            leakage-safe prediction and market-data alignment
    simulator.py       two-session portfolio transition logic
    interpreter.py     Qlib RL state/action adapters
    reward.py          net-return reward calculation
    integration.py     Gym-compatible QlibRL environment assembly
    policy.py          DQN construction and configuration

examples/rl/portfolio/
    README.md
    workflow_config_portfolio_dqn.yaml
    train.py
    backtest.py

tests/rl/portfolio/
    test_action.py
    test_data.py
    test_integration.py
    test_reward.py
    test_simulator.py
```

Data preparation must remain separate from the simulator. The simulator
receives validated arrays or tables and must not load arbitrary files or train
models internally.

## Phase 4 environment interface

`make_portfolio_env(data)` assembles the portfolio simulator, fixed-size state
interpreter, discrete action interpreter, and net-return reward with QlibRL's
`EnvWrapper`. The resulting Gym interface has:

- `Discrete(12)` actions with the stable IDs documented above;
- a 65-value `float32` observation;
- one scalar net two-session return per step;
- one independent episode for the supplied chronological data split.

The observation contains 17 global features describing portfolio exposure,
recent realized portfolio returns, tradability, and cross-sectional score and
volatility summaries. It also contains four predicted properties for each of
the twelve candidate portfolios: equity exposure, turnover, score exposure,
and diagonal volatility. Realized forward asset returns are outcomes owned by
the simulator and are never passed to the state interpreter.

## Phase 5 market benchmark

Run the validation- and test-period engineering baselines from the dedicated
environment:

```bash
conda run -n qlib_rl_env python examples/rl/portfolio/run_benchmarks.py
```

The command evaluates only the configured `^NDX` market benchmark. Individual
portfolio commands remain inside the DQN action space but are not standalone
research strategies. Reports are written by default under the source signal
experiment at
`alpha158_szrankguard_rolling_horizon2_step10/portfolio_dqn_rl/baselines`,
outside Git.

The market index is a price-return reference and does not generate simulated
stock orders or transaction costs. The DQN training and evaluation commands
read their portfolio cost parameters from workflow configurations.

Returns are non-overlapping two-session returns, so annualization uses 126
periods per year. Reported turnover is cumulative one-way turnover. The Sharpe
ratio assumes a zero risk-free rate, and maximum drawdown includes the initial
portfolio value.

## Phase 6 DQN engineering training

Train the small Double-DQN model from the dedicated environment:

```bash
conda run -n qlib_rl_env python examples/rl/portfolio/train.py
```

The command trains on the engineering training split, selects the best epoch
using penalized validation learning return, saves and reloads the checkpoint, and then
runs one test episode. The model receives 65 features and emits twelve raw
Q-values. Training uses replay sampling, linearly decayed epsilon-greedy
exploration, a target network, Huber loss, gradient clipping, and fixed random
seeds.

By default, the checkpoint, training history, exact configuration, validation
and test metrics, action frequencies, and transition audit are written under
the source signal experiment at
`alpha158_szrankguard_rolling_horizon2_step10/portfolio_dqn_rl/training`,
outside Git. This short run is an engineering verification and not evidence of
a robust learned investment strategy.

## Phase 7 engineering evaluation

Compare the saved DQN checkpoint with the configured market benchmark:

```bash
conda run -n qlib_rl_env python examples/rl/portfolio/evaluate.py
```

The evaluation reuses the saved model without fitting it again and reads costs
only from the workflow stored in the training record. It reports action
persistence, market-regime behavior, gross and net returns, turnover, cash
exposure, stock-weight concentration, drawdown, and action collapse. The
Markdown report and audit CSVs are written under the source signal experiment
at `alpha158_szrankguard_rolling_horizon2_step10/portfolio_dqn_rl/evaluation`,
outside Git.

When `PortfolioDQNRecord` runs the Qlib backtest, this directory is replaced
with a live-execution audit from that same run:

- `dqn_actions.csv`: every scheduled DQN decision and its requested turnover;
- `executed_orders.csv`: orders actually filled by Qlib, including side, amount,
  price, value, and cost;
- `daily_holdings.csv`: each daily stock position and `holding_sessions`;
- `portfolio_report.csv`: Qlib's daily account, return, cost, cash, and benchmark;
- `action_frequency.csv`: action counts from the live Qlib strategy;
- `comparison_metrics.csv`: DQN versus market return from the Qlib report;
- `strategy_diagnostics.csv`: decision, execution, dominant-action, and maximum
  holding-session diagnostics;
- `run_manifest.json`: dates and row counts identifying the source run.

These files describe the Qlib execution path. They are not copied from the
separate offline simulator.

## Rolling checkpoint lifecycle

Rolling RL trainers use the shared `qlib.rl.checkpoint.RollingCheckpointRun`
manager. It accepts the dynamically generated window IDs, stages the complete
new checkpoint set, and publishes it only after every expected window exists.
Publication replaces and clears the previous set, so a shorter later run does
not leave stale window directories. An incomplete run discards only staging
and preserves the previous completed set. This lifecycle is reusable by other
rolling RL strategies; it is not specific to Portfolio DQN.

## Milestone 1 acceptance criteria

- All twelve actions produce deterministic target portfolios.
- Target portfolios satisfy the long-only, cash, sum-to-one, and tradability
  constraints.
- Reward includes the configured transaction cost exactly once.
- One environment step advances exactly two entries in the trading calendar.
- Tests cover missing data, no tradable stocks, forced non-tradable holdings,
  zero turnover, and full turnover.
- A synthetic-data episode runs without requiring external market data.
- Existing Qlib RL tests continue to pass.

## Development workflow

Development occurs on `research/portfolio-rl-v1`. Changes should be committed
in small units, for example:

```text
docs: define portfolio DQN research contract
feat: add discrete portfolio action interpreter
test: cover portfolio action constraints
feat: add two-session portfolio simulator
```

The branch should be merged only after the new unit tests and the relevant
existing Qlib RL tests pass.
