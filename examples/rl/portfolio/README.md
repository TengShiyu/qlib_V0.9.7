# Portfolio DQN Research Project

This project adds a multi-asset portfolio-rebalancing experiment alongside
Qlib's existing single-asset order-execution RL implementation. The first
version is deliberately isolated from existing Qlib behavior: it adds a new
package, examples, and tests without changing the order-execution modules.

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

The initial reward is the net portfolio return over the two-session holding
period:

```text
reward_t = gross_return(close(t+1), close(t+3)) - transaction_cost_t
```

Turnover and transaction cost must be calculated from executed weights, not
from an infeasible raw target. Risk penalties may be introduced later as
separately configured reward components.

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

- `Discrete(8)` actions with the stable IDs documented above;
- a 49-value `float32` observation;
- one scalar net two-session return per step;
- one independent episode for the supplied chronological data split.

The observation contains 17 global features describing portfolio exposure,
recent realized portfolio returns, tradability, and cross-sectional score and
volatility summaries. It also contains four predicted properties for each of
the eight candidate portfolios: equity exposure, turnover, score exposure,
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
research strategies. Reports are written by default to
`/home/shiyu/qlib_experiment/portfolio_dqn_baselines`, outside Git.

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
using validation total return only, saves and reloads the checkpoint, and then
runs one test episode. The model receives 49 features and emits eight raw
Q-values. Training uses replay sampling, linearly decayed epsilon-greedy
exploration, a target network, Huber loss, gradient clipping, and fixed random
seeds.

By default, the checkpoint, training history, exact configuration, validation
and test metrics, action frequencies, and transition audit are written to
`/home/shiyu/qlib_experiment/portfolio_dqn_training`, outside Git. This short
run is an engineering verification and not evidence of a robust learned
investment strategy.

## Phase 7 engineering evaluation

Compare the saved DQN checkpoint with the configured market benchmark:

```bash
conda run -n qlib_rl_env python examples/rl/portfolio/evaluate.py
```

The evaluation reuses the saved model without fitting it again and reads costs
only from the workflow stored in the training record. It reports action
persistence, market-regime behavior, gross and net returns, turnover, cash
exposure, stock-weight concentration, drawdown, and action collapse. The
Markdown report and audit CSVs are written
to `/home/shiyu/qlib_experiment/portfolio_dqn_evaluation`, outside Git.

## Milestone 1 acceptance criteria

- All eight actions produce deterministic target portfolios.
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
