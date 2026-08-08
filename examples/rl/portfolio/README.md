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
    state.py           typed simulator state
    simulator.py       two-session portfolio transition logic
    interpreter.py     Qlib RL state/action adapters
    reward.py          net-return reward calculation
    policy.py          DQN construction and configuration

examples/rl/portfolio/
    README.md
    workflow_config_portfolio_dqn.yaml
    train.py
    backtest.py

tests/rl/portfolio/
    test_action.py
    test_reward.py
    test_simulator.py
```

Data preparation must remain separate from the simulator. The simulator
receives validated arrays or tables and must not load arbitrary files or train
models internally.

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
