================================
Portfolio DQN Development Workframe
================================

Purpose
========

This document defines the staged research and engineering plan for a
multi-asset portfolio strategy built with QlibRL and a Deep Q-Network (DQN).
The strategy rebalances every two trading sessions over a configured stock
universe.

The first objective is a correct, reproducible simulation. Profitability is
evaluated only after the data, portfolio accounting, and baseline strategies
have been verified.

Target System
=============

The intended decision flow is:

.. code-block:: text

    Qlib data and signals
            |
            v
    Leakage-safe observation builder
            |
            v
    DQN selects a discrete portfolio command
            |
            v
    Deterministic action interpreter
            |
            v
    Feasible target portfolio and cash weights
            |
            v
    Two-trading-session simulation
            |
            v
    Net reward and next state

DQN selects a portfolio-construction command rather than independently
selecting a weight for every stock. This keeps the action space discrete and
manageable. The deterministic action interpreter converts the command into
valid stock and cash weights.

Phase 0: Project Safety and Research Contract
=============================================

Development takes place on the ``research/portfolio-rl-v1`` branch. The
initial implementation adds a parallel ``qlib.rl.portfolio`` package and does
not change the behavior of the existing order-execution modules.

The research contract must define:

* the stock universe and its point-in-time membership rules;
* the features and prediction scores available at each decision time;
* chronological training, validation, and test periods;
* the two-trading-session rebalance convention;
* transaction costs, slippage, and tradability assumptions;
* benchmark strategies and evaluation metrics.

Every data field must have an explicit timestamp convention. Forward returns
are outcomes used by the simulator and must never enter an observation.

Deliverable
-----------

An agreed and version-controlled research specification.

Phase 1: Deterministic Portfolio Actions
========================================

Before introducing a neural network, implement and test the portfolio rules
available to DQN. The initial action vocabulary is:

.. list-table:: Initial discrete actions
   :header-rows: 1
   :widths: 10 30 60

   * - ID
     - Command
     - Behavior
   * - 0
     - Hold
     - Keep the current portfolio.
   * - 1
     - Cash
     - Move tradable positions to cash.
   * - 2
     - Equal weight
     - Equal-weight eligible positive-score stocks.
   * - 3
     - Conservative score
     - Use score weights with a lower equity budget.
   * - 4
     - Aggressive score
     - Use score weights with a higher equity budget.
   * - 5
     - Volatility adjusted
     - Weight eligible stocks by score divided by volatility.
   * - 6
     - Reduce exposure
     - Reduce tradable risky exposure by half.
   * - 7
     - Partial rebalance
     - Move halfway toward the standard score portfolio.

Each command must produce a portfolio satisfying these invariants:

.. code-block:: text

    every weight is finite and non-negative
    sum(stock weights) + cash weight = 1
    non-tradable stocks receive no new allocation
    existing non-tradable positions remain explicitly locked

The action meanings must remain stable within a model version. Changing an
action's meaning invalidates checkpoints trained with the earlier vocabulary.

Deliverables
------------

* Stable action definitions.
* A deterministic target-portfolio constructor.
* Unit tests covering all actions and constraints.

Phase 2: Leakage-Safe Data Pipeline
===================================

Build a Qlib-backed table for each decision date and instrument containing:

.. code-block:: text

    trading date
    instrument
    features available at the decision time
    prediction score available at the decision time
    tradable flag
    forward two-session return used only after the action

The pipeline must:

* use trading-calendar offsets rather than calendar-day offsets;
* preserve a stable and documented instrument ordering;
* define missing-feature and missing-return behavior;
* handle suspensions, delistings, and universe membership explicitly;
* fit normalization and other learned preprocessing on training data only;
* join data explicitly by trading date and instrument;
* prohibit backward filling from future observations.

Deliverables
------------

* Qlib data loader and typed data contract.
* Dataset validation and leakage tests.
* A small cached dataset for deterministic development tests.

Phase 3: Portfolio Simulator and Accounting
===========================================

The simulator owns the decision date, stock weights, cash weight, portfolio
value, executed turnover, costs, and episode termination state.

One environment transition follows this sequence:

#. Read the state at trading date ``t``.
#. Receive one discrete action.
#. Construct the requested target weights.
#. Apply tradability and portfolio constraints to obtain executed weights.
#. Calculate turnover, transaction costs, and any configured slippage.
#. Hold the executed portfolio for two trading sessions.
#. Apply the realized asset returns.
#. Update portfolio value and post-return weights.
#. Advance to the next decision date.

The initial reward is:

.. code-block:: text

    reward(t) = gross portfolio return(t, t+2) - transaction costs(t)

Transaction costs are calculated from executed weights and charged exactly
once. Risk penalties may be added later as independently configured reward
components.

Deliverables
------------

* A simulator that runs on synthetic data.
* Reward and transaction-cost calculations.
* Tests for state transitions and accounting invariants.
* Tests for zero turnover, full turnover, missing returns, and locked assets.

Phase 4: QlibRL Environment Integration
=======================================

Integrate the simulator with QlibRL's existing component boundaries:

* ``Simulator`` manages portfolio transitions.
* ``StateInterpreter`` produces the policy observation.
* ``ActionInterpreter`` converts the DQN action into a simulator action.
* ``Reward`` calculates the scalar learning signal.
* ``EnvWrapper`` supplies the Gym-compatible environment interface.

The action space is ``Discrete(8)`` in the initial version. The observation
must have a fixed shape even when universe membership changes. A practical
first observation contains:

* market-level aggregate features;
* current equity exposure and cash;
* portfolio turnover and backward-looking risk statistics;
* summary statistics of the cross-sectional score distribution;
* predicted properties of each candidate action portfolio.

This initial fixed-size representation should be implemented before exploring
per-stock neural encoders.

Deliverables
------------

* State and action interpreters.
* QlibRL environment configuration.
* A random-policy episode smoke test.

Phase 5: Deterministic Benchmarks
================================

Evaluate the action rules directly before training DQN. Required benchmarks
include:

* cash;
* equal weight;
* standard score weight;
* volatility-adjusted score weight;
* hold;
* random action selection;
* the configured market benchmark where appropriate.

Report at least annualized return, volatility, Sharpe ratio, maximum drawdown,
turnover, transaction costs, and average cash exposure. These results establish
whether the simulator and candidate actions behave sensibly.

Deliverable
-----------

A reproducible baseline report covering the validation and test periods.

Phase 6: DQN Training
=====================

Use Qlib's existing Tianshou integration where it remains compatible with the
portfolio environment. The initial training implementation includes:

* an online Q-network and target Q-network;
* replay buffer sampling;
* epsilon-greedy exploration;
* Huber loss and gradient clipping;
* periodic target-network updates;
* deterministic seeds, checkpoints, and metric logging.

All dataset splits are chronological:

.. code-block:: text

    training period -> validation period -> untouched test period

Dates must not be randomly mixed across these periods. Hyperparameters and
early stopping decisions use the validation period only.

Deliverables
------------

* Reproducible DQN configuration and training entry point.
* A short smoke-training test.
* Checkpoints and training diagnostics stored outside Git.

Phase 7: Evaluation and Diagnosis
=================================

Compare the selected DQN checkpoint against every deterministic baseline on
the untouched test period. In addition to portfolio metrics, inspect:

* frequency and persistence of every selected action;
* performance by market regime;
* sensitivity to transaction costs and slippage;
* gross versus net performance;
* turnover and exposure concentration;
* drawdown behavior;
* whether DQN collapses to one permanent action.

If DQN predominantly selects one command, the deterministic version of that
command is the essential comparison. Claims of improvement require repeated
seeds and economically meaningful out-of-sample results.

Deliverable
-----------

An out-of-sample evaluation report with diagnostics and baseline comparisons.

Phase 8: Hardening and Extensions
=================================

Only after the first complete pipeline is trustworthy should the project add:

* Double DQN to reduce Q-value overestimation;
* prioritized replay when supported by measured evidence;
* risk-aware reward components;
* walk-forward evaluation;
* more realistic liquidity, limit, suspension, and delisting behavior;
* integration with Qlib's detailed backtest engine;
* continuous integration, linting, and expanded documentation.

Proposed Repository Layout
==========================

.. code-block:: text

    qlib/rl/portfolio/
    |-- action.py
    |-- state.py
    |-- data.py
    |-- simulator.py
    |-- interpreter.py
    |-- reward.py
    |-- network.py
    `-- policy.py

    tests/rl/portfolio/
    |-- test_action.py
    |-- test_data.py
    |-- test_reward.py
    |-- test_simulator.py
    `-- test_environment.py

    examples/rl/portfolio/
    |-- README.md
    |-- workflow_config_portfolio_dqn.yaml
    |-- train.py
    `-- backtest.py

Development Gates
=================

Work proceeds in this order:

#. Finalize the research contract.
#. Implement and test deterministic actions.
#. Implement and test reward and accounting logic.
#. Implement and test the synthetic-data simulator.
#. Connect and validate Qlib data.
#. Run deterministic baselines.
#. Integrate the QlibRL environment.
#. Train a small DQN smoke model.
#. Run the formal out-of-sample backtest.
#. Review and merge only after all relevant tests pass.

Each phase is a gate. Later work must not be used to hide unresolved failures
in an earlier phase. In particular, DQN training begins only after the data,
action construction, portfolio accounting, and baselines are trustworthy.
