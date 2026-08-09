===============================
Portfolio DQN Research Contract
===============================

Status
======

Phase 0 is complete. This research contract was approved on 2026-08-08 for the
initial portfolio DQN project. Implementation must follow this contract and
the :doc:`portfolio_dqn_workframe` unless a later, reviewed change updates the
documentation first.

Research Objective
==================

Build a reproducible DQN strategy that selects a discrete portfolio command
for the ``szrankguard`` stock universe every two trading sessions. A
deterministic action interpreter converts the selected command into feasible
stock and cash weights.

The first milestone validates integration and portfolio accounting. It does
not attempt to establish that the strategy is profitable.

Upstream Signal Experiment
==========================

The authoritative upstream configuration is:

.. code-block:: text

    /home/shiyu/qlib_experiment/
        alpha158_szrankguard_rolling_horizon2_step10/
        configs/
        workflow_config_szrankguard_topk20drop2_rolling_h2_step10.yaml

The portfolio project inherits the following settings from that experiment:

.. list-table:: Upstream configuration
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Approved value
   * - Qlib provider
     - ``/mnt/hdd/qlib_data/us_data``
   * - Region
     - ``us``
   * - Model and execution universe
     - ``szrankguard``
   * - Feature handler
     - ``Alpha158``
   * - Signal model
     - ``DEnsembleModel`` with the upstream configuration
   * - Prediction horizon
     - 2 trading sessions
   * - Rolling-model step
     - 10 trading sessions
   * - Rolling type
     - ``ROLL_SD``
   * - Benchmark
     - ``^NDX``
   * - Initial account value
     - 100,000
   * - Execution price convention
     - Close

Horizon, Rolling Step, and RL Interval
======================================

These three settings describe different parts of the system and must not be
used interchangeably:

``prediction horizon = 2``
    The DoubleEnsemble score predicts a return spanning two trading sessions.

``rolling-model step = 10``
    The supervised rolling experiment advances its train, validation, and
    test windows by ten trading sessions before fitting the next model. This
    does not control portfolio rebalancing.

``RL holding interval = 2``
    The DQN agent selects a portfolio command every two trading sessions.

The upstream rolling runner generates this label:

.. code-block:: text

    Ref($close, -3) / Ref($close, -1) - 1

For a score associated with trading date ``t``, the label measures the return
from close ``t+1`` through close ``t+3``.

Decision Timeline
=================

The approved transition timeline is:

.. code-block:: text

    after close t:
        observe information and score associated with t
        select one DQN action

    close t+1:
        execute the feasible target portfolio

    close t+1 through close t+3:
        hold the executed portfolio for two trading sessions

    close t+3:
        end the holding period and execute the next approved target

    reward:
        net portfolio return from close(t+1) to close(t+3)

All offsets refer to entries in the Qlib trading calendar, not calendar days.
The one-session delay between observation and execution prevents the strategy
from trading on a closing price before the corresponding daily information is
available.

Observation Contract
====================

An observation may contain only information available after close ``t`` and
before execution at close ``t+1``. The initial observation may include:

* the rolling DoubleEnsemble scores associated with ``t``;
* current portfolio weights and cash;
* the tradable mask known for the execution decision;
* backward-looking market and portfolio statistics;
* summary properties of each deterministic candidate portfolio.

The observation must not contain:

* the realized return from ``t+1`` to ``t+3``;
* prices, features, scores, or tradability learned after the decision time;
* preprocessing statistics fitted using validation or test data.

Action Contract
===============

DQN selects one discrete portfolio-construction command. It does not assign a
separate discrete weight to every stock. The initial action vocabulary is:

.. list-table:: Initial action vocabulary
   :header-rows: 1
   :widths: 10 35 55

   * - ID
     - Command
     - Intended behavior
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

Every constructed target must be long-only, finite, non-negative, and sum to
one including cash. A non-tradable stock cannot receive new allocation.
Existing non-tradable holdings must remain explicitly represented until they
can be sold.

Reward and Cost Contract
========================

The initial reward is:

.. code-block:: text

    reward = gross portfolio return over two sessions
             - transaction costs

To reproduce the upstream experiment, the initial integration uses:

.. code-block:: text

    buy cost:     0
    sell cost:    0
    minimum cost: 0

Zero costs are a reproduction setting, not an assumption that trading is
free. Formal evaluation must include a documented nonzero-cost sensitivity
test. Costs must be calculated from executed turnover and charged exactly
once.

The runtime reads ``account``, ``open_cost``, ``close_cost``, and ``min_cost``
from the selected workflow YAML. ``open_cost`` maps to the simulator's buy
rate, ``close_cost`` maps to its sell rate, and ``min_cost`` is a dollar
minimum applied separately to every nonzero stock order. The documented
sensitivity workflow passes through ``open_cost=0.0005`` and
``close_cost=0.0015`` as decimal cost ratios, with ``min_cost=5`` as the
absolute minimum per stock order.

Missing Data Accounting
=======================

The engineering simulator uses these explicit rules:

* a missing score makes that stock ineligible for a new score-based allocation;
* a missing or non-positive execution close or volume makes the stock
  non-tradable at that execution boundary;
* a missing reward-end price must not be used in advance to block an otherwise
  feasible purchase, because that would reveal future availability;
* when a held stock has no valid reward-end return, its execution valuation is
  carried forward by applying a zero return for that transition;
* every transition reports the total held weight affected by missing returns.

Engineering and smoke-test reports must disclose missing-return weight. The
zero-return convention is a practical valuation rule for the first runnable
pipeline, not evidence that missing market data has no economic effect.

Supervised Data Periods
=======================

The upstream experiment defines:

.. code-block:: text

    supervised train:      2021-01-01 through 2023-12-31
    supervised validation: 2024-01-01 through 2024-12-31
    supervised test:       2025-01-01 through 2026-03-31
    configured backtest:   2025-01-01 through 2026-03-30

Qlib maps these calendar boundaries to valid trading sessions.

Existing Prediction Artifact
============================

The inspected final combined artifact contains:

.. code-block:: text

    rows:                78,868
    trading dates:       320
    unique instruments:  247
    first date:          2025-01-02
    last date:           2026-04-14

Initial integration must restrict the artifact to the configured backtest end
date, 2026-03-30. Later dates must not enter the experiment silently.

The artifact provides roughly 160 non-overlapping two-session decisions. This
is sufficient for integration tests, deterministic baselines, and a technical
DQN smoke run. It is not sufficient evidence for a robust learned investment
policy.

Engineering DQN Split
=====================

The first implementation milestone is an end-to-end runnable program. For
that milestone only, the existing artifact is divided chronologically:

.. list-table:: Engineering-only RL split
   :header-rows: 1
   :widths: 20 30 30 20

   * - Purpose
     - Start
     - End
     - Complete transitions
   * - DQN training
     - 2025-01-02
     - 2025-09-30
     - 92
   * - DQN validation
     - 2025-10-01
     - 2025-12-31
     - 31
   * - DQN test
     - 2026-01-02
     - 2026-03-30
     - 29

The counts were measured from the combined prediction artifact and include
only complete two-session transitions. Each split is an independent episode.
A transition must not begin in one split and calculate its execution, return,
or reward using dates from another split.

This split is approved to demonstrate that the program can:

* load real rolling predictions;
* construct feasible portfolio actions;
* run two-session transitions and calculate rewards;
* train a small DQN without failing;
* save and reload a checkpoint;
* evaluate validation and test episodes;
* produce a metrics report.

The split is not approved as evidence that DQN is robust or profitable. Any
reported results must be labelled as engineering or smoke-test results.

RL Dataset Separation
=====================

The existing combined prediction artifact is approved for the chronological
engineering split above. The three periods must remain isolated, and test
results must not influence model fitting or checkpoint selection.

Before formal DQN evaluation, the project must generate a longer chronological
series of rolling out-of-sample predictions using the same upstream
configuration structure. That series must be divided in time order:

.. code-block:: text

    earlier predictions -> DQN training
    middle predictions  -> DQN validation and model selection
    latest predictions  -> untouched DQN test

Dates must not be randomly distributed across these periods.

Benchmarks and Required Metrics
===============================

The market benchmark is ``^NDX``. DQN must also be compared with deterministic
strategy baselines:

* cash;
* hold;
* equal weight;
* standard score weight;
* volatility-adjusted score weight;
* random action selection.

At minimum, report annualized return, volatility, Sharpe ratio, maximum
drawdown, turnover, transaction costs, average cash exposure, and action
selection frequency. Results must be reported both with the reproduction cost
setting and with a documented nonzero-cost sensitivity setting.

Universe Limitation
===================

The current ``szrankguard`` instrument file assigns broad static membership
dates to stocks selected from currently available support files. Historical
results may therefore contain survivorship bias. Early results are engineering
results unless and until the universe is replaced with defensible point-in-time
membership data.

Phase 0 Exit Criteria
=====================

Phase 0 is complete when:

* this contract is reviewed and explicitly approved;
* the workframe and contract are linked from the RL documentation index;
* the implementation branch remains isolated from ``main``;
* no portfolio runtime behavior has been changed prematurely.

Any later change to the horizon, decision timing, universe, cost model, data
split, or action meanings requires a documented contract update before the
implementation is changed.
