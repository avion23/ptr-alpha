# Trading prediction architecture and validation

## Status

This document is the architectural contract for predictive work in PTR Alpha.
It separates four things that must not be conflated:

1. reconstructing what the market could have known at a historical instant;
2. estimating a conditional distribution of future net returns;
3. converting that distribution into a portfolio decision; and
4. deciding whether the complete research process produced evidence that can be
   used outside the sample.

The current repository is strongest at point-in-time replay and fail-closed
validation. Its main weakness is not a missing optimizer or a missing neural
network. It is that prediction experiments are spread across more than one
harness, while member effects and strategy-family dependence have historically
used stronger assumptions than the data justify.

## 1. Current system at a high level

```text
Official House filings          Capitol Trades fallback
          |                              |
          +---------- ingestion --------+
                         |
                 parser cascade / OCR
                         |
            canonical transaction records
                         |
                 DuckDB repositories
                         |
       +-----------------+------------------+
       |                                    |
point-in-time labels/features       operational analysis
       |                                    |
member ranking + ticker scoring       CLI / reports
       |
walk-forward backtest
       |
portfolio simulation
       |
multiple-testing correction
       |
locked retrospective/final evaluation
```

The major components are:

| Component | Responsibility | Primary modules |
| --- | --- | --- |
| Ingestion | Download disclosures and optional external records | `download.py`, `capitol_trades.py` |
| Parsing | Convert heterogeneous PDFs into normalized rows | `parser_cascade.py`, `parsing/` |
| Persistence | Store transactions, prices, metadata, parse runs, and provenance | `database.py`, repository modules |
| Point-in-time data | Build entry prices, feature histories, and completed forward outcomes | `price_snapshot.py`, `signals/`, `pipeline.py` |
| Descriptive member analysis | Estimate hit rates, alpha, and partially pooled member effects | `member_ranking/` |
| Candidate generation | Score recent multi-buyer ticker events | `member_ranking/buyer_scoring.py` |
| Replay | Generate recommendations and evaluate realizable historical outcomes | `backtest/`, `portfolio/`, `portfolio_sim.py` |
| Statistical validation | Purge horizons, preserve scheduled support, correct search, and consume evaluations once | `validation.py`, `snooping.py` |
| Retrospective optimization | A second locked selection/retrospective/final workflow | `optimize_profit/` |
| Presentation | User input and formatting only | `cli.py`, reporting modules |

## 2. The problem from first principles

At public time `t`, PTR Alpha observes a delayed and lossy disclosure event:

```text
private execution time ---- disclosure delay ----> public filing time t
       unknown to us                                 tradable information set
```

The trading problem is:

```text
Given only information public by t,
estimate the distribution of executable net return over horizon h,
then choose a capital-constrained action that maximizes predeclared utility.
```

A useful mathematical boundary is:

```text
Forecast:
    p(r_net | public_snapshot_t, horizon_h)

Policy:
    action_t = pi(forecast, prices_t, costs_t, risk_state_t, capital_t)

Evaluation:
    utility(action_t, realized_r_net)
```

This is **not primarily a member leaderboard problem**. Member identity is one
possible feature. The deployable object is a forecast for a public event and a
policy acting on that forecast.

The observations have difficult structure:

- disclosures are delayed and occasionally corrected;
- transaction values are intervals, not exact amounts;
- labels mature after long and overlapping horizons;
- the same member, ticker, sector, and date create dependent observations;
- member histories are sparse and highly unequal in size;
- market regimes and legislative relationships change;
- parser failures and symbol resolution create measurement error;
- trying many features, horizons, scorers, and policies creates a strategy
  family even when the trials were chosen manually.

No algorithm removes these constraints. More model capacity can make selection
bias worse when the effective sample is small.

## 3. Known computer-science and statistical formulations

The problem is a composition of solved problem classes, not one novel monolith.

### 3.1 Event sourcing and point-in-time feature stores

A historical replay must reconstruct the exact public information set. This is
an event-sourcing problem with bitemporal semantics:

```text
valid time       = when the underlying trade occurred
knowledge time   = when the record became publicly observable
```

Trading features must be indexed by knowledge time. Corrections must create a
new observable event rather than rewriting what an earlier replay supposedly
knew.

### 3.2 Delayed supervised forecasting

Outcomes arrive after a horizon. This is supervised learning with delayed
feedback and censoring. Rows whose horizon has not completed are unlabeled, not
negative and not shorter-horizon substitutes.

Because PTR Alpha does not control which congressional trades are disclosed and
its actions do not materially change the data-generating process, reinforcement
learning is unnecessary. A probabilistic forecaster plus a separate decision
policy is smaller, easier to test, and statistically more efficient.

### 3.3 Hierarchical panel estimation

Member, ticker, sector, and time effects form a sparse panel. Partial pooling is
the standard solution: noisy entities shrink toward a population or subgroup
mean, while repeated evidence permits separation. A static normal-normal model
is a reasonable baseline; a dynamic multilevel model is the natural extension.

### 3.4 Forecast comparison under data snooping

Searching configurations and then reporting the winner is multiple hypothesis
testing. The search algorithm changes computational efficiency, not the need
for an untouched outer evaluation. Reality Check, Superior Predictive Ability,
step-down procedures, model-confidence sets, block bootstrap, and explicit
family-wise bounds are established solutions.

### 3.5 Decision theory and portfolio construction

A high predicted return is not yet a trade. Costs, uncertainty, correlation,
turnover, exposure, concentration, and available capital belong in a policy
layer. The forecast should not silently encode one portfolio construction rule.

## 4. What the repository already gets right

The following properties should be preserved:

- public disclosure time, not private transaction time, drives availability;
- incomplete forward windows remain missing;
- recommendation replay calls production analysis code rather than a separate
  approximation;
- no-trade dates remain in scheduled support as cash observations;
- strategy and benchmark use identical dates;
- the primary statistic is one per-date net-alpha series rather than a mixture
  of incompatible metrics;
- production selection is fail-closed when no corrected survivor exists;
- the consensus scorer is identity-invariant by construction;
- member-skill modes are descriptive and cannot authorize deployment;
- retrospective history is labeled as reused history, not fresh evidence;
- final evaluation is locked and consumed through an auditable ledger.

These are more valuable than replacing grid search with a fashionable optimizer.

## 5. Principles that have been violated

### 5.1 One experiment, one canonical harness

`analyzer.validation` and `optimize_profit` both implement selection,
retrospective evaluation, null diagnostics, support checks, manifests, and final
locking. Even when each is individually careful, two authorities create:

- divergent definitions of return, support, costs, and significance;
- an ambiguous total number of tried strategies;
- duplicated bug fixes;
- a path for choosing whichever report looks better.

The repository needs one experiment engine and multiple declarative experiment
specifications, not multiple engines.

### 5.2 Search is not evidence

Hill climbing, Bayesian optimization, random search, grid search, and manual
iteration all optimize a noisy historical criterion. None validates the chosen
configuration. Every observed trial, failed run, early-stopped run, and
human-directed follow-up belongs to the effective research family.

For the current small, mostly categorical grid, exhaustive search is preferable:
it is transparent, reproducible, and cheap. Bayesian optimization becomes useful
only when the trial space is materially larger or each inner evaluation is
expensive. It must remain inside the training phase.

### 5.3 Partial pooling must permit complete pooling

The previous empirical normal-normal helper forced estimated between-member
variance to be at least the within-member residual variance. That manufactures
population heterogeneity when the observed spread of member means is explainable
by sampling noise. The consequence is too little shrinkage of sparse member
histories. The corrected estimator clips the method-of-moments variance at a
numerical floor only.

This does not make member alpha causal or deployable. It makes the descriptive
baseline internally coherent.

### 5.4 Resampling must preserve actual support

The previous max-stat bootstrap synchronized trials through shared ordinal
uniforms even when their observation calendars differed. Row 12 of a 30-day
schedule is not necessarily contemporaneous with row 12 of a 60-day schedule.
The corrected bootstrap now:

1. groups trials by exact post-missing-value calendar support;
2. uses common circular block starts only inside a support group; and
3. applies a conservative Bonferroni bound across different support groups.

The arbitrary-dependence marginal Bonferroni gate remains controlling.

### 5.5 A benchmark is not a risk model

SPY-relative return removes one market component but does not isolate the event
signal. Congressional portfolios can load on technology, size, momentum,
volatility, and sector regimes. A member who repeatedly buys a concentrated
factor is not necessarily demonstrating stock-selection skill.

The prediction target should be an executable return net of costs and, for
research diagnostics, residualized against predeclared market/sector/factor
controls. Raw return, SPY alpha, and factor-residual alpha should be retained as
separate fields rather than substituted after results are known.

### 5.6 Endpoint labels throw away information

A binary `outperformed / did not outperform` label discards magnitude and makes
results threshold-dependent. The primary model should forecast a continuous net
return distribution. `P(net alpha > 0)` and downside probabilities are derived
outputs. Binary classification remains useful as a secondary calibrated view.

## 6. Target architecture

```text
                    immutable data plane

RawDocument -> ParseObservation -> CanonicalDisclosureEvent
                                      |
                                      v
                            PointInTimeSnapshot
                            /                 \
                    FeatureFrame          MaturedLabelFrame
                            \                 /
                             TrainingDataset

                    pure prediction plane

TrainingDataset -> ForecastModel.fit() -> FrozenForecastModel
PointInTimeSnapshot ------------------> ForecastFrame

                    imperative decision plane

ForecastFrame + prices + costs + risk state -> Policy -> TargetPortfolio
TargetPortfolio + execution assumptions     -> Replay / live adapter

                    independent evidence plane

ExperimentSpec -> InnerSearch -> FrozenCandidate -> OuterEvaluation
      |               |                |                 |
      +---------- append-only TrialLedger --------------+
                                      |
                               LockedFinalEvaluation
```

### 6.1 Minimal interfaces

The architectural boundary should be small and typed:

```python
@dataclass(frozen=True, slots=True)
class Forecast:
    event_id: str
    as_of: datetime
    horizon_days: int
    expected_net_alpha: float
    net_alpha_std: float
    probability_positive: float
    model_id: str
    feature_snapshot_sha256: str

class ForecastModel(Protocol):
    def fit(self, dataset: TrainingDataset) -> FrozenForecastModel: ...

class FrozenForecastModel(Protocol):
    def predict(self, snapshot: PointInTimeSnapshot) -> tuple[Forecast, ...]: ...

class Policy(Protocol):
    def allocate(
        self,
        forecasts: tuple[Forecast, ...],
        market: MarketSnapshot,
        state: PortfolioState,
    ) -> TargetPortfolio: ...
```

Models do not query a database, fetch prices, print, size positions, or know the
final test period. The shell supplies immutable snapshots. This is the
functional-core / imperative-shell boundary.

### 6.2 Canonical experiment specification

Every search should be represented by one immutable `ExperimentSpec` containing:

- data snapshot and source hashes;
- public-time feature schema version;
- label and executable-entry definition;
- model family and hyperparameter domain;
- optimizer, seed, and maximum trial budget;
- all train/validation/final windows and embargoes;
- transaction-cost and portfolio policy version;
- one primary selection utility;
- null controls and release thresholds.

The trial ledger must record suggested, started, failed, pruned, completed, and
manually added configurations. A Bayesian optimizer is then only an alternate
producer of `TrialSpec` records.

## 7. Recommended prediction stack

### Layer 0: identity-free consensus baseline

Keep the current distinct-buyer recency score. It is cheap, deterministic,
interpretable, invariant to member-name permutation, and an excellent canary.
Every more complex model must beat it on identical support after costs.

### Layer 1: dynamic hierarchical model

Use a robust multilevel return model as the primary member-aware baseline:

```text
net_alpha_i = time_effect_t
            + sector_effect[s_i, t]
            + member_effect[m_i, t]
            + member_sector_effect[m_i, s_i]
            + observable_event_features_i * beta
            + StudentT_noise_i
```

Recommended properties:

- member and member-sector effects are partially pooled;
- member effects evolve slowly through a random walk or discount factor;
- residuals are heavy-tailed;
- same-date and same-ticker observations are clustered in evaluation;
- missing or immature outcomes never enter the likelihood;
- posterior uncertainty is exported with the mean;
- the model is fitted independently in every walk-forward fold.

Do not rank by posterior mean alone. A policy can use expected net alpha,
probability of positive net alpha, and a downside-aware lower bound.

### Layer 2: regularized tabular nonlinear model

For the present data scale, add one strong tree baseline before deep learning:
XGBoost, LightGBM, CatBoost, or a dependency-light histogram gradient booster.
Use continuous net alpha or a distributional objective, with fold-local
calibration for any probabilities.

Candidate public-time features:

- number of distinct recent buyers and concentration;
- recency and disclosure lag;
- transaction-value interval features, not a false exact amount;
- owner and instrument type;
- repeat-buy and cross-member episode structure;
- ticker, sector, and factor state known at filing time;
- momentum, volatility, liquidity, and gap since prior disclosure;
- committee, lobbying, campaign-finance, and district-industry links only after
  a bitemporal provenance layer exists.

The model should consume member posterior summaries as fold-local features only,
never full-history rankings.

### Layer 3: out-of-fold ensemble

Combine consensus, hierarchical, and tree forecasts only from outer-training
out-of-fold predictions. A simple non-negative linear stack is preferable to a
large meta-model. The stack is another family member and must be recorded before
the outer evaluation.

### Layer 4: temporal graph model, research only

The relational problem is naturally a temporal heterogeneous graph. A 2026
preprint applies a latency-aware Temporal Graph Network to congressional trades,
lobbying, campaign finance, and geographic links. It is relevant research, but
its reported AUROC is roughly random and its XGBoost baseline has slightly
higher AUROC at the stated long horizons; the graph model mainly improves F1.
That is not yet evidence of tradable economic value.

A graph model belongs after the bitemporal relationship data, strong tabular
baselines, probability calibration, and economic replay exist. It should be a
plugin implementing the same `ForecastModel` interface, not a new pipeline.

## 8. How to use hill climbing and Bayesian optimization correctly

### Hill climbing

Hill climbing is a poor default here because the objective is noisy,
non-convex, categorical, and path-dependent. It can be useful for a strictly
local sensitivity check around a frozen configuration, but not for proving the
configuration is good.

### Bayesian optimization

Bayesian optimization is appropriate when:

- one fold-complete trial is expensive;
- the domain has several continuous dimensions;
- the budget is fixed in advance; and
- the optimizer sees only inner-training results.

It must not see the retrospective or final phase. Acquisition-function choices,
optimizer seeds, restarts, and human restarts are trial-family decisions.

### Multi-fidelity methods

Hyperband/BOHB-style pruning is risky for trading backtests because early time
windows are not generally monotone proxies for full-history performance. If
used, resource must mean a predeclared prefix or number of folds; pruned trials
stay in the ledger; and surviving configurations are rerun at full budget before
selection.

### Preferred order for this repository

```text
small discrete family  -> exhaustive grid
larger sparse family   -> seeded random/TPE search
expensive continuous   -> Bayesian optimization
all cases              -> same outer walk-forward evaluation and family ledger
```

## 9. Validation protocol

```text
outer fold k

past only ------------------------------------------------------ future
| inner fit/search | purge/embargo | outer score |
                                      ^ optimizer never sees this
```

Required protocol:

1. Freeze the data snapshot, feature schema, label, costs, primary utility, and
   trial budget.
2. Run optimizer trials only inside the outer-training window.
3. Refit the selected configuration from scratch on allowed outer-training data.
4. Score one scheduled per-date series on the outer fold.
5. Aggregate outer-fold predictions and economic results.
6. Apply block/dependence-aware family correction to the complete search.
7. Compare against cash, SPY, consensus, and shuffled/no-information canaries on
   identical support.
8. Lock one candidate and consume the final period once.

For comparing a large model family, Hansen's Superior Predictive Ability test or
a step-down/model-confidence-set procedure is a less blunt research diagnostic
than Bonferroni. Bonferroni should remain the fail-safe release gate until the
more powerful procedure is implemented and tested against adversarial canaries.

Metrics must be separated by purpose:

| Purpose | Metrics |
| --- | --- |
| Probability quality | log loss, Brier score, calibration slope/intercept |
| Distribution quality | CRPS or pinball losses, interval coverage |
| Ranking | rank correlation, precision at a predeclared capacity |
| Economics | net alpha, utility, turnover, drawdown, exposure, capacity |
| Evidence | corrected p-value, null percentile, support and trial counts |

A single predeclared economic utility chooses the model. The rest are diagnostics.

## 10. Testability

Every proposed boundary is directly testable.

### Unit and property tests

- adding future rows cannot change an earlier snapshot or forecast;
- permuting member labels leaves the consensus model unchanged;
- duplicating one disclosure cannot create another distinct buyer;
- scaling all return observations scales posterior means and standard deviations
  but not shrinkage;
- when between-member spread is explainable by noise, partial pooling approaches
  complete pooling;
- shifted calendars are not treated as contemporaneous bootstrap support;
- forecast and policy functions are deterministic for a frozen snapshot and seed.

### Leakage canaries

- inject an impossibly predictive future-only feature and assert the feature
  builder refuses or the point-in-time join leaves it missing;
- move disclosure timestamps after the decision time and assert predictions do
  not use those rows;
- truncate price history before label maturity and assert the outcome is missing;
- rename all members and assert identity-free production output is unchanged.

### Statistical canaries

- all-zero alpha must never deploy;
- shuffled event-to-outcome alignment must fail;
- a synthetic planted effect must be detected at adequate sample size;
- unsupported block lengths and duplicate calendar observations must fail closed;
- adding null strategies to the family cannot improve corrected significance.

### Integration tests

One synthetic DuckDB fixture should drive the real ingestion-independent path:
point-in-time snapshot, feature generation, model fit, forecast, policy, replay,
and ledger. No duplicate implementation of production formulas is allowed in
tests.

## 11. Prioritized implementation plan

### P0: preserve evidence integrity

- Keep the corrected calendar-aware bootstrap and empirical-Bayes variance.
- Keep one documented verification command and run it before every direct
  update of `main`.
- Make every future model/search change add a trial-ledger schema migration or
  explicitly prove that no schema change is required.

### P1: remove duplicate experiment authority

Extract one package, for example `analyzer.experiments`, containing:

```text
spec.py       immutable ExperimentSpec / TrialSpec
search.py     grid, random, Bayesian trial producers
runner.py     canonical fold execution
inference.py  family correction and model comparison
ledger.py     append-only manifests and final consumption
```

Migrate `validation.py` first. Convert `optimize_profit` into an experiment spec
and reporting command, then delete its duplicate statistical engine.

### P2: establish forecast and policy protocols

Wrap the current consensus scorer behind `ForecastModel`; move allocation and
portfolio constraints behind `Policy`. Preserve current output byte-for-byte in
an adapter test before adding models.

### P3: add dynamic hierarchy and tree baseline

Implement both against the same immutable `TrainingDataset`. Predeclare a small
search family and compare out-of-fold forecasts, not in-sample member rankings.

### P4: enrich relationships

Add point-in-time committee, lobbying, donation, and district-industry edges.
First expose them to the tree model. Attempt a temporal graph model only when the
simpler model proves those features add stable outer-fold value.

## 12. References

- Cawley, G. C. and Talbot, N. L. C. (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
- Bailey, D. H. et al. (2015), [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).
- Hansen, P. R. (2005), [A Test for Superior Predictive Ability](https://doi.org/10.1198/073500105000000063).
- Gneiting, T. and Raftery, A. E. (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).
- Snoek, J., Larochelle, H., and Adams, R. P. (2012), [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html).
- Li, L. et al. (2018), [Hyperband](https://www.jmlr.org/papers/v18/16-558.html).
- Falkner, S., Klein, A., and Hutter, F. (2018), [BOHB](https://proceedings.mlr.press/v80/falkner18a.html).
- Gu, S., Kelly, B., and Xiu, D. (2020), [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009).
- Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022), [Why do tree-based models still outperform deep learning on typical tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).
- Rossi, E. et al. (2020), [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637).
- Roodman, B. P. et al. (2026), [Detecting Information Channels in Congressional Trading via Temporal Graph Learning](https://arxiv.org/abs/2602.05514).
