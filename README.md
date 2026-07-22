## Black-Box Optimization Capstone

## Section 1: Project Overview

This project is a Bayesian optimization challenge where the goal is to maximize the output of eight unknown (black-box) functions through a series of iterative queries. Each function accepts a fixed-dimensional input vector and returns a single scalar output. The function internals are hidden; only input-output pairs are observable.

The challenge simulates real-world scenarios where function evaluations are expensive, slow, or otherwise limited, and where the objective structure is unknown. Examples include hyperparameter tuning, drug candidate screening, industrial process optimization, and materials science experiments. In all these settings, a practitioner cannot evaluate every configuration and must make intelligent, data-driven decisions about where to query next.

The skills developed here transfer directly to applied ML roles: reasoning under uncertainty, building surrogate models, balancing exploration and exploitation, and revising strategy based on new evidence. These are central to any data science role.

## Section 2: Inputs and Outputs

Each of the eight functions takes a continuous input vector with values constrained to [0,1] and returns a single scalar output. Dimensionality ranges from 2D to 8D.

Input format: x1-x2-x3-...-xn
Each value must begin with 0 and be specified to six decimal places.

## Examples:

| Function | Example Query |
|----------|---------------|
| F1 (2D) | `0.445557-0.302343` |
| F2 (2D) | `0.703750-0.927693` |
| F3 (3D) | `0.505391-0.581772-0.369623` |
| F4 (4D) | `0.370669-0.409253-0.350881-0.421815` |
| F5 (4D) | `0.279987-0.927602-0.818844-0.938221` |
| F6 (5D) | `0.674119-0.288605-0.668024-0.702245-0.175134` |
| F7 (6D) | `0.241985-0.410284-0.295099-0.277951-0.413597-0.671281` |
| F8 (8D) | `0.116930-0.191051-0.114310-0.234380-0.744811-0.447932-0.147187-0.658030` |

Output: A single floating-point scalar returned by the portal after each submission. This value is used to update the surrogate model.

## Function reference table:

| Function | Dims | Initial Points | Simulated Domain |
|----------|------|----------------|-------------------|
| F1 | 2 | 10 | Radiation source detection |
| F2 | 2 | 10 | Noisy ML log-likelihood surface |
| F3 | 3 | 15 | Drug discovery – adverse reaction minimisation |
| F4 | 4 | 30 | Warehouse placement optimisation |
| F5 | 4 | 20 | Chemical process yield |
| F6 | 5 | 20 | Cake recipe multi-objective scoring |
| F7 | 6 | 30 | ML model hyperparameter tuning |
| F8 | 8 | 40 | High-dimensional model performance |

## Section 3: Challenge Objectives

The goal is maximisation for every function: find the input vector that produces the highest possible output value. Some functions are naturally minimisation problems (e.g., drug side effects, cake recipe cost), but the returned output is transformed so that higher is always better.

## Constraints and limitations:

One query per function per week. Evaluations are expensive by design.
No access to function internals. No gradients, no structure, no closed-form expression.
Unknown noise levels. Some functions are noisy (F2), complicating exploitation.
Unknown number of optima. Functions may be unimodal or multimodal.
Delayed feedback. Results return weekly – no rapid iteration.
Dimensionality varies. Higher-dimensional functions (F7, F8) require more data and are more susceptible to the curse of dimensionality.
Section 4: Technical Approach

## Surrogate Model

I used a Gaussian Process (GP) regressor with an RBF kernel, implemented via scikit-learn's GaussianProcessRegressor. The GP is fitted to all available input-output pairs for each function and produces a predicted mean and uncertainty estimate for any candidate point. The kernel's length-scale and noise parameters were tuned per function via grid search and LOOCV in early rounds, and later set manually based on observed behaviour.

## Acquisition Function

The Upper Confidence Bound (UCB) acquisition function was used:

UCB(x) = mean(x) + kappa * std(x)
where mean(x) is the GP predicted mean, std(x) is the predicted standard deviation, and kappa controls exploration-exploitation trade-off. A pool of 20,000 random candidates was generated within a trust region around the current best point, and the highest-scoring point was submitted.

## Adaptive Kappa and Trust Regions

kappa and trust radius were tuned per function based on observed performance:

kappa	Trust radius	When applied
2.5	0.15	Early exploration (Rounds 1–2)
1.5–2.0	0.05–0.10	Balanced cases
0.8–1.0	0.01–0.02	Functions with steady improvement
0.5	0.01	Functions with confirmed stable best (final exploitation)
For functions that regressed or showed noise (F2), I raised the GP's noise parameter (alpha=0.1) to smooth through scatter. For functions where the search direction became clear (F5, F7), I moved them to heuristic perturbations instead of GP-UCB.

## Sanity Checks

Every query was validated against:

Trust region – rejected if more than radius away from the best point.
Boundary clipping – clamped to [0,1].
Nearest neighbour output – warned if very close to a known poor output.
Special Case: Function 1

Function 1 returned near-zero outputs for most rounds. The GP had no signal to learn from. I switched to tiny perturbations (±0.0005) around the best known point once a positive signal was found (Round 2), and later repeated the best point when the signal was lost.

## Progress After Thirteen Rounds

| Function | Initial Best | Final Best | Best Round | Improvement |
|----------|--------------|------------|------------|-------------|
| F1 | ~0 | 1.63e-12 | 13 | Signal found, unstable |
| F2 | 0.611 | 0.643 | 6 | +5.2% |
| F3 | -0.0348 | -0.0170 | 13 | +51% |
| F4 | -4.03 | 0.6496 | 13 | Turned positive |
| F5 | 1088 | 1587 | 12 | +46% |
| F6 | -0.714 | -0.404 | 7 | +43% |
| F7 | 1.365 | 2.033 | 13 | +49% |
| F8 | 9.598 | 9.980 | 11 | +4.0% |

## Considered Alternatives

LOOCV model selection – used in Rounds 1–3 to choose between Ridge, KNN, RandomForest, SVR, and GP. Dropped after SVR overfit on F4.
Voronoi space-filling – used for F1 in Round 2; found the first positive signal.
LHS exploration – used in Round 5; failed on F2, F6, F7 and abandoned.
Midpoint heuristic – used on F3 in Round 11; failed and dropped.
Weighted centroid – used as fallback for F6 when GP failed.

## Repository Structure

```
bbo-capstone/
├── README.md
├── datasheet.md
├── model-card.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── initial/
│   │   ├── initial_inputs_F1.npy
│   │   ├── initial_inputs_F2.npy
│   │   ├── initial_inputs_F3.npy
│   │   ├── initial_inputs_F4.npy
│   │   ├── initial_inputs_F5.npy
│   │   ├── initial_inputs_F6.npy
│   │   ├── initial_inputs_F7.npy
│   │   ├── initial_inputs_F8.npy
│   │   ├── initial_outputs_F1.npy
│   │   ├── initial_outputs_F2.npy
│   │   ├── initial_outputs_F3.npy
│   │   ├── initial_outputs_F4.npy
│   │   ├── initial_outputs_F5.npy
│   │   ├── initial_outputs_F6.npy
│   │   ├── initial_outputs_F7.npy
│   │   └── initial_outputs_F8.npy
│   └── results/
│       ├── all_results.csv
│       ├── raw_round_1.txt
│       └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_round1_query_generation.ipynb
│   ├── 03_round2_query_generation.ipynb
│   ├── ...
│   └── 13_round13_query_generation.ipynb
├── src/
│   ├── gp_utils.py
│   ├── acquisition.py
│   ├── heuristics.py
│   └── sanity_checks.py
└── reports/
    ├── figures/
    │   ├── progress_all_functions.png
    │   └── peer_comparison.png
    └── final_report.md
```
