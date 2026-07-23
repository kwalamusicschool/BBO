# Model Card: Adaptive Gaussian Process Optimiser

This model card follows the *Model Cards for Model Reporting* framework (Mitchell et al., 2019) and documents the Bayesian optimisation approach used in the BBO Capstone project.

---

## Model Details

| Field | Detail |
|-------|--------|
| **Model name** | Adaptive Gaussian Process Optimiser |
| **Version** | Round 13 – Final |
| **Model type** | Gaussian Process surrogate with Upper Confidence Bound acquisition |
| **Developed by** | Gazel |
| **Date** | July 2026 |
| **License** | Academic use only |
| **Contact** | Available via GitHub repository |

---

## Intended Use

**Suitable for:**
- Sequential maximisation of unknown black‑box functions with continuous inputs bounded between 0 and 1.
- Functions with 2 to 8 input dimensions where each evaluation is expensive or infrequent (one query per week).
- Settings where no gradient information or function structure is available.
- Problems where exploration‑exploitation balance must be managed across a fixed query budget.
- Educational settings.

**Avoid for:**
- Functions with discrete or categorical inputs – violates the continuous GP assumption.
- Problems where queries are cheap and unlimited – sample‑efficient strategies are unnecessary.
- Settings where the response surface is highly non‑stationary in ways the kernel cannot capture.
- Applications requiring real‑time decisions – GP inference scales poorly beyond ~100 points.

---

## Factors

| Factor | Relevance |
|--------|-----------|
| **Function dimensionality** | Applied to 2D through 8D functions. Performance degrades as dimensionality increases due to the curse of dimensionality. |
| **Budget** | Results depend on the query budget. With 13 rounds completed, higher‑dimensional functions remain undersampled relative to the size of their input spaces. |
| **Function smoothness** | The RBF kernel assumes smooth, continuous functions. Performance degrades on functions with discontinuities or isolated spikes. |
| **Noise level** | Some functions (F2) appear noisy. Noisy outputs reduce GP accuracy and may mislead the acquisition function. |

---

## Metrics

**Performance metrics used:**

| Metric | Description |
|--------|-------------|
| **Best‑found output** | Maximum output observed across all rounds for each function |
| **Round‑over‑round improvement** | Whether each new query produced a new best |
| **Stability** | Variance of the best point across rounds |

**Final performance summary (all 13 rounds complete):**

| Function | Initial Best | Final Best | Improvement | Best Round |
|----------|--------------|------------|-------------|------------|
| F1 | ~0 | 1.63e-12 | Signal found, unstable | 13 |
| F2 | 0.611 | 0.643 | +5.2% | 6 |
| F3 | -0.0348 | -0.0170 | +51% | 13 |
| F4 | -4.03 | 0.6496 | Turned positive | 13 |
| F5 | 1088 | 1587 | +46% | 12 |
| F6 | -0.714 | -0.404 | +43% | 7 |
| F7 | 1.365 | 2.033 | +49% | 13 |
| F8 | 9.598 | 9.980 | +4.0% | 11 |

**Key result:** Adaptive per‑function tuning lifted F4 and F7 significantly. Exploration (higher kappa) in Round 10 discovered a new peak on F5 (1587). Simple perturbations gave steady but small gains. Aggressive exploration (Round 5) failed on narrow peaks but succeeded on F4.

---

## Training Data

The GP surrogate is fitted at each round to all accumulated (input, output) pairs for each function. There is no separate training/test split – the model is updated online with each new observation. See `DATASHEET.md` for full dataset documentation.

---

## Evaluation Data

The model is evaluated implicitly through the optimisation loop: a better query choice yields a higher output value. There is no held‑out evaluation set.

---

## Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Surrogate model** | `GaussianProcessRegressor` (scikit‑learn) |
| **Kernel** | RBF (length scale tuned per function; noise parameter `alpha` raised for noisy functions) |
| **Output normalisation** | `normalize_y=True` – outputs standardised before fitting |
| **Acquisition function** | Upper Confidence Bound (UCB): `UCB(x) = mean(x) + kappa * std(x)` |
| **Candidate generation** | 20,000 random points drawn within trust region at each round |
| **Adaptive kappa schedule** | 2.5 → 1.5–2.0 → 0.8–1.0 → 0.5 |
| **Trust regions** | 0.15 → 0.05–0.10 → 0.01–0.02 → 0.01 |
| **Special case – F1** | Tiny perturbations (±0.0005) around best point after signal found; no GP used |

---

## Caveats and Recommendations

**Known limitations:**

- **Cubic scaling:** GP inference scales as O(n³). Not a constraint at current dataset sizes, but would become limiting beyond ~200 points.
- **Single‑query budget:** One query per week means a misleading result can misdirect strategy for multiple rounds.
- **Smoothness assumption:** The RBF kernel assumes smooth, continuous functions. Functions with narrow isolated peaks (F1) are not well‑modelled.
- **Local optima:** With limited data, the GP may converge to a local rather than global maximum.
- **Dimensionality:** Results for F7 (6D) and F8 (8D) should be interpreted cautiously – 50 observations provide minimal coverage of a 6D/8D space.

**Recommendations for future use:**
- For datasets larger than ~200 points, consider GPyTorch or sparse GP approximations.
- For higher‑dimensional functions, use Latin hypercube sampling or Sobol sequences for better initial coverage.
- For functions with sudden discontinuities, use fallback heuristics (centroid, perturbations) as in this approach.

---

## Ethical Considerations

**Transparency:** All decisions (model selection, hyperparameters, fallbacks) are documented per round. Sanity checks are explicit and reproducible. The full query history is preserved in the GitHub repository.

**Reproducibility:** Any peer or facilitator can replicate the optimisation trajectory from any round. Strategy decisions are documented so reasoning can be reconstructed. The adaptive nature means strategy changes are driven explicitly by observed data – the logic is auditable at each step.

**Bias and fairness:**
- Synthetic functions only – no real‑world data, human subjects, or sensitive information involved.
- Sampling bias: queries cluster around promising regions; large parts of the search space remain unsampled.
- Strategy bias: early choices influence later queries.
- Function‑specific bias: strategies that worked for one function may not generalise to others.

**Mitigations:**
- Trust regions and fallback heuristics prevent wild extrapolation.
- Per‑function hyperparameter tuning adapts to each function’s observed behaviour.
- If a model consistently fails, the approach switches to heuristics.

---

## Distribution

| Field | Detail |
|-------|--------|
| **Location** | GitHub repository |
| **Access** | Publicly accessible to course peers and facilitators |
| **Terms of use** | Imperial College programme terms; no additional licensing restrictions |
| **Maintained by** | Gazel |
| **Update process** | Manual; updates applied directly to the repository with version notes in commit history |
| **Automated pipeline** | None – all changes are managed and documented manually |

---

## Related Documents

| Document | Description |
|----------|-------------|
| **Datasheet** | Documents the query history and function evaluation dataset |
| **Project README** | Full project overview and repository structure |

---

## References

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS. arXiv:1206.2944.
- Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions*. arXiv:1012.2599.
- Srinivas, N., Krause, A., Kakade, S., & Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting*. ICML. arXiv:0912.3995.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAccT.

---

**Model card version:** Round 13 – Final. Last updated following completion of all 13 query rounds. Project complete.
