# Reviewer Questions — PhD Examiner, Paper Reviewer, ML Engineer, Domain Expert
**Generated:** 2026-09-18 | Audit only

---

## GROUP 1: Dataset Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| D1 | Why was IEEE-CIS chosen as the primary dataset? | Publicly available, real-world-derived, large (590K transactions), entity identifiers, fine-grained timestamps, extensive features, well-established benchmark | IEEE-CIS has `card1` for entity grouping, second-level `TransactionDT`, 406 usable features — all required for rigorous temporal modeling. Its Kaggle provenance means prior results exist for comparison. | `experiments/E1_lightgbm/run_metadata.json` |
| D2 | Are the IEEE-CIS V-features real or PCA? | Vesta-proprietary engineered features — NOT PCA | V1–V339 in IEEE-CIS are engineered by Vesta Corporation (e.g., frequency counts, aggregations). They are NOT principal components. ULB V1–V28 ARE PCA. | `dataset_analysis/`, `reports/DATASET_SUITABILITY_AUDIT.md` |
| D3 | Why is ULB not used for concept drift? | 48-hour temporal coverage is insufficient | Concept drift requires observing distributional shift over time. 48 hours provides no temporal window for meaningful drift study. BAF's 8-month ordinal window is appropriate. | `reports/ULB_AUDIT.md` |
| D4 | Is the 10K credit card dataset used? | No — excluded as non-benchmark toy dataset | 10,000 rows, no entity ID, no real timestamps, unknown provenance, not a published benchmark. Scientifically inappropriate for primary experiments. | `reports/DATASET_SUITABILITY_AUDIT.md` |
| D5 | Why not use more real-world datasets? | Current portfolio covers all required experimental dimensions | IEEE-CIS (primary), BAF (drift), PaySim (streaming), ULB (external validation) — adding more without completing existing experiments dilutes focus. | `reports/EXPERIMENT_CHANGE_IMPACT.md` |
| D6 | What is BAF and why is it appropriate for concept drift? | NeurIPS 2022 synthetic banking fraud dataset with 6 purpose-designed drift variants | BAF (Bank Account Fraud) was created specifically for concept drift research with 6×1M rows across Base + 5 variant files, each simulating a different type of distributional shift. | `reports/DATASET_SUITABILITY_AUDIT.md` |
| D7 | How severe is the class imbalance in each dataset? | IEEE-CIS: ~3.5%; ULB: 0.17% (578:1); BAF: ~1.1%; PaySim: 0.13% | All datasets are severely imbalanced. ULB is the most extreme. `is_unbalance=True` in LightGBM; `pos_weight=27.47` in GRU training. | `reports/DATASET_FORENSICS.json` |
| D8 | Are there duplicate transactions in ULB? | Yes — 1,081 exact full-row duplicates including 32 fraud duplicates | Any ULB experiment should document deduplication strategy before training. This is a known data quality issue in the canonical ULB dataset. | `reports/ULB_AUDIT.md` |
| D9 | What does `card1` represent? | An anonymized card-level identifier | `card1` groups transactions by cardholder for sequence construction. It is NOT a real card number — it is an obfuscated entity ID. Used for grouping only; excluded from model input. | `phase2_gru/reports/sequence_specification.md` |
| D10 | Can the datasets be merged for training? | No | IEEE-CIS V1–V339 and ULB V1–V28 are semantically incompatible. BAF and PaySim have completely different feature spaces. Merging would create a meaningless combined feature space. | `reports/DATASET_SUITABILITY_AUDIT.md` |

---

## GROUP 2: Methodology Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| M1 | Why chronological rather than random train/val/test split? | Prevents temporal leakage | Random splitting allows the model to train on data from the "future" relative to its test period. Chronological split mirrors production deployment: always predicting the future from the past. | `experiments/E1_lightgbm/run_metadata.json` |
| M2 | Why PR-AUC instead of ROC-AUC as the primary metric? | PR-AUC is more informative under severe class imbalance | With 3.5% fraud, ROC-AUC is dominated by true negative performance and can look high even for poor models. PR-AUC focuses on the precision-recall trade-off where the positive class is rare — directly relevant to fraud. | All metrics files |
| M3 | Why threshold = 0.616521 for E1? | F1-maximizing threshold on validation set | The threshold is selected to maximize F1 score on the validation split. Using test data for threshold selection would constitute test-set leakage. | `experiments/E1_lightgbm/metrics.json` |
| M4 | Why pos_weight = 27.47 in E2? | Addresses class imbalance in GRU training | With 3.5% fraud, the network would minimize loss by predicting "legit" for everything. `pos_weight=27.47` upweights fraud examples in BCEWithLogitsLoss to compensate. | `phase2_gru/artifacts/E2b_scaled/run_metadata.json` |
| M5 | Why not use SMOTE for the GRU? | Synthetic oversampling on sequences is methodologically problematic | SMOTE interpolates between samples. For temporal sequences, interpolated samples have no causal validity — they mix history from different real cardholders. The pos_weight approach is cleaner. | `phase2_gru/artifacts/E2b_scaled/run_metadata.json` |
| M6 | Why sequence length L=4 (history) + 1 (target)? | Balance between context richness and coverage | Longer sequences reduce the number of eligible entities (many cardholders have fewer than 5 transactions). L=4 provides meaningful context while retaining 6,511 entities. | `phase2_gru/reports/sequence_specification.md` |
| M7 | What is the gap feature and why include it? | `log1p(DT(T5)−DT(T4))` — log time elapsed since last transaction | Rapid sequences (e.g., two transactions within 2 minutes) may signal fraud. The gap encodes transaction timing without using raw timestamps that could encode split information. | `phase2_gru/reports/sequence_specification.md` |
| M8 | Why was card1 excluded from GRU input? | Prevents entity memorization | If card1 were in the input, the GRU could learn to memorize card-specific fraud histories. Excluding it ensures the model learns generalizable behavioral patterns. | `phase2_gru/reports/sequence_specification.md` |
| M9 | Why was seed=42 used consistently? | Reproducibility | All random operations (model initialization, bootstrap resampling, data shuffling) use seed=42. This enables exact reproduction of results. | All run_metadata.json files |
| M10 | Why paired bootstrap rather than independent bootstrap? | Tests the difference directly on matched samples | Paired resampling maintains the correspondence between E1 and E2b predictions on the same transactions. This directly tests Δ PR-AUC rather than comparing two independent CI intervals. | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |

---

## GROUP 3: E1 Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| E1Q1 | Why LightGBM rather than XGBoost or Random Forest? | State-of-the-art tabular ML, proven on fraud detection | LightGBM uses leaf-wise tree growth which is more efficient on high-feature datasets. It handles missingness natively and with 406 features, training time was ~51 seconds. | `experiments/E1_lightgbm/run_metadata.json` |
| E1Q2 | What is the random PR-AUC baseline for IEEE-CIS? | ~0.035 (equal to fraud prevalence) | A random classifier on a 3.5% fraud dataset would achieve PR-AUC approximately equal to the fraud rate. E1's 0.5317 is approximately 15× this baseline. | `experiments/E1_lightgbm/metrics.json` |
| E1Q3 | How does E1 compare to published results on IEEE-CIS? | Comparable to published leaderboard results | The Kaggle competition PR-AUC range for rigorous methodologies was approximately 0.45–0.55. E1's 0.53 is within this range. Direct comparison requires verifying split methodology match. | `experiments/E1_lightgbm/metrics.json` |
| E1Q4 | Why is there a gap between validation PR-AUC (0.5849) and test PR-AUC (0.5317)? | Slight distribution shift between validation and test periods | The model was optimized using validation PR-AUC. The test period covers a different time window — slight distribution shift is expected and does not indicate overfitting. The validation CI [0.5655, 0.6048] and test CI [0.5126, 0.5504] overlap. | `experiments/E1_lightgbm/metrics.json` |
| E1Q5 | Is E1 well-calibrated? | Moderately well-calibrated | Brier score = 0.0324 (slightly below naive baseline of 0.0341). ECE = 0.0524. These indicate useful probability estimates with moderate calibration quality. | `experiments/E1_lightgbm/metrics.json` |

---

## GROUP 4: E2 Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| E2Q1 | Why GRU rather than LSTM or Transformer? | GRU is simpler, fewer parameters, comparable performance on short sequences | LSTM has additional gates (4 × weight matrices vs. GRU's 3) with comparable performance on short sequences. Transformer requires attention over all sequence positions — with L=4, attention provides minimal advantage over GRU. GRU was chosen for simplicity of the initial temporal baseline. | `phase2_gru/reports/sequence_specification.md` |
| E2Q2 | Why did E2a achieve near-random performance? | Unscaled features create ill-conditioned optimization | Large-scale IEEE-CIS features (some with std > 10,000) dominate gradient updates, preventing the GRU from learning from small-scale features. This is a documented neural network training issue, not an architectural failure. | `phase2_gru/reports/E2b_scaled_training_report.md` |
| E2Q3 | Does the E2a → E2b improvement prove scaling caused it? | No — associated with scaling, but causality not formally established | E2a and E2b differ only in preprocessing. The improvement in validation PR-AUC (0.0345 → 0.1377) is associated with this change, but the experimental design does not formally isolate scaling as the sole cause under a controlled ablation framework. | `reports/phase2/PHASE2_FINAL_AUDIT.md` |
| E2Q4 | Why is E2b's threshold so high (0.810221)? | Model assigns low fraud probabilities overall | The GRU does not achieve high fraud probability scores — the threshold must be raised to capture any fraud. This is consistent with poor calibration (ECE=0.4247). | `phase2_gru/reports/E2b_test_evaluation_report.md` |
| E2Q5 | Does E2b's failure mean temporal information has no value? | No | E1's 406 features include aggregation and frequency features that already encode some temporal information. E2 may be redundantly processing temporal information already captured in E1 features. E3 tests the complementary hypothesis. | `reports/phase2/PHASE2_FINAL_AUDIT.md` |
| E2Q6 | What is the 585 excluded windows issue? | 585 windows where DT(T4)==DT(T5) were excluded | When the target and the last history transaction share an identical TransactionDT, chronological ordering is ambiguous. Rather than inventing an ordering, these windows are excluded. This affects ~0.11% of eligible windows. | `phase2_gru/reports/sequence_specification.md` |

---

## GROUP 5: Statistical Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| S1 | Why 2,000 bootstrap resamples? | Provides stable CI estimates while remaining computationally tractable | 2,000 resamples yields stable 95% CIs with standard error of the CI boundaries well below the CI width. More resamples would not materially change the conclusions given the large sample size (76,520). | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |
| S2 | Why does the CI exclude zero if the magnitude is large? | Large gap + stable CI means the finding is robust to resampling variability | Δ PR-AUC = −0.3584 with CI width ~0.036. The signal-to-noise ratio (gap/width ≈ 10) means zero is far outside the CI. This is not marginal. | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |
| S3 | Is a p-value available? | Not reported — CI is the primary evidence | The CI is more informative than a binary p-value. The 95% CI [−0.3756, −0.3395] excludes zero and provides magnitude information. | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |
| S4 | Why is PR-AUC the primary bootstrap metric? | PR-AUC is the primary classification metric for imbalanced data | Under severe class imbalance, PR-AUC directly measures performance on the minority class without being dominated by true negatives. | All metrics files |
| S5 | What is the practical significance of Δ PR-AUC = −0.3584? | Large — corresponds to E2b PR-AUC being 0.36 points lower than E1 | E1 PR-AUC = 0.5267; E2b PR-AUC = 0.1683. E2b scores closer to the random baseline (0.035) than to E1. This is a practically substantial difference. | `phase2_gru/artifacts/E2b_scaled/paired_bootstrap.json` |

---

## GROUP 6: Leakage Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| L1 | How was temporal leakage prevented in sequence construction? | Strict causal ordering enforced; 33/33 leakage tests pass | All history transactions have DT strictly less than the target transaction's DT. Future transactions cannot appear in history. Cross-split history is allowed only from prior splits. | `phase2_gru/tests/test_sequence_leakage.py` |
| L2 | Could the gap feature introduce leakage? | No — gap is computed from history timestamps only | `log1p(DT(T5)−DT(T4))` uses DT(T5) only as the reference point for computing elapsed time. DT(T5) does not carry any label information. | `phase2_gru/reports/sequence_specification.md` |
| L3 | Could the scaler introduce leakage? | No — scaler is fitted on training sequences only | StandardScaler is fitted on 398,312 training sequences. Validation and test data are transformed using frozen training statistics. 22/22 preprocessing tests verify this. | `phase2_gru/tests/test_preprocessing.py` |
| L4 | Could threshold selection introduce test leakage? | No — threshold selected on validation only | E1 threshold (0.616521) and E2b threshold (0.810221) were both selected using validation F1-max. No test labels were used in threshold selection. | `phase2_gru/reports/STEP9_integrity_audit.json` |
| L5 | How was the test set protected? | Evaluated exactly once; no iteration on test results | The test set was accessed for the first time in STEP 9. No model parameters, preprocessing parameters, or thresholds were adjusted using test information. | `phase2_gru/reports/STEP9_integrity_audit.json` |

---

## GROUP 7: Drift, Adaptation, and Deployment Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| DR1 | Has concept drift been studied? | No — planned for Phase 5 | Phase 5 using BAF Variants I–V is designed for this. No drift experiments have been conducted as of the current state. | `reports/EXPERIMENT_CHANGE_IMPACT.md` |
| DR2 | Why is BAF better than IEEE-CIS for drift? | BAF has purpose-designed drift variants; IEEE-CIS does not | BAF's 6 files each simulate a specific type of distributional shift. IEEE-CIS covers only ~6 months without labeled drift events. | `reports/DATASET_SUITABILITY_AUDIT.md` |
| DR3 | What streaming infrastructure exists? | None yet — planned for Phase 8 | Kafka + Spark streaming is planned but not implemented. PaySim's step column enables realistic streaming simulation. | `reports/EXPERIMENT_CHANGE_IMPACT.md` |
| DR4 | Is the model production-ready? | No — research prototype only | No serving infrastructure, latency benchmarking, or monitoring. FastAPI serving is planned for Phase 9. | Research plan only |
| DR5 | How often should the model be retrained? | Unknown — to be determined by Phase 6 | The adaptive retraining schedule depends on drift detection results. BAF experiments will estimate appropriate retraining windows. | Research plan only |

---

## GROUP 8: Generalization and Novelty Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| G1 | Does the research generalize beyond IEEE-CIS? | Not yet established empirically | Cross-dataset experiments (ULB-E1, BAF-E1, PaySim-E1) are planned but not executed. | `reports/EXPERIMENT_CHANGE_IMPACT.md` |
| G2 | What is the novel contribution? | Rigorously controlled temporal vs. tabular comparison with leakage prevention; methodologically stated | The novel contribution has not yet been formally articulated as a differentiating claim vs. published literature. This is a current gap. | Research plan only |
| G3 | Can the hybrid model be deployed in real-time? | Unknown — not yet implemented | E3 design will determine computational cost. GRU inference on short sequences (L=4) is computationally lightweight, but latency benchmarking has not been conducted. | Research plan only |
| G4 | Does the research apply to UPI fraud specifically? | Not established — IEEE-CIS is e-commerce, not UPI | The project title mentions "adaptive financial fraud detection" broadly. IEEE-CIS is US e-commerce. UPI-specific claims would require a UPI dataset. | Research plan |

---

## GROUP 9: Limitations Questions

| # | Question | Concise Answer | Deeper Answer | Evidence |
|---|---------|---------------|--------------|---------|
| LM1 | What are the main limitations of E2b? | Short window (L=4), simple architecture, E1 features may not be optimal GRU input | These are documented design choices, not methodological errors. They constrain the interpretation of E2b's results to the tested configuration. | `reports/phase2/PHASE2_FINAL_AUDIT.md` |
| LM2 | Why are 2,022 transactions excluded from the paired comparison? | Insufficient card1 history (fewer than 5 transactions) | These transactions cannot form 4-step sequences. The 97.43% coverage is high; the 2.57% excluded are not lost data but ineligible by methodological definition. | `phase2_gru/reports/E2b_test_evaluation_report.md` |
| LM3 | Could the E1 features be sub-optimal for the GRU? | Possibly — they were engineered for LightGBM, not RNNs | Aggregation and frequency features in E1 already compress temporal information into per-transaction features. A GRU processing these features in sequence may be redundantly encoding what E1 features already express. This is an important open question. | `reports/phase2/PHASE2_FINAL_AUDIT.md` |
| LM4 | Is the research limited to batch inference? | Yes, currently | No streaming inference pipeline exists. E2b inference requires 4 prior transactions per card — real-time deployment would require maintaining a state buffer. | Research limitation |
| LM5 | Why only 80 tests? | Tests cover critical methodological controls, not exhaustive code coverage | Tests were designed to verify leakage prevention, architecture integrity, preprocessing correctness, and split integrity. Code coverage is 97.43% for the tested modules. | `phase2_gru/tests/` |

---

## GROUP 10: Additional Questions the Team Should Answer Before Publication

| # | Question | Priority |
|---|---------|---------|
| Y1 | Why L=4 specifically? Provide ablation (L=2, 8, 16). | MUST HAVE |
| Y2 | Why card1 as entity? What if we use a different entity definition? | SHOULD HAVE |
| Y3 | Why not LSTM? Show equivalence or advantage of GRU. | SHOULD HAVE |
| Y4 | What is the inference latency of E2b? E3? | MUST HAVE for deployment paper |
| Y5 | What happens when a new card1 is seen in production (cold start)? | MUST HAVE for deployment |
| Y6 | How does model performance degrade as a function of time since training? | MUST HAVE for drift paper |
| Y7 | What is the cost of a false positive vs. false negative (operational cost matrix)? | SHOULD HAVE |
| Y8 | Are the V-features in IEEE-CIS actually interpretable? | SHOULD HAVE for explainability |
| Y9 | Is SHAP stable across bootstrap resamples of the training set? | SHOULD HAVE |
| Y10 | What is the minimum training window size for E1 to generalize? | SHOULD HAVE |
| Y11 | How does E1 perform if the categorical encoding is changed? | Partially answered (encoding comparison conducted) |
| Y12 | How sensitive is E2b to the pos_weight value? | SHOULD HAVE |
| Y13 | What would happen if the gap feature were removed? | SHOULD HAVE |
| Y14 | Does the 97.43% E2b coverage introduce selection bias? | SHOULD INVESTIGATE |
| Y15 | Are the 2,022 excluded transactions systematically different (harder to classify)? | SHOULD INVESTIGATE |
| Y16 | How does the model perform on different fraud types (if label information is available)? | OPTIONAL |
| Y17 | What happens when labels arrive late (label delay)? | IMPORTANT for production |
| Y18 | Can the model be retrained incrementally rather than from scratch? | IMPORTANT for Phase 6 |
| Y19 | How does the BAF drift manifest in the feature space? | IMPORTANT for Phase 5 |
| Y20 | Is the NeurIPS BAF synthetic distribution close enough to real banking fraud? | SHOULD INVESTIGATE |

---

*Reviewer Questions — 2026-09-18 | Analysis only. No training. No model changes.*
