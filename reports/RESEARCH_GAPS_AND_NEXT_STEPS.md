# Research Gaps and Next Steps
**Generated:** 2026-09-18 | Analysis only. No training.

---

## 1. Must-Have (Required for Stated Research Objectives)

### E3 Hybrid Model — IEEE-CIS
**Gap:** The central open research question — whether temporal information provides complementary signal when combined with tabular features — remains entirely unanswered.  
**Why must-have:** Without E3, the research cannot address whether a hybrid architecture improves over E1. E2b established the standalone GRU result; E3 is the natural next step in the experimental program.  
**Experiment design:** Concatenate GRU-derived temporal embedding with E1 tabular features → linear/MLP classification head. Evaluate on the same 76,520 common test population with paired bootstrap vs. E1.

### Cross-Dataset Tabular Baselines — ULB-E1, BAF-E1
**Gap:** No generalization evidence exists. All completed experiments are on IEEE-CIS only.  
**Why must-have:** A single-dataset result cannot support any generalization claim. Research papers require at least two datasets for any empirical conclusion about methodology.  
**Priority order:** ULB-E1 first (clean, no missingness, well-known benchmark), then BAF-E1 (establishes drift anchor).

### Concept Drift Experiments — BAF Variants I–V
**Gap:** The research title includes "concept drift adaptation" but zero drift experiments have been conducted.  
**Why must-have:** The stated research contributes to drift-robust fraud detection. Without drift experiments, the contribution is incomplete.  
**Experiment design:** Train on BAF Base (anchor); evaluate sequentially on Variants I–V; measure PR-AUC degradation across variants; document which drift types affect performance most.

### Adaptive Retraining — BAF
**Gap:** No adaptive retraining has been implemented or evaluated.  
**Why must-have:** "Adaptive retraining" is explicitly named in the research title.  
**Experiment design:** Monthly retraining windows on BAF; drift detection → trigger retraining → measure performance recovery.

### Explainability — SHAP on E1 and E3
**Gap:** No explainability analysis conducted.  
**Why must-have:** "Explainable AI" is explicitly named in the research title. SHAP feature importance on E1 is a near-complete analysis (model already frozen).

---

## 2. Should-Have (Strengthens Research Substantially)

### Ablation: Sequence Window Length (L = 2, 4, 8, 16)
**Gap:** Why L=4? Reviewers will ask.  
**Experiment:** Retrain E2b-equivalent with different window lengths; show PR-AUC vs. L curve.

### Ablation: GRU vs. LSTM vs. Transformer (attention over sequences)
**Gap:** Why GRU? What does LSTM add? What does attention add?  
**Experiment:** Fixed sequence construction; vary architecture only.

### Literature Comparison Table
**Gap:** No systematic comparison against published fraud detection results on IEEE-CIS.  
**Action:** Collect published PR-AUC values from recent papers on IEEE-CIS; note whether they use chronological splits and proper leakage controls.

### Streaming Implementation — Kafka + Spark
**Gap:** "Real-Time Stream Processing" is in the title; no implementation exists.  
**Implementation:** PaySim `step` column enables realistic event replay; Kafka topic ingestion; Spark Structured Streaming inference.

### PaySim-E1 Tabular Baseline
**Gap:** Mobile money domain not yet evaluated.  
**Experiment:** LightGBM on PaySim, filtering to CASH_OUT + TRANSFER types (only fraud-containing types).

---

## 3. Optional (Enhances but Not Required)

### FastAPI Serving
**Gap:** No REST endpoint for model inference.  
**Note:** Engineering task, not a research contribution. Useful for demonstration.

### Docker Containerization
**Gap:** No containerized deployment.  
**Note:** DevOps task. Does not affect research outcomes.

### Formal Latency Benchmarking
**Gap:** No inference speed measurements.  
**Note:** Relevant if deployment paper is intended; not required for research paper.

### Sensitivity Analysis
**Gap:** No sensitivity to hyperparameters documented.  
**Note:** Would strengthen robustness claims. Low priority vs. missing experiments.

---

## 4. Known Issues to Resolve Before Publication

| Issue | Type | Action Needed |
|-------|------|--------------|
| W1 RuntimeWarning in threshold code | Minor implementation | Document — no fix needed (experiment frozen) |
| W2 SMOTE/oversample fields null | Documentation | Document — no fix needed (experiment frozen) |
| W3 ROC-AUC CI rounding | Documentation | ✅ Fixed in PHASE2_FINAL_AUDIT.md |
| Novel contribution not formally stated | Research gap | Define differentiating claim vs. published literature |
| No literature comparison table | Research gap | Collect published IEEE-CIS results |
| Ablations not conducted | Research gap | Plan window length and architecture ablations |

---

## 5. Recommended Priority Order

```
IMMEDIATE (before any other new experiment):
  1. E3 Hybrid — IEEE-CIS
     ↓
  2. ULB-E1 tabular baseline
     ↓
  3. BAF-E1 tabular baseline
     ↓

NEXT PHASE:
  4. BAF Concept Drift (Variants I–V)
     ↓
  5. BAF Adaptive Retraining
     ↓
  6. Explainability — SHAP on E1 + E3
     ↓

ENGINEERING (parallel or later):
  7. Kafka + Spark streaming (PaySim)
  8. FastAPI serving
  9. Ablations (window length, architecture)
```

---

*Research Gaps and Next Steps — 2026-09-18 | Analysis only.*
