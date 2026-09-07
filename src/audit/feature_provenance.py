"""
src/audit/feature_provenance.py
Phase 1.5C — Feature provenance audit for IEEE-CIS feature groups.

PURPOSE
  Determines the temporal risk level of each IEEE-CIS feature group to assess
  whether features could encode future information not available at prediction time.

PROVENANCE TIERS (4 levels)
  KNOWN          : Temporal scope confirmed from IEEE-CIS competition documentation
                   or Vesta public statements.
  STRONGLY_INFERRED : Strong indirect evidence from feature naming, values,
                   or competition forum posts, but not formally documented.
  UNKNOWN_OPAQUE : No public documentation; feature is anonymized or proprietary.
                   Cannot rule out future-information embedding.
  POTENTIAL_LEAKAGE : Has concrete reason to suspect forward-looking computation.
                   Flag for sensitivity test before paper submission.

OUTPUT
  reports/phase1/feature_provenance_audit.md
  Printed summary with per-group risk assessment.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Provenance tier constants ─────────────────────────────────────────────────

KNOWN           = "KNOWN"
STRONGLY_INFERRED = "STRONGLY_INFERRED"
UNKNOWN_OPAQUE  = "UNKNOWN_OPAQUE"
POTENTIAL_LEAKAGE = "POTENTIAL_LEAKAGE"

# ── Per-group knowledge base ──────────────────────────────────────────────────
# Each entry: (group, tier, temporal_scope, evidence, action)
FEATURE_PROVENANCE = [
    # ── Core transaction features ─────────────────────────────────────────────
    {
        "group": "TransactionDT",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — timestamp of current transaction",
        "evidence": (
            "Documented in competition description as a timedelta from a fixed "
            "reference epoch, measured in seconds. No future information possible."
        ),
        "action": "KEEP — used for temporal features (hour_sin, hour_cos); raw DT excluded from features",
    },
    {
        "group": "TransactionAmt",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — amount of current transaction",
        "evidence": "Transaction amount is observed at the moment of transaction.",
        "action": "KEEP",
    },
    {
        "group": "ProductCD",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — product category",
        "evidence": "Five-category product code; available at transaction time.",
        "action": "KEEP",
    },
    {
        "group": "card1–card6",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — card static attributes",
        "evidence": (
            "Card number prefix, card type, issuing country/bank, and card category. "
            "These are static properties of the card, not computed aggregates. "
            "No temporal dependency."
        ),
        "action": "KEEP — primary entity key (card1) for Phase 2 sequences",
    },
    {
        "group": "addr1, addr2",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Point-in-time — billing address region (current transaction)",
        "evidence": (
            "Billing and shipping region codes. Addresses are reported per transaction "
            "and are not aggregated. However, the encoding (numeric region code) may "
            "reflect a global mapping computed offline — minor risk."
        ),
        "action": "KEEP — low risk",
    },
    {
        "group": "dist1, dist2",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Point-in-time — distance between addresses",
        "evidence": (
            "Inferred to be distance between billing and shipping addresses for the "
            "current transaction. dist2 dropped (93.6% missing). No future dependency "
            "expected, but not formally documented."
        ),
        "action": "dist1 KEPT, dist2 DROPPED (>80% missing)",
    },
    {
        "group": "P_emaildomain, R_emaildomain",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Point-in-time — email domain of purchaser and recipient",
        "evidence": (
            "Email domains are per-transaction attributes. No aggregation expected. "
            "Some forum posts confirm these are raw field values."
        ),
        "action": "KEEP",
    },

    # ── C-columns: Counting aggregates ────────────────────────────────────────
    {
        "group": "C1–C14 (counting aggregates)",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": (
            "Inferred backward-looking: cumulative count of distinct entities "
            "(addresses, emails, cards) linked to the card up to the current transaction"
        ),
        "evidence": (
            "Competition host (Vesta) described C-columns as 'counting' features. "
            "Competition forum posts suggest C1 = # addresses associated with card, "
            "C13 = # transactions on card, etc. If counts are cumulative up to (but NOT "
            "including) the current transaction, they are point-in-time safe. However, "
            "if they include the current transaction or use future data, they carry "
            "POTENTIAL_LEAKAGE. Exact window is NOT formally documented."
        ),
        "action": (
            "KEEP for E1 baseline. Flag for Phase 1.5C sensitivity test: "
            "measure model performance with C-columns removed to assess dependency. "
            "Report in paper as 'inferred backward-looking, not formally verified'."
        ),
        "tier": STRONGLY_INFERRED,  # reassign to distinguish from KNOWN
    },

    # ── D-columns: Time-delta features ───────────────────────────────────────
    {
        "group": "D1–D5, D10, D11 (time-delta, kept)",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": (
            "Inferred: days since a historical event (first card seen, last transaction, etc.) "
            "measured at the current transaction time"
        ),
        "evidence": (
            "Competition host described D-columns as 'timedelta' features. "
            "D1 in particular is widely interpreted as 'days since card was first seen'. "
            "If measured relative to current transaction's TransactionDT, they are "
            "definitionally backward-looking. NOT formally documented by Vesta."
        ),
        "action": "KEEP — but document as inferred, not verified",
    },
    {
        "group": "D6, D7, D8, D9, D12, D13, D14 (dropped)",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Inferred time-delta (same as above)",
        "evidence": "Same reasoning as D1–D5 group, but these have >80% missingness.",
        "action": "DROPPED due to >80% missingness — provenance irrelevant",
    },

    # ── M-columns: Match flags ────────────────────────────────────────────────
    {
        "group": "M1–M9 (match flags)",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Point-in-time — binary match between fields of current transaction",
        "evidence": (
            "Described as boolean match flags (e.g., 'does billing name match card name?', "
            "'does billing address match shipping address?'). These compare fields within "
            "the same transaction, so no future information is involved. T/F encoded."
        ),
        "action": "KEEP — treat as categorical after encoding",
    },

    # ── V-columns: Vesta proprietary signals ─────────────────────────────────
    {
        "group": "V1–V339 (Vesta proprietary, fully anonymized)",
        "tier": UNKNOWN_OPAQUE,
        "temporal_scope": (
            "UNKNOWN — Vesta has not publicly documented V-column construction. "
            "Some V-columns may be velocity checks (transactions per hour/day), "
            "device fingerprint risk scores, or historical pattern signals. "
            "Some may be point-in-time; others may use short look-back windows."
        ),
        "evidence": (
            "No public documentation available. Competition forums note that V-columns "
            "are 'Vesta-engineered features'. The fact that many have >80% missingness "
            "and appear in discrete bands suggests they are complex risk signals, "
            "not simple measurements. High-missingness subset (>80%) already dropped."
        ),
        "action": (
            "KEEP for E1 baseline (as Kaggle convention and prior literature does). "
            "MUST be reported as 'Unknown provenance — temporal leakage cannot be "
            "formally ruled out' in any paper. Consider V-column ablation experiment "
            "before claiming final results. Tier: UNKNOWN_OPAQUE."
        ),
    },

    # ── Identity features ─────────────────────────────────────────────────────
    {
        "group": "id_01–id_38 (anonymized device/network features)",
        "tier": UNKNOWN_OPAQUE,
        "temporal_scope": (
            "Mostly point-in-time (device/browser fingerprint at transaction time). "
            "Some id-columns may be device-level aggregates."
        ),
        "evidence": (
            "Competition documentation says identity table provides 'digital signature' "
            "information. Device type, browser, screen resolution, etc. are point-in-time. "
            "Some id-columns (id_31–id_38) appear to be numeric features with unknown "
            "construction — possibly aggregated behavioral signals."
        ),
        "action": "KEEP — but id_31–id_38 flagged as UNKNOWN_OPAQUE for paper documentation",
    },
    {
        "group": "DeviceType, DeviceInfo",
        "tier": STRONGLY_INFERRED,
        "temporal_scope": "Point-in-time — device type and model at transaction time",
        "evidence": (
            "DeviceType (mobile/desktop) and DeviceInfo (device model string) are "
            "clearly per-transaction device attributes. No temporal dependency."
        ),
        "action": "KEEP",
    },

    # ── Engineered features (our construction) ────────────────────────────────
    {
        "group": "hour_sin, hour_cos (engineered)",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — cyclical encoding of transaction hour",
        "evidence": (
            "Computed from TransactionDT by our pipeline. "
            "hour = (TransactionDT // 3600) % 24. "
            "Strictly point-in-time; no future information."
        ),
        "action": "KEEP",
    },
    {
        "group": "is_missing_* (engineered missingness flags)",
        "tier": KNOWN,
        "temporal_scope": "Point-in-time — binary flag indicating field is missing",
        "evidence": (
            "Computed from per-transaction missingness pattern. "
            "Whether a field is missing is determined at transaction time. "
            "Missingness indicators fit only on training data (fit on column-level "
            "presence/absence — no statistics required). No leakage."
        ),
        "action": "KEEP",
    },
]


def _tier_emoji(tier: str) -> str:
    return {
        KNOWN: "✅",
        STRONGLY_INFERRED: "🟡",
        UNKNOWN_OPAQUE: "🟠",
        POTENTIAL_LEAKAGE: "🔴",
    }.get(tier, "❓")


def run_provenance_audit(report_dir: Path) -> list[dict]:
    """
    Execute the feature provenance audit and write the markdown report.

    Parameters
    ----------
    report_dir : Directory where feature_provenance_audit.md is written.

    Returns
    -------
    List of feature provenance entries (raw data).
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "feature_provenance_audit.md"

    # Summary counts
    tier_counts = {KNOWN: 0, STRONGLY_INFERRED: 0, UNKNOWN_OPAQUE: 0, POTENTIAL_LEAKAGE: 0}
    for entry in FEATURE_PROVENANCE:
        tier_counts[entry["tier"]] = tier_counts.get(entry["tier"], 0) + 1

    logger.info("Feature provenance audit: %d feature groups assessed", len(FEATURE_PROVENANCE))
    for tier, count in tier_counts.items():
        logger.info("  %s %s: %d groups", _tier_emoji(tier), tier, count)

    lines = [
        "# IEEE-CIS Feature Provenance Audit",
        "## Phase 1.5C — Temporal Leakage Risk Assessment",
        "",
        "**Experiment:** E1_lightgbm | **Dataset:** IEEE-CIS Fraud Detection (Vesta/Kaggle 2019)",
        "**Status:** Additive audit — E1 model is IMMUTABLE",
        "",
        "## Provenance Tiers",
        "",
        "| Tier | Icon | Meaning |",
        "|------|------|---------|",
        f"| KNOWN | ✅ | Temporal scope confirmed from competition documentation or Vesta statements |",
        f"| STRONGLY_INFERRED | 🟡 | Strong indirect evidence but not formally documented |",
        f"| UNKNOWN_OPAQUE | 🟠 | No public documentation; cannot rule out future-information embedding |",
        f"| POTENTIAL_LEAKAGE | 🔴 | Concrete reason to suspect forward-looking computation |",
        "",
        "## Tier Summary",
        "",
        "| Tier | Feature Groups |",
        "|------|--------------|",
    ]
    for tier, count in tier_counts.items():
        lines.append(f"| {_tier_emoji(tier)} {tier} | {count} |")

    lines += [
        "",
        "> [!IMPORTANT]",
        "> V1–V339 and some id-columns are classified as UNKNOWN_OPAQUE. Any paper or report",
        "> using these features must explicitly state that their temporal scope is unverified",
        "> and that temporal leakage cannot be formally ruled out without Vesta's internal documentation.",
        "",
        "---",
        "",
        "## Detailed Provenance by Feature Group",
        "",
    ]

    for i, entry in enumerate(FEATURE_PROVENANCE, 1):
        icon = _tier_emoji(entry["tier"])
        lines += [
            f"### {i}. {entry['group']}",
            f"**Tier:** {icon} `{entry['tier']}`  ",
            f"**Temporal Scope:** {entry['temporal_scope']}  ",
            f"**Evidence:** {entry['evidence']}  ",
            f"**Action:** {entry['action']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Recommendations for Paper Writing",
        "",
        "1. **V1–V339:** State explicitly: *'Vesta proprietary signals with unknown construction. "
        "Temporal leakage cannot be formally ruled out. Used as per Kaggle competition convention "
        "and prior literature.'*",
        "",
        "2. **C1–C14:** State: *'Inferred backward-looking counting aggregates. "
        "Exact temporal scope is not formally documented. A V-column and C-column ablation "
        "experiment is planned to assess sensitivity.'*",
        "",
        "3. **D1:** State: *'Inferred time-delta feature. Lower D1 is strongly correlated "
        "with fraudulent transactions (fraud mean: 38.7 days vs. legitimate mean: 95.6 days). "
        "Causal interpretation is a hypothesis, not verified ground truth.'*",
        "",
        "4. **id_31–id_38:** State: *'Anonymized features from identity table with unknown construction. "
        "Classified as UNKNOWN_OPAQUE.'*",
        "",
        "*Generated by: src/audit/feature_provenance.py | Phase 1.5C*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Feature provenance audit written: %s", out_path)
    return FEATURE_PROVENANCE


if __name__ == "__main__":
    get_logger(__name__)
    report_dir = _ROOT / "reports" / "phase1"
    run_provenance_audit(report_dir)
    print(f"\nAudit complete. Report: {report_dir / 'feature_provenance_audit.md'}")
