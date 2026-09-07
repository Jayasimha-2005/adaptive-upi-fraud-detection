# Phase 1 — Leakage Audit

All column-level decisions documented below.

| Column | Action | Reason |
|--------|--------|--------|
| `V138` | **DROP** | >=80% missing in training data |
| `V139` | **DROP** | >=80% missing in training data |
| `V140` | **DROP** | >=80% missing in training data |
| `V141` | **DROP** | >=80% missing in training data |
| `V142` | **DROP** | >=80% missing in training data |
| `V143` | **DROP** | >=80% missing in training data |
| `V144` | **DROP** | >=80% missing in training data |
| `V145` | **DROP** | >=80% missing in training data |
| `V146` | **DROP** | >=80% missing in training data |
| `V147` | **DROP** | >=80% missing in training data |
| `V148` | **DROP** | >=80% missing in training data |
| `V149` | **DROP** | >=80% missing in training data |
| `V150` | **DROP** | >=80% missing in training data |
| `V151` | **DROP** | >=80% missing in training data |
| `V152` | **DROP** | >=80% missing in training data |
| `V153` | **DROP** | >=80% missing in training data |
| `V154` | **DROP** | >=80% missing in training data |
| `V155` | **DROP** | >=80% missing in training data |
| `V156` | **DROP** | >=80% missing in training data |
| `V157` | **DROP** | >=80% missing in training data |
| `V158` | **DROP** | >=80% missing in training data |
| `V159` | **DROP** | >=80% missing in training data |
| `V160` | **DROP** | >=80% missing in training data |
| `V161` | **DROP** | >=80% missing in training data |
| `V162` | **DROP** | >=80% missing in training data |
| `V163` | **DROP** | >=80% missing in training data |
| `V164` | **DROP** | >=80% missing in training data |
| `V165` | **DROP** | >=80% missing in training data |
| `V166` | **DROP** | >=80% missing in training data |
| `V322` | **DROP** | >=80% missing in training data |
| `V323` | **DROP** | >=80% missing in training data |
| `V324` | **DROP** | >=80% missing in training data |
| `V325` | **DROP** | >=80% missing in training data |
| `V326` | **DROP** | >=80% missing in training data |
| `V327` | **DROP** | >=80% missing in training data |
| `V328` | **DROP** | >=80% missing in training data |
| `V329` | **DROP** | >=80% missing in training data |
| `V330` | **DROP** | >=80% missing in training data |
| `V331` | **DROP** | >=80% missing in training data |
| `V332` | **DROP** | >=80% missing in training data |
| `V333` | **DROP** | >=80% missing in training data |
| `V334` | **DROP** | >=80% missing in training data |
| `V335` | **DROP** | >=80% missing in training data |
| `V336` | **DROP** | >=80% missing in training data |
| `V337` | **DROP** | >=80% missing in training data |
| `V338` | **DROP** | >=80% missing in training data |
| `V339` | **DROP** | >=80% missing in training data |
| `D6` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D7` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D8` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D9` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D12` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D13` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `D14` | **DROP** | >80% missing in IEEE-CIS (measured in forensic analysis) |
| `dist2` | **DROP** | 93.6% missing (measured in forensic analysis) |
| `TransactionID` | **DROP** | Identifier — not a predictive feature |
| `isFraud` | **DROP** | Target — must not appear in X |
| `TransactionDT` | **TRANSFORM** | Converted to hour_sin/cos + day_index; raw DT excluded from X to prevent temporal positional memorisation |

## Key Leakage Prevention Measures

1. `isFraud` excluded from X in all splits.
2. `TransactionID` excluded from X (identifier, not predictive feature).
3. `TransactionDT` converted to hour_sin/cos + day_index; raw excluded from X.
4. All preprocessing (imputation, encoding) fit ONLY on training data.
5. Validation and test data only transformed — never fit.
6. Threshold selected on validation; test evaluated once with frozen threshold.
7. V-columns with >80% missing dropped (measured on training data only).
8. D-columns with >80% missing dropped (D6,D7,D8,D9,D12,D13,D14).
9. dist2 dropped (93.6% missing in training).