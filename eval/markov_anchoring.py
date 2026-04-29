"""
Marginal anchoring analysis (using actual first_vs_last.csv).

Approach:
    1. Reconstruct pi_0 (initial right/wrong distribution) from first_vs_last.csv:
           pi_0[R] = rateRR + rateRW    (row marginal: by first answer)
           pi_0[W] = rateWR + rateWW
    2. Reconstruct empirical final distribution pi_T_emp:
           pi_T_emp[R] = rateRR + rateWR    (col marginal: by last answer)
           pi_T_emp[W] = rateRW + rateWW
    3. Build the per-turn 2x2 transition matrix P from turn_by_turn.csv
       (joint -> conditional via row-marginal normalization).
    4. Predict the final distribution under a memoryless Markov assumption:
           pi_T_pred = pi_0 @ P^6
    5. Compare pi_T_pred to pi_T_emp.

Two views of the gap:
    anchor_gap_R = emp_piT_R - pred_piT_R
        > 0 => model retained "right" more than Markov predicts (anchoring)
        < 0 => model drifted to wrong more than Markov predicts (compounding sycophancy)

    Total Variation distance between predicted and empirical is also reported
    as a single-number summary of how badly the Markov assumption fails.
"""

import argparse
import numpy as np
import pandas as pd
import os

# -------------------------------------------------------------------------
# Load both files.
# -------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Markov anchoring analysis.")
ap.add_argument("results_dir", help="Path to a per-model results folder (contains turn_by_turn.csv and first_vs_last.csv)")
args = ap.parse_args()

TBT = pd.read_csv(os.path.join(args.results_dir, "turn_by_turn.csv")).set_index("static_type")
FVL = pd.read_csv(os.path.join(args.results_dir, "first_vs_last.csv")).set_index("static_type")

JOINT_COLS = ["Stayed_Right", "Right_to_Wrong", "Wrong_to_Right", "Stayed_Wrong"]
joint = TBT[JOINT_COLS] / 100.0

FVL_COLS = ["rateRR", "rateRW", "rateWR", "rateWW"]
fvl = FVL[FVL_COLS] / 100.0

N_STEPS = 6  # 7 turns => 6 transitions


def transition_matrix(row: pd.Series) -> np.ndarray:
    """Conditional 2x2 transition matrix from joint probabilities.

    State 0 = Right, 1 = Wrong. P[i, j] = Pr(next = j | prev = i).
    """
    rR = row["Stayed_Right"]   + row["Right_to_Wrong"]
    rW = row["Wrong_to_Right"] + row["Stayed_Wrong"]
    return np.array([
        [row["Stayed_Right"]   / rR, row["Right_to_Wrong"] / rR],
        [row["Wrong_to_Right"] / rW, row["Stayed_Wrong"]   / rW],
    ])


def stationary(P: np.ndarray) -> np.ndarray:
    """Left-eigenvector with eigenvalue 1, normalized to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    return pi / pi.sum()


def marginals_from_fvl(row: pd.Series):
    """Returns (pi_0, pi_T_emp) over (R, W)."""
    pi0   = np.array([row["rateRR"] + row["rateRW"],
                      row["rateWR"] + row["rateWW"]])
    piTe  = np.array([row["rateRR"] + row["rateWR"],
                      row["rateRW"] + row["rateWW"]])
    return pi0, piTe


# -------------------------------------------------------------------------
# Run the comparison per static_type.
# -------------------------------------------------------------------------
rows = []
for static_type in fvl.index:
    if static_type not in joint.index:
        continue
    P  = transition_matrix(joint.loc[static_type])
    Pn = np.linalg.matrix_power(P, N_STEPS)
    pi_inf = stationary(P)

    pi0, piT_emp = marginals_from_fvl(fvl.loc[static_type])
    piT_pred = pi0 @ Pn

    # Total variation distance between predicted and empirical final marginals.
    tv = 0.5 * np.abs(piT_pred - piT_emp).sum()
    # Per-turn mixing rate (|lambda_2|): how fast P forgets pi_0.
    eigvals = np.linalg.eigvals(P)
    lam2 = sorted(np.abs(eigvals))[-2]   # second-largest in modulus

    rows.append({
        "static_type":   static_type,
        "P_RR":          P[0, 0], "P_RW": P[0, 1],
        "P_WR":          P[1, 0], "P_WW": P[1, 1],
        "|lambda_2|":    lam2,
        "|lambda_2|^6":  lam2 ** N_STEPS,
        "pi0_R":         pi0[0],
        "pi0_W":         pi0[1],
        "stationary_R":  pi_inf[0],
        "pred_piT_R":    piT_pred[0],
        "emp_piT_R":     piT_emp[0],
        "anchor_gap_R":  piT_emp[0] - piT_pred[0],
        "TV_dist":       tv,
    })

result = pd.DataFrame(rows).set_index("static_type")

# -------------------------------------------------------------------------
# Display.
# -------------------------------------------------------------------------
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 92)
print("Per-turn transition matrix P  (joint probabilities -> conditional)")
print("=" * 92)
print(result[["P_RR", "P_RW", "P_WR", "P_WW"]].round(4).to_string())

print()
print("=" * 92)
print("Mixing diagnostics: per-turn |lambda_2|, and its 6th power")
print("=" * 92)
print("|lambda_2|^6 ~ 0  =>  P^6 has essentially mixed to stationary")
print("|lambda_2|^6 ~ 1  =>  P^6 still close to identity (pi_0 dependence preserved)")
print()
print(result[["|lambda_2|", "|lambda_2|^6", "stationary_R"]].round(4).to_string())

print()
print("=" * 92)
print("Marginal comparison: pi_0 @ P^6  vs  empirical final marginal")
print("=" * 92)
cols = ["pi0_R", "pred_piT_R", "emp_piT_R", "anchor_gap_R", "TV_dist"]
print(result[cols].round(4).to_string())

print()
print("Interpretation:")
print("  anchor_gap_R = emp_piT_R - pred_piT_R")
print("     > 0  =>  model is stickier than memoryless Markov predicts (anchoring)")
print("     < 0  =>  model drifts to wrong more than Markov predicts (compounding)")
print("  TV_dist  =>  total variation distance between predicted and empirical")
print("                final marginals (0 = perfect Markov fit, 1 = totally off).")

# -------------------------------------------------------------------------
# Save.
# -------------------------------------------------------------------------
model_name = os.path.basename(os.path.normpath(args.results_dir))
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.results_dir)))
markov_dir = os.path.join(repo_root, "markov_results")
os.makedirs(markov_dir, exist_ok=True)
out = os.path.join(markov_dir, f"{model_name}.csv")
result.round(4).to_csv(out)
print(f"\nWrote {out}")