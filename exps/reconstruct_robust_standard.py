import sys
import csv
from collections import defaultdict


WINDOW = 5
path = sys.argv[1]
if len(sys.argv) > 2:
    WINDOW = int(sys.argv[2])

rows_by_domain = defaultdict(dict)
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows_by_domain[row["domain"]][int(row["round"])] = float(row["acc"])

print(f"{'domain':10s}{'robust_round':14s}{'robust_acc':12s}{'rolling_avg_at_peak':22s}{'final_round_acc':16s}{'first_round_acc':16s}")
robust_accs = {}
for domain in sorted(rows_by_domain.keys()):
    per_round = rows_by_domain[domain]
    rounds_sorted = sorted(per_round.keys())
    best_round, best_ma = None, float("-inf")
    for i, r in enumerate(rounds_sorted):
        if i + 1 < WINDOW:
            continue
        window_rounds = rounds_sorted[i - WINDOW + 1: i + 1]
        ma = sum(per_round[wr] for wr in window_rounds) / WINDOW
        if ma > best_ma:
            best_ma = ma
            best_round = r
    if best_round is None:
        best_round = max(rounds_sorted, key=lambda r: per_round[r])
        best_ma = per_round[best_round]
    robust_acc = per_round[best_round]
    robust_accs[domain] = robust_acc
    final_round = rounds_sorted[-1]
    first_round = rounds_sorted[0]
    print(f"{domain:10s}{best_round:<14d}{robust_acc*100:<12.2f}{best_ma*100:<22.2f}{per_round[final_round]*100:<16.2f}{per_round[first_round]*100:<16.2f}")

print(f"\nrobust mean across {len(robust_accs)} domains: {sum(robust_accs.values())/len(robust_accs)*100:.2f}%")

LATE_WINDOW_ROUNDS = 50
print(f"\nlate-window (last {LATE_WINDOW_ROUNDS} rounds by round-count, not robust-selected) mean/median per domain:")
print(f"{'domain':10s}{'late_mean':12s}{'late_median':14s}{'n_rounds':10s}")
import statistics
for domain in sorted(rows_by_domain.keys()):
    per_round = rows_by_domain[domain]
    rounds_sorted = sorted(per_round.keys())
    cutoff = rounds_sorted[-1] - LATE_WINDOW_ROUNDS + 1
    late_vals = [per_round[r] for r in rounds_sorted if r >= cutoff]
    print(f"{domain:10s}{statistics.mean(late_vals)*100:<12.2f}{statistics.median(late_vals)*100:<14.2f}{len(late_vals):<10d}")
