import csv
import sys


X_D = 0.3387
LATE_D = 0.2645
X_R = 0.7823
LATE_R = 0.7290


def load_acc(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[(int(r["round"]), r["domain"])] = float(r["acc"])
    domains = sorted(set(d for (_, d) in rows.keys()))
    rounds = sorted(set(rd for (rd, _) in rows.keys()))
    raw_count = sum(1 for _ in open(path)) - 1
    dup_count = raw_count - len(rows)
    if dup_count != 0:
        print(f"  WARNING: {path} has {dup_count} duplicate (round,domain) rows -- resume dedup issue?")
    acc = {d: [rows[(rd, d)] for rd in rounds] for d in domains}
    return acc, rounds, domains


def x_dom_best(acc, rounds, domains):
    avg_by_round = [sum(acc[d][i] for d in domains) / len(domains) for i in range(len(rounds))]
    best_i = max(range(len(rounds)), key=lambda i: avg_by_round[i])
    return rounds[best_i], avg_by_round[best_i], {d: acc[d][best_i] for d in domains}


def late_window(acc, rounds, domains, lo=90, hi=99):
    idx = [i for i, rd in enumerate(rounds) if lo <= rd <= hi]
    if not idx:
        return None
    return {d: sum(acc[d][i] for i in idx) / len(idx) for d in domains}


def report(tag, path):
    acc, rounds, domains = load_acc(path)
    print(f"\n{'=' * 78}\n{tag}  ({path})\n{'=' * 78}")
    print(f"  n_rounds={len(rounds)}  domains={domains}")
    r_best, avg_best, at_best = x_dom_best(acc, rounds, domains)
    print(f"  X-dom-best: round={r_best}  four-domain avg={avg_best:.4f}")
    for d in domains:
        print(f"    {d:<8} @that round = {at_best[d]:.4f}")
    x_dslr = at_best["dslr"]
    print(f"  -> dslr X-dom-best = {x_dslr:.4f}   own-peak = {max(acc['dslr']):.4f} @round {rounds[acc['dslr'].index(max(acc['dslr']))]}")
    lw = late_window(acc, rounds, domains)
    l_dslr = None
    if lw is not None:
        for d in domains:
            print(f"  late-window {d:<8} = {lw[d]:.4f}")
        l_dslr = lw["dslr"]
    else:
        print("  late-window: not enough rounds yet (need round 90-99)")
    return x_dslr, l_dslr


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_ab_pathdecomp_results.py <a_acc.csv> <b_acc.csv>")
        sys.exit(1)

    x_a, late_a = report("ARM A (expanded local SGD, fixed feature/upload)", sys.argv[1])
    x_b, late_b = report("ARM B (fixed local SGD, expanded feature/upload)", sys.argv[2])

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"  D           X-dom-best={X_D:.4f}  late-window={LATE_D:.4f}")
    print(f"  A           X-dom-best={x_a:.4f}  late-window={late_a if late_a is None else f'{late_a:.4f}'}")
    print(f"  B           X-dom-best={x_b:.4f}  late-window={late_b if late_b is None else f'{late_b:.4f}'}")
    print(f"  Resampling  X-dom-best={X_R:.4f}  late-window={LATE_R:.4f}  (matched ceiling)")

    span = X_R - X_D
    pos_a = (x_a - X_D) / span
    pos_b = (x_b - X_D) / span
    print(f"\n  position in [D, Resampling] span (X-dom-best):  A={pos_a:.4f}   B={pos_b:.4f}")
    print("    (0 = no recovery, 1 = matches full Resampling)")

    if late_a is not None and late_b is not None:
        span_late = LATE_R - LATE_D
        pos_a_late = (late_a - LATE_D) / span_late
        pos_b_late = (late_b - LATE_D) / span_late
        print(f"  position in [D, Resampling] span (late-window): A={pos_a_late:.4f}   B={pos_b_late:.4f}")

    I_X = X_R - x_a - x_b + X_D
    print(f"\n  Interaction I_X = X_R - X_A - X_B + X_D = {I_X:.4f}")
    print("    I_X > 0 and large: local x server support have a synergistic (super-additive) relationship")
    print("    I_X ~ 0: A and B's individual recoveries roughly add up to Resampling's total")
    print("    I_X < 0: A and/or B alone already overshoot the additive expectation (unlikely here)")

    print(f"\n  Reading rule:")
    if pos_a > pos_b + 0.15:
        print("    A >> B -> local SGD support is the dominant path.")
        print("    -> repair candidates: multi-prototype / mean+dispersion / feature synthesis feeding the ENCODER via local training.")
    elif pos_b > pos_a + 0.15:
        print("    B >> A -> post-local/server-side (prototype/GC/discriminator) coverage is the dominant path.")
        print("    -> repair candidates: same family, but consumed via Layer1/Layer2 aggregation, not local SGD's own loss.")
        print("    -> CAVEAT: B bundles server coverage with the BN-buffer side effect (see run_b_server_expand_office_dslr.py header) -- a strong B needs the BN-buffer-preserved follow-up before this reading is trusted.")
    else:
        print("    A and B are comparable -> check I_X above for synergy; if I_X is large positive, both paths likely need repair.")
