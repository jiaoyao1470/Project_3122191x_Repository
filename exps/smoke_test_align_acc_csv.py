import os
import csv
import shutil
import torch


def align_acc_csv_to_checkpoint(checkpoint_path, acc_log_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_round = int(ckpt["round"])

    if not os.path.exists(acc_log_path):
        raise RuntimeError(
            f"Resume checkpoint is at round {ckpt_round}, but acc.csv is missing: {acc_log_path}"
        )

    with open(acc_log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rows = [r for r in rows if int(r["round"]) <= ckpt_round]
    dedup = {}
    for r in rows:
        dedup[(int(r["round"]), r["domain"])] = r
    rows = sorted(dedup.values(), key=lambda r: (int(r["round"]), r["domain"]))

    with open(acc_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aligned acc.csv to checkpoint round {ckpt_round}; training resumes at round {ckpt_round + 1}.")


scratch = "smoke_align_acc_csv_TEST"
if os.path.exists(scratch):
    shutil.rmtree(scratch)
os.makedirs(scratch)

DOMAINS = ["caltech", "amazon", "webcam", "dslr"]
FIELDS = ["round", "domain", "train_loss", "kl_loss", "acc"]


def write_synthetic_acc_csv(path, round_range, extra_dup_row=None):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in round_range:
            for d in DOMAINS:
                writer.writerow({"round": r, "domain": d, "train_loss": 1.0, "kl_loss": 0.1,
                                  "acc": 0.1 * r})
        if extra_dup_row is not None:
            writer.writerow(extra_dup_row)


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


print("=" * 20 + " Test 1: basic truncation (exact bug scenario) " + "=" * 20)
acc_path = f"{scratch}/t1_acc.csv"
ckpt_path = f"{scratch}/t1_checkpoint.pt"
write_synthetic_acc_csv(acc_path, range(0, 8))
torch.save({"round": 4}, ckpt_path)
align_acc_csv_to_checkpoint(ckpt_path, acc_path)
rows = read_rows(acc_path)
found_rounds = sorted(set(int(r["round"]) for r in rows))
print(f"rows after align: {len(rows)}  rounds present: {found_rounds}")
assert len(rows) == 5 * len(DOMAINS), f"FAIL: expected {5*len(DOMAINS)} rows (rounds 0-4 x 4 domains), got {len(rows)}"
assert found_rounds == [0, 1, 2, 3, 4], f"FAIL: expected rounds 0-4, got {found_rounds}"
print("PASS: rounds 5-7 correctly truncated, rounds 0-4 fully preserved.")

print("\n" + "=" * 20 + " Test 2: dedup keeps the LAST occurrence " + "=" * 20)
acc_path2 = f"{scratch}/t2_acc.csv"
ckpt_path2 = f"{scratch}/t2_checkpoint.pt"
write_synthetic_acc_csv(acc_path2, range(0, 3))
with open(acc_path2, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writerow({"round": 1, "domain": "dslr", "train_loss": 9.9, "kl_loss": 9.9, "acc": 0.9999})
torch.save({"round": 2}, ckpt_path2)
align_acc_csv_to_checkpoint(ckpt_path2, acc_path2)
rows2 = read_rows(acc_path2)
dslr_r1_rows = [r for r in rows2 if int(r["round"]) == 1 and r["domain"] == "dslr"]
print(f"total rows: {len(rows2)}  (round=1,domain=dslr) rows: {len(dslr_r1_rows)}  value: {dslr_r1_rows[0]['acc'] if dslr_r1_rows else None}")
assert len(rows2) == 3 * len(DOMAINS), f"FAIL: expected {3*len(DOMAINS)} rows after dedup, got {len(rows2)}"
assert len(dslr_r1_rows) == 1, f"FAIL: expected exactly 1 (round=1,domain=dslr) row after dedup, got {len(dslr_r1_rows)}"
assert dslr_r1_rows[0]["acc"] == "0.9999", (
    f"FAIL: dedup should keep the LAST occurrence (the injected duplicate), got acc={dslr_r1_rows[0]['acc']}"
)
print("PASS: duplicate (round,domain) collapsed to 1 row, keeping the last occurrence as intended.")

print("\n" + "=" * 20 + " Test 3: idempotency (already-aligned file is untouched by a second call) " + "=" * 20)
acc_path3 = f"{scratch}/t3_acc.csv"
ckpt_path3 = f"{scratch}/t3_checkpoint.pt"
write_synthetic_acc_csv(acc_path3, range(0, 5))
torch.save({"round": 4}, ckpt_path3)
align_acc_csv_to_checkpoint(ckpt_path3, acc_path3)
with open(acc_path3, "rb") as f:
    content_after_first = f.read()
align_acc_csv_to_checkpoint(ckpt_path3, acc_path3)
with open(acc_path3, "rb") as f:
    content_after_second = f.read()
assert content_after_first == content_after_second, (
    "FAIL: a second call on an already-aligned file changed its content -- not idempotent."
)
print("PASS: calling align twice on already-aligned data is a true no-op (matches the earlier "
      "checkpoint_every=1 smoke test's normal-resume case, which relies on this being a no-op).")

print("\n" + "=" * 20 + " Test 4: missing acc.csv raises RuntimeError " + "=" * 20)
ckpt_path4 = f"{scratch}/t4_checkpoint.pt"
torch.save({"round": 4}, ckpt_path4)
try:
    align_acc_csv_to_checkpoint(ckpt_path4, f"{scratch}/does_not_exist.csv")
    print("FAIL: expected RuntimeError, but no exception was raised")
except RuntimeError as e:
    print(f"PASS: RuntimeError raised as expected: {e}")

shutil.rmtree(scratch)
print("\n=== ALL CHECKS COMPLETE ===")
