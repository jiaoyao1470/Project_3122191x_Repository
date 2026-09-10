import os
import json
import shutil
import tempfile

tmp_dir = tempfile.mkdtemp(prefix="checkpoint_guard_test_")


def guard_logic(checkpoint_path, provenance_path, lambda_G_by_domain, resume):
    if os.path.exists(checkpoint_path):
        if not resume:
            raise RuntimeError("checkpoint exists, RESUME=False")
        if not os.path.exists(provenance_path):
            raise RuntimeError("checkpoint exists but provenance missing")
        with open(provenance_path) as f:
            recorded_lambda_G = json.load(f)
        mismatch = {
            d: (recorded_lambda_G.get(d), lambda_G_by_domain[d])
            for d in lambda_G_by_domain
            if d not in recorded_lambda_G or abs(recorded_lambda_G[d] - lambda_G_by_domain[d]) > 1e-9
        }
        if mismatch:
            raise RuntimeError(f"lambda_G mismatch: {mismatch}")
        return "resumed_ok"
    else:
        os.makedirs(os.path.dirname(provenance_path), exist_ok=True)
        with open(provenance_path, "w") as f:
            json.dump(lambda_G_by_domain, f)
        return "fresh_start"


try:
    ckpt_path = os.path.join(tmp_dir, "seed0_checkpoint.pt")
    prov_path = os.path.join(tmp_dir, "seed0_lambda_G_provenance.json")
    lambda_G_v1 = {"amazon": 0.05, "caltech": 0.03908, "webcam": 0.03094, "dslr": 0.00411}
    lambda_G_v2_changed = {"amazon": 0.05, "caltech": 0.03908, "webcam": 0.03094, "dslr": 0.02}

    result = guard_logic(ckpt_path, prov_path, lambda_G_v1, resume=False)
    assert result == "fresh_start", f"TEST 1 FAIL: expected fresh_start, got {result}"
    assert os.path.exists(prov_path), "TEST 1 FAIL: provenance file was not written"
    with open(prov_path) as f:
        written = json.load(f)
    assert written == lambda_G_v1, f"TEST 1 FAIL: written provenance {written} != {lambda_G_v1}"
    print("TEST 1 PASS: fresh run (no checkpoint) writes provenance correctly, no error raised")

    with open(ckpt_path, "w") as f:
        f.write("fake checkpoint bytes")

    raised = False
    try:
        guard_logic(ckpt_path, prov_path, lambda_G_v1, resume=False)
    except RuntimeError as e:
        raised = True
        assert "RESUME=False" in str(e)
    assert raised, "TEST 2 FAIL: expected RuntimeError when checkpoint exists and RESUME=False"
    print("TEST 2 PASS: checkpoint exists + RESUME=False correctly raises, refusing a silent resume")

    result = guard_logic(ckpt_path, prov_path, lambda_G_v1, resume=True)
    assert result == "resumed_ok", f"TEST 3 FAIL: expected resumed_ok, got {result}"
    print("TEST 3 PASS: checkpoint exists + RESUME=True + matching provenance proceeds without error")

    raised = False
    try:
        guard_logic(ckpt_path, prov_path, lambda_G_v2_changed, resume=True)
    except RuntimeError as e:
        raised = True
        assert "dslr" in str(e), f"TEST 4 FAIL: error message should name the mismatched domain, got: {e}"
    assert raised, "TEST 4 FAIL: expected RuntimeError when lambda_G config changed since checkpoint was written"
    print("TEST 4 PASS: checkpoint exists + RESUME=True + CHANGED lambda_G config correctly raises "
          "(this is the exact silent-config-drift bug the review caught)")

    os.remove(prov_path)
    raised = False
    try:
        guard_logic(ckpt_path, prov_path, lambda_G_v1, resume=True)
    except RuntimeError as e:
        raised = True
        assert "provenance missing" in str(e)
    assert raised, "TEST 5 FAIL: expected RuntimeError when provenance file is missing"
    print("TEST 5 PASS: checkpoint exists + RESUME=True + missing provenance correctly raises "
          "(don't resume blindly from an unverifiable checkpoint)")

    print("\nALL CHECKPOINT-GUARD LOGIC TESTS PASSED (5/5)")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
