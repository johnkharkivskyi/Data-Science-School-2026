"""
Parametric UMAP — Self-Check
============================

    python selfcheck.py

A feedback loop for your work in `student_tasks.py`. Run it as often as you
like while implementing.

WHAT THIS IS NOT
----------------
This is not the grader, and passing it does not mean full marks. It checks
STRUCTURAL properties -- shapes, ranges, symmetry, that gradients flow, that
the defining equations hold. It deliberately contains no hardcoded expected
values, because the real test suite compares your outputs against fixed
tensors to 4 decimal places and you are meant to work from the specification
in the docstrings rather than tune against a test.

So: a failure here is definitely a bug. A pass here means you have not made
any of the common structural mistakes -- it does not prove your constants,
your convergence schedule, or your fit grid match.

The other signal worth watching is `val_knn_acc` while `boilerplate.py` runs.
Near 0.10 means chance, i.e. something upstream is wrong.
"""

from __future__ import annotations

import math
import traceback

import torch

from student_tasks import (
    compute_asymmetric_p,
    compute_min_pairwise_distance,
    compute_symmetric_p,
    find_a_b,
    find_sigma,
    umap_loss,
)

DTYPE = torch.float64
SEED = 0

# Every fixture below draws from an explicitly seeded `torch.Generator`, so
# these checks give the same answer on every machine and every run. This global
# seed is belt-and-braces for anything that slips through to the ambient RNG.
torch.manual_seed(SEED)


# --------------------------------------------------------------------------
# Tiny check runner
# --------------------------------------------------------------------------

_RESULTS: list[tuple[str, str, str]] = []


def check(name: str):
    """Decorator: run the function, record pass / fail / not-yet-implemented."""

    def decorator(fn):
        try:
            fn()
            _RESULTS.append((name, "PASS", ""))
        except NotImplementedError:
            _RESULTS.append((name, "TODO", "not implemented yet"))
        except AssertionError as exc:
            _RESULTS.append((name, "FAIL", str(exc)))
        except Exception:
            _RESULTS.append((name, "ERROR", traceback.format_exc(limit=2).strip()))
        return fn

    return decorator


def _images(n: int = 6) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.rand(n, 1, 28, 28, generator=g, dtype=DTYPE)


def _distances(n: int = 6) -> torch.Tensor:
    """A valid symmetric distance matrix with a zero diagonal."""
    g = torch.Generator().manual_seed(SEED + 1)
    points = torch.rand(n, 5, generator=g, dtype=DTYPE) * 3.0
    return torch.cdist(points, points)


# --------------------------------------------------------------------------
# compute_min_pairwise_distance
# --------------------------------------------------------------------------


@check("compute_min_pairwise_distance: shape is (B,)")
def _():
    images = _images()
    rho = compute_min_pairwise_distance(images)
    assert rho.shape == (images.shape[0],), (
        f"got {tuple(rho.shape)}, expected {(images.shape[0],)} — you must "
        f"flatten (B, 1, 28, 28) to (B, 784) yourself"
    )


@check("compute_min_pairwise_distance: excludes the point itself")
def _():
    rho = compute_min_pairwise_distance(_images())
    assert torch.isfinite(rho).all(), "rho contains nan/inf — see the masked_fill note"
    assert (rho > 0).all(), (
        "rho has zero entries, so d_ii = 0 is leaking into the minimum. "
        "Mask the diagonal with inf before .min()."
    )


@check("compute_min_pairwise_distance: agrees with a brute-force loop")
def _():
    """Independent cross-check against the slow, obviously-correct version."""
    images = _images()
    flat = images.reshape(images.shape[0], -1)

    expected = []
    for i in range(len(flat)):
        best = math.inf
        for j in range(len(flat)):
            if i != j:
                best = min(best, torch.linalg.norm(flat[i] - flat[j]).item())
        expected.append(best)

    got = compute_min_pairwise_distance(images)
    assert torch.allclose(got, torch.tensor(expected, dtype=DTYPE), atol=1e-8), (
        f"vectorised result disagrees with the brute-force loop\n"
        f"  got:      {got.tolist()}\n  expected: {expected}"
    )


# --------------------------------------------------------------------------
# find_a_b
# --------------------------------------------------------------------------


@check("find_a_b: returns the documented (a, b) for the defaults")
def _():
    a, b = find_a_b(min_dist=0.1, spread=1.0)
    assert abs(a - 1.5769) < 1e-3, f"a = {a}, expected ~1.5769 (see the docstring)"
    assert abs(b - 0.8951) < 1e-3, f"b = {b}, expected ~0.8951 (see the docstring)"


@check("find_a_b: the fitted curve actually tracks the target")
def _():
    min_dist, spread = 0.1, 1.0
    a, b = find_a_b(min_dist=min_dist, spread=spread)

    for d in (0.05, 0.5, 1.0, 2.0):
        phi = 1.0 / (1.0 + a * d ** (2 * b))
        psi = 1.0 if d <= min_dist else math.exp(-(d - min_dist) / spread)
        assert abs(phi - psi) < 0.1, (
            f"at d={d} the fitted curve gives {phi:.4f} but the target is "
            f"{psi:.4f} — check your grid and the form of Psi(d)"
        )


# --------------------------------------------------------------------------
# find_sigma
# --------------------------------------------------------------------------


@check("find_sigma: shape and positivity")
def _():
    d = _distances()
    sigma = find_sigma(d, k=4)
    assert sigma.shape == (d.shape[0],), f"got {tuple(sigma.shape)}, expected {(d.shape[0],)}"
    assert torch.isfinite(sigma).all(), "sigma contains nan/inf"
    assert (sigma > 0).all(), "sigma must be strictly positive"


@check("find_sigma: the defining constraint sum = log2(k) holds")
def _():
    """The whole point of the binary search. If this fails, nothing else matters."""
    d = _distances()
    for k in (2, 4, 5):
        sigma = find_sigma(d, k=k)

        rho = d.masked_fill(torch.eye(d.shape[0], dtype=torch.bool), torch.inf).min(dim=1).values
        weights = torch.exp(-torch.clamp(d - rho.unsqueeze(1), min=0.0) / sigma.unsqueeze(1))
        weights = weights.masked_fill(torch.eye(d.shape[0], dtype=torch.bool), 0.0)

        got = weights.sum(dim=1)
        target = math.log2(k)
        assert torch.allclose(got, torch.full_like(got, target), atol=1e-3), (
            f"k={k}: local weights sum to {got.tolist()}, expected {target:.4f} "
            f"per row. Check that you exclude the diagonal and subtract rho."
        )


@check("find_sigma: does not create tensors on a hardcoded device/dtype")
def _():
    """float32 in must not silently become float64 out (and vice versa)."""
    d = _distances().to(torch.float32)
    sigma = find_sigma(d, k=4)
    assert sigma.dtype == torch.float32, (
        f"passed float32, got {sigma.dtype} back — derive dtype from the input "
        f"instead of hardcoding it, or your GPU training run will break"
    )


# --------------------------------------------------------------------------
# compute_asymmetric_p
# --------------------------------------------------------------------------


def _rho_sigma(d: torch.Tensor, k: int = 4):
    rho = d.masked_fill(torch.eye(d.shape[0], dtype=torch.bool), torch.inf).min(dim=1).values
    return rho, find_sigma(d, k=k)


@check("compute_asymmetric_p: range [0, 1] and zero diagonal")
def _():
    d = _distances()
    rho, sigma = _rho_sigma(d)
    p = compute_asymmetric_p(d, rho, sigma)

    assert p.shape == d.shape, f"got {tuple(p.shape)}, expected {tuple(d.shape)}"
    assert torch.isfinite(p).all(), "p_asym contains nan/inf"
    assert (p >= 0).all() and (p <= 1).all(), "p_asym must lie in [0, 1]"
    assert torch.allclose(torch.diagonal(p), torch.zeros(d.shape[0], dtype=DTYPE), atol=1e-12), (
        "p_asym must have a zero diagonal — a point is not its own neighbour"
    )


@check("compute_asymmetric_p: nearest neighbour gets weight exactly 1")
def _():
    """d_ij = rho_i gives exp(0) = 1. This is what the max(0, .) clamp is for."""
    d = _distances()
    rho, sigma = _rho_sigma(d)
    p = compute_asymmetric_p(d, rho, sigma)

    row_max = p.max(dim=1).values
    assert torch.allclose(row_max, torch.ones_like(row_max), atol=1e-8), (
        f"every row should peak at exactly 1.0 (its nearest neighbour), got "
        f"{row_max.tolist()} — check the max(0, d - rho) clamp"
    )


@check("compute_asymmetric_p: rho/sigma broadcast along ROWS, not columns")
def _():
    """Detects the .unsqueeze(0) / .unsqueeze(1) mix-up.

    Note that mix-up does NOT produce a symmetric matrix -- since `distances`
    is symmetric, it produces the exact TRANSPOSE, which is asymmetric too. So
    checking "is it asymmetric" proves nothing. Instead we give point 0 a much
    smaller sigma than everyone else: its weights must decay sharply along
    ROW 0. If they decay down COLUMN 0 instead, the broadcast axis is wrong.
    """
    d = _distances()
    rho, _ = _rho_sigma(d)

    sigma = torch.full((d.shape[0],), 5.0, dtype=DTYPE)
    sigma[0] = 0.05  # point 0 has a very tight neighbourhood

    p = compute_asymmetric_p(d, rho, sigma)
    row_0, col_0 = p[0].sum(), p[:, 0].sum()

    assert row_0 < col_0, (
        f"row 0 sums to {row_0:.4f} but column 0 sums to {col_0:.4f}. "
        f"sigma[0] is tiny, so ROW 0 should be the one that decays. "
        f"rho and sigma are indexed by the row i — broadcast them with "
        f".unsqueeze(1), not .unsqueeze(0). This is the most common bug in "
        f"the assignment and it silently transposes your whole graph."
    )


# --------------------------------------------------------------------------
# compute_symmetric_p
# --------------------------------------------------------------------------


@check("compute_symmetric_p: symmetric, in [0, 1]")
def _():
    d = _distances()
    rho, sigma = _rho_sigma(d)
    p = compute_symmetric_p(compute_asymmetric_p(d, rho, sigma))

    assert torch.isfinite(p).all(), "p_sym contains nan/inf"
    assert torch.allclose(p, p.t(), atol=1e-10), "p_ij must be symmetric"
    assert (p >= 0).all() and (p <= 1).all(), "p_ij must lie in [0, 1]"


@check("compute_symmetric_p: is a fuzzy union, not an average")
def _():
    d = _distances()
    rho, sigma = _rho_sigma(d)
    p_asym = compute_asymmetric_p(d, rho, sigma)
    p = compute_symmetric_p(p_asym)

    assert (p >= torch.maximum(p_asym, p_asym.t()) - 1e-9).all(), (
        "p_ij dropped below max(p_j|i, p_i|j) — you averaged the two "
        "directions instead of using the t-conorm a + b - a*b"
    )


# --------------------------------------------------------------------------
# umap_loss
# --------------------------------------------------------------------------


def _pq():
    g = torch.Generator().manual_seed(SEED + 2)
    p = torch.rand(5, 5, generator=g, dtype=DTYPE) * 0.8 + 0.1
    q = torch.rand(5, 5, generator=g, dtype=DTYPE) * 0.8 + 0.1
    return p, q


@check("umap_loss: returns a scalar")
def _():
    p, q = _pq()
    loss = umap_loss(p, q)
    assert loss.dim() == 0, f"expected a 0-dim scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), "loss is nan/inf"


@check("umap_loss: vanishes when q == p")
def _():
    p, _ = _pq()
    loss = umap_loss(p, p.clone())
    assert abs(loss.item()) < 1e-6, (
        f"a perfect embedding should give ~0 loss, got {loss.item()} — check "
        f"the sign and the (1-p) repulsive term"
    )


@check("umap_loss: is non-negative")
def _():
    p, q = _pq()
    assert umap_loss(p, q).item() >= -1e-9, "cross-entropy cannot be negative"


@check("umap_loss: survives q at exactly 0 and 1 (clamping)")
def _():
    q = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
    p = torch.tensor([[0.3, 0.7], [0.7, 0.3]], dtype=DTYPE)
    loss = umap_loss(p, q)
    assert torch.isfinite(loss), (
        "loss blew up to nan/inf for q in {0, 1} — clamp q to "
        "[1e-4, 1 - 1e-4] BEFORE any logarithm"
    )


@check("umap_loss: gradients flow to q")
def _():
    p, q = _pq()
    q = q.requires_grad_(True)
    loss = umap_loss(p, q)

    assert loss.requires_grad and loss.grad_fn is not None, (
        "loss has no grad_fn — the graph is broken. Look for .detach(), "
        ".numpy(), .item(), float(), math.log, or torch.no_grad()."
    )
    loss.backward()
    assert q.grad is not None, "no gradient reached q"
    assert torch.isfinite(q.grad).all(), "gradient contains nan/inf"
    assert not torch.allclose(q.grad, torch.zeros_like(q.grad)), (
        "gradient is all zeros — q is not participating in the loss"
    )


@check("umap_loss: works on a flat (E,) edge list, not just (B, B)")
def _():
    """Training feeds sampled edges as 1-D tensors. Same values, same answer."""
    p, q = _pq()
    matrix = umap_loss(p, q)
    flat = umap_loss(p.reshape(-1), q.reshape(-1))
    assert flat.dim() == 0, "must return a scalar for 1-D input too"
    assert abs(flat.item() - matrix.item()) < 1e-9, (
        f"(B, B) gave {matrix.item():.6f} but the same values as a flat (E,) "
        f"list gave {flat.item():.6f} — umap_loss must be purely elementwise. "
        f"No torch.eye, masked_fill, .t(), diagonal indexing or p.shape[0]."
    )


@check("umap_loss: passes torch.autograd.gradcheck")
def _():
    """The strict differentiability requirement. Same check used at grading."""
    p, q = _pq()
    q = q.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda q_: umap_loss(p, q_), (q,), eps=1e-6, atol=1e-4)


# --------------------------------------------------------------------------


def main() -> int:
    width = 62
    print("=" * width)
    print("  SELF-CHECK — structural checks on student_tasks.py")
    print("=" * width)

    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "TODO": 0}
    for name, status, detail in _RESULTS:
        counts[status] += 1
        marker = {"PASS": "ok  ", "FAIL": "FAIL", "ERROR": "ERR ", "TODO": "todo"}[status]
        print(f"  [{marker}] {name}")
        if detail and status != "TODO":
            for line in detail.splitlines():
                print(f"         {line}")

    print("=" * width)
    print(
        f"  {counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['ERROR']} errored, {counts['TODO']} not yet implemented"
    )

    if counts["TODO"] == len(_RESULTS):
        print("\n  Nothing implemented yet — start with compute_min_pairwise_distance.")
    elif counts["FAIL"] or counts["ERROR"]:
        print("\n  Fix the failures above, then re-run.")
    elif counts["TODO"]:
        print("\n  Everything implemented so far looks structurally sound. Keep going.")
    else:
        print(
            "\n  All structural checks pass. Remember these do not verify your\n"
            "  exact numeric constants — re-read the docstrings for the parts\n"
            "  that pin down an exact procedure, then train and watch\n"
            "  val_knn_acc."
        )
    print("=" * width)

    return 1 if (counts["FAIL"] or counts["ERROR"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
