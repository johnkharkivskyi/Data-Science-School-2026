"""
Parametric UMAP — Global Probability Pre-computation
====================================================

    python generate_global_probabilities.py

RUN THIS AFTER you have implemented `student_tasks.py`, and BEFORE
`boilerplate.py`. It uses your functions to build the fuzzy neighbourhood
graph `p_ij` once, over the WHOLE 54,000-image training subset, and saves it
to disk. Training then looks up the relevant sub-block per batch instead of
rebuilding a graph from whatever images happened to land in it.

Why this is the real algorithm
------------------------------
Building `p` inside `training_step` (the earlier approach) computes each
point's neighbours *within its minibatch*. With 256 random images out of
54,000, a point's "nearest neighbour" is usually not a neighbour at all --
it is simply the closest of 255 strangers. The manifold you are trying to
preserve is a property of the full dataset, so the graph has to be built on
the full dataset. That is what UMAP actually does, and what this script does.

Why the result is SPARSE
------------------------
A dense (54000, 54000) float32 matrix is 11.7 GB, which will not fit in
memory. It also does not need to: UMAP only keeps each point's `k` nearest
neighbours (k=15 here), so the graph has ~810k edges out of 2.9 billion
possible pairs -- a density of about 0.05%. We store it as a SciPy sparse
matrix, roughly 13 MB on disk.

Everything below is fully implemented. You do not need to edit this file; you
just need `student_tasks.py` to be correct, because this script calls straight
into it:

    find_sigma            -> the per-point bandwidth sigma_i
    compute_asymmetric_p  -> the directed weights p_{j|i}
    compute_symmetric_p   -> validates the sparse symmetrisation

Usage
-----
    python generate_global_probabilities.py
    python generate_global_probabilities.py --limit 5000   # fast smoke test
    python generate_global_probabilities.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch

from boilerplate import (
    DATA_DIR,
    GLOBAL_P_META,
    GLOBAL_P_PATH,
    N_NEIGHBORS,
    SEED,
    VAL_FRACTION,
    get_datasets,
)

# Rows processed at a time when computing sigma / p. Must exceed N_NEIGHBORS
# (see `_padded_distance_block` for why).
BLOCK_SIZE = 1024

# Rows per chunk of the k-NN scan. Each chunk materialises a
# (CHUNK_SIZE, N) distance matrix, so this is the main memory knob:
# 512 x 54000 x 4 bytes ~= 110 MB.
CHUNK_SIZE = 512


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verify_student_code() -> None:
    """Fail fast, with a readable message, if `student_tasks.py` is incomplete."""
    from student_tasks import (
        compute_asymmetric_p,
        compute_symmetric_p,
        find_sigma,
    )

    d = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]], dtype=torch.float32
    )
    try:
        sigma = find_sigma(d, k=2)
        rho = d.masked_fill(torch.eye(3, dtype=torch.bool), torch.inf).min(dim=1).values
        compute_symmetric_p(compute_asymmetric_p(d, rho, sigma))
    except NotImplementedError as exc:
        raise SystemExit(
            "\n  student_tasks.py is not finished yet.\n\n"
            "  This script needs find_sigma, compute_asymmetric_p and\n"
            "  compute_symmetric_p. Implement them, check with\n"
            "  `python selfcheck.py`, then run this again.\n"
        ) from exc


# --------------------------------------------------------------------------
# Step 1 — load the training subset into memory
# --------------------------------------------------------------------------


def load_train_images(data_dir: str, limit: int | None = None) -> torch.Tensor:
    """Return the training subset flattened to `(N, 784)` float32.

    54,000 x 784 x 4 bytes is about 169 MB, so the whole subset fits in RAM
    comfortably. Order matters: row i here must be sample i of the training
    Subset, because that is the index the DataLoader will hand back during
    training and use to look up this matrix.
    """
    train_ds, _val_ds = get_datasets(data_dir=data_dir)

    n = len(train_ds) if limit is None else min(limit, len(train_ds))
    images = torch.empty(n, 784, dtype=torch.float32)
    for i in range(n):
        image, _label = train_ds[i]
        images[i] = image.reshape(-1)

    return images


# --------------------------------------------------------------------------
# Step 2 — the k-nearest-neighbour graph
# --------------------------------------------------------------------------


def compute_knn(
    images: torch.Tensor,
    k: int,
    device: torch.device,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k-NN over the whole subset, computed in row chunks.

    Brute force, not an approximate index: at 54k x 784 this takes a couple of
    minutes and keeps the graph exactly reproducible.

    Returns
    -------
    (distances, indices) : each `(N, k)`, distances ascending, self excluded.
    """
    n = images.shape[0]
    images_dev = images.to(device)

    knn_distances = torch.empty(n, k, dtype=torch.float32)
    knn_indices = torch.empty(n, k, dtype=torch.long)

    start_time = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # (chunk, N) distances from these rows to every image.
        chunk = torch.cdist(images_dev[start:end], images_dev)

        # Exclude self: point i must not be its own nearest neighbour.
        rows = torch.arange(end - start, device=device)
        chunk[rows, torch.arange(start, end, device=device)] = torch.inf

        values, indices = torch.topk(chunk, k, dim=1, largest=False)
        knn_distances[start:end] = values.cpu()
        knn_indices[start:end] = indices.cpu()

        done = end / n
        elapsed = time.time() - start_time
        print(
            f"\r  k-NN {end:>6}/{n}  ({done:5.1%})  "
            f"elapsed {elapsed:5.1f}s  eta {elapsed / done - elapsed:5.1f}s",
            end="",
            flush=True,
        )

    print()
    return knn_distances, knn_indices


# --------------------------------------------------------------------------
# Step 3 — sigma and the directed weights, via the student's functions
# --------------------------------------------------------------------------


def _padded_distance_block(
    knn_distances: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack `(M, k)` neighbour distances into an `(M, M)` matrix.

    `find_sigma` and `compute_asymmetric_p` are specified against a square
    (B, B) distance matrix with a zero diagonal -- that is what they were
    graded on, and this script must not require you to have written anything
    different. But globally we only have each point's k nearest neighbours,
    not a full row of 54,000 distances.

    So we fabricate a square block that is equivalent for their purposes.
    For local row i:

        column i          -> 0.0    (the diagonal, i.e. the point itself)
        k other columns   -> that point's k neighbour distances
        every remaining   -> +inf   ("infinitely far away")

    Both functions then behave exactly as they should:

        rho_i  = min over j != i        -> the nearest real neighbour  ✓
        exp(-max(0, inf - rho) / sigma) -> exactly 0, contributing nothing  ✓

    +inf, not a large finite number: during the binary search `sigma` can grow
    large, and `exp(-1e6 / big_sigma)` is not necessarily 0, which would
    quietly corrupt the sum. `exp(-inf)` is 0 for every sigma.

    A note on what this changes
    --------------------------
    The budget constraint now reads

        sum over the k NEAREST neighbours = log2(k)

    rather than "sum over all N-1 other points". That is deliberate, and it is
    what the reference UMAP implementation does: only a point's k nearest
    neighbours are considered part of its neighbourhood at all.

    It does mean sigma here differs from what you would get by handing
    `find_sigma` a full dense (N, N) matrix — with 53,999 terms instead of 15,
    even negligible far-away contributions accumulate, so the search settles on
    a smaller sigma to hit the same budget. Your function is unchanged and
    correct either way; it is the neighbourhood definition that differs, and
    the k-NN one is the right one.

    The neighbour columns are placed at `(i + 1 + j) % M`, which is guaranteed
    to dodge the diagonal as long as M > k -- hence the BLOCK_SIZE floor.

    Returns
    -------
    (block, columns) : `(M, M)` padded distances, and the `(M, k)` column
    indices the neighbour values were written to (for reading results back).
    """
    m, k = knn_distances.shape
    assert m > k, f"block of {m} rows must exceed k={k}"

    offsets = torch.arange(1, k + 1).unsqueeze(0)  # (1, k)
    columns = (torch.arange(m).unsqueeze(1) + offsets) % m  # (M, k)

    block = torch.full((m, m), torch.inf, dtype=knn_distances.dtype)
    block.scatter_(1, columns, knn_distances)
    block.fill_diagonal_(0.0)

    return block, columns


def compute_directed_weights(
    knn_distances: torch.Tensor,
    n_neighbors: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the student's `find_sigma` / `compute_asymmetric_p` over all rows.

    Returns
    -------
    (sigma, p_directed) : `(N,)` bandwidths and `(N, k)` weights p_{j|i}
    aligned with `knn_indices`.
    """
    from student_tasks import compute_asymmetric_p, find_sigma

    n, k = knn_distances.shape
    sigma = torch.empty(n, dtype=torch.float32)
    p_directed = torch.empty(n, k, dtype=torch.float32)

    # Never leave a final block with fewer than k+1 rows -- the padding trick
    # needs M > k. Merge a short tail into the previous block instead.
    bounds = list(range(0, n, block_size)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] <= k:
        bounds.pop(-2)

    for start, end in zip(bounds[:-1], bounds[1:]):
        block, columns = _padded_distance_block(knn_distances[start:end], block_size)

        # rho comes from the GLOBAL k-NN, not from within a batch: it is the
        # distance to this point's true nearest neighbour in the full dataset.
        # (`find_sigma` derives the identical value internally from `block`.)
        rho = knn_distances[start:end, 0]

        block_sigma = find_sigma(block, k=n_neighbors)
        block_p = compute_asymmetric_p(block, rho, block_sigma)

        sigma[start:end] = block_sigma
        p_directed[start:end] = block_p.gather(1, columns)

        print(f"\r  weights {end:>6}/{n}  ({end / n:5.1%})", end="", flush=True)

    print()
    return sigma, p_directed


# --------------------------------------------------------------------------
# Step 4 — symmetrise
# --------------------------------------------------------------------------


def symmetrize(p_asymmetric: sp.csr_matrix) -> sp.csr_matrix:
    """Apply the probabilistic t-conorm across the whole sparse graph.

        p_ij = p_{j|i} + p_{i|j} - p_{j|i} * p_{i|j}

    Identical to `compute_symmetric_p`, but done sparsely: the dense version
    would need an 11.7 GB transpose. `validate_symmetrisation` below checks the
    two agree on a real sub-block.
    """
    transposed = p_asymmetric.transpose().tocsr()
    result = p_asymmetric + transposed - p_asymmetric.multiply(transposed)
    result = sp.csr_matrix(result)
    result.setdiag(0.0)  # no self-loops
    result.eliminate_zeros()
    return result


def validate_symmetrisation(
    p_asymmetric: sp.csr_matrix,
    p_symmetric: sp.csr_matrix,
    size: int = 200,
) -> None:
    """Cross-check the sparse t-conorm against the student's dense function.

    A sub-block over the same index set on both axes is closed under
    transposition, so symmetrising the sub-block and taking the sub-block of
    the symmetrised matrix must agree exactly.
    """
    from student_tasks import compute_symmetric_p

    size = min(size, p_asymmetric.shape[0])
    index = np.arange(size)

    dense_asym = torch.from_numpy(p_asymmetric[index][:, index].toarray())
    expected = compute_symmetric_p(dense_asym).numpy()
    actual = p_symmetric[index][:, index].toarray()

    if not np.allclose(expected, actual, atol=1e-6):
        worst = np.abs(expected - actual).max()
        raise SystemExit(
            f"\n  Sparse symmetrisation disagrees with your compute_symmetric_p "
            f"(max difference {worst:.2e}).\n"
            f"  Check that compute_symmetric_p implements the t-conorm\n"
            f"  p + p.T - p * p.T, not the average.\n"
        )

    print(f"  validated against your compute_symmetric_p on a {size}x{size} block  ok")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute the global UMAP probability graph."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N training images (smoke test; training will "
        "refuse to use the result unless it covers the whole subset)",
    )
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--out", default=GLOBAL_P_PATH)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.block_size <= args.n_neighbors:
        raise SystemExit(
            f"--block-size ({args.block_size}) must exceed "
            f"--n-neighbors ({args.n_neighbors})"
        )

    verify_student_code()
    device = pick_device(args.device)

    print("=" * 66)
    print("  GLOBAL UMAP PROBABILITIES")
    print("=" * 66)
    print(f"  device      : {device}")
    print(f"  k neighbours: {args.n_neighbors}")

    started = time.time()

    print("\n[1/4] loading the training subset into memory")
    images = load_train_images(DATA_DIR, limit=args.limit)
    n = images.shape[0]
    print(f"  {n} images, {images.numel() * 4 / 1e6:.0f} MB")

    print("\n[2/4] exact k-nearest neighbours")
    knn_distances, knn_indices = compute_knn(images, args.n_neighbors, device)

    print("\n[3/4] sigma and directed weights (your student_tasks.py)")
    sigma, p_directed = compute_directed_weights(
        knn_distances, args.n_neighbors, args.block_size
    )

    print("\n[4/4] symmetrising")
    rows = np.repeat(np.arange(n), args.n_neighbors)
    p_asymmetric = sp.csr_matrix(
        (
            p_directed.reshape(-1).numpy().astype(np.float32),
            (rows, knn_indices.reshape(-1).numpy()),
        ),
        shape=(n, n),
    )
    p_symmetric = symmetrize(p_asymmetric)
    validate_symmetrisation(p_asymmetric, p_symmetric)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sp.save_npz(args.out, p_symmetric)
    with open(GLOBAL_P_META, "w") as handle:
        json.dump(
            {
                "n_samples": int(n),
                "n_neighbors": int(args.n_neighbors),
                "seed": int(SEED),
                "val_fraction": float(VAL_FRACTION),
                "limited": args.limit is not None,
            },
            handle,
            indent=2,
        )

    nnz = p_symmetric.nnz
    density = nnz / float(n * n)
    size_mb = os.path.getsize(args.out) / 1e6

    print("\n" + "=" * 66)
    print(f"  saved            : {args.out}  ({size_mb:.1f} MB)")
    print(f"  shape            : {n} x {n}")
    print(f"  stored edges     : {nnz:,}   (density {density:.2e})")
    print(f"  mean degree      : {nnz / n:.1f} neighbours per point")
    print(f"  sigma            : min {sigma.min():.4f}  "
          f"median {sigma.median():.4f}  max {sigma.max():.4f}")
    print(f"  rho              : min {knn_distances[:, 0].min():.4f}  "
          f"median {knn_distances[:, 0].median():.4f}")
    print(f"  elapsed          : {time.time() - started:.1f}s")
    print("=" * 66)

    if args.limit is not None:
        print(
            "\n  NOTE: --limit was used, so this covers only part of the\n"
            "  training subset. boilerplate.py will refuse it. Re-run without\n"
            "  --limit before training for real."
        )
    else:
        print("\n  Next:  python boilerplate.py")


if __name__ == "__main__":
    main()
