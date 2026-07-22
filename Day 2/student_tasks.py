"""
Parametric UMAP — Student Tasks
================================

Implement every function below. Replace each `raise NotImplementedError` with
your own code.

RULES
-----
1. Use PURE PyTorch tensor operations. No Python `for` loops over the batch,
   no list comprehensions over elements, no numpy round-trips.
   (The ONLY exception is `find_a_b`, which is an offline pre-computation and
   is allowed to use scipy.)
2. Never break the autograd graph. Do not call `.numpy()`, `.item()`,
   `.detach()`, or `torch.no_grad()` inside `umap_loss`.
3. Your functions must be dtype- AND device-agnostic. They will be graded with
   float64 CPU tensors, while your training loop passes float32 tensors living
   on whatever accelerator Lightning picked (CUDA / MPS). So every tensor you
   create from scratch must inherit both properties from an input:

       mask = torch.eye(n, dtype=torch.bool, device=distances.device)
       lo   = torch.zeros(n, dtype=distances.dtype, device=distances.device)

   Forget `device=` and it still works on CPU, then dies the moment you train
   on a GPU with "expected self and mask to be on the same device".

Checking your work
------------------
Your submission is graded by a test suite you do not have: it feeds fixed
tensors into each function and compares against expected values to 4 decimal
places. You cannot run it, so you cannot tune against it -- the docstrings
below are the complete specification, and where one pins down an exact
procedure (the binary search in `find_sigma`, the fit grid in `find_a_b`),
follow it literally rather than improvising an equivalent.

Two feedback loops you DO have:

    python selfcheck.py     structural checks -- shapes, ranges, symmetry,
                            differentiability. Catches most mistakes, but
                            passing it is necessary, not sufficient.

    python generate_global_probabilities.py   builds the UMAP graph over all
                            54k training images using your code, then
                            cross-checks your compute_symmetric_p.

    python boilerplate.py   watch `val_knn_acc` in the progress bar. If your
                            math is wrong it will sit near 0.10 (chance).
"""

import torch

# `find_a_b` is an offline pre-computation, so scipy is allowed there (and
# ONLY there).
from scipy.optimize import curve_fit  # noqa: F401


def compute_min_pairwise_distance(images: torch.Tensor) -> torch.Tensor:
    r"""Compute rho_i, the distance from each point to its nearest neighbour.

    In UMAP, rho_i is the local connectivity offset. Subtracting it guarantees
    every point is connected to at least one neighbour with weight exactly 1.0,
    which is what stops the fuzzy graph from fragmenting in sparse regions.

    Math
    ----
        x_i          = flatten(images[i])                    in R^784
        d_ij         = || x_i - x_j ||_2                     (Euclidean)
        rho_i        = min_{j != i} d_ij

    The `j != i` exclusion is essential: d_ii is always 0, so without masking
    the diagonal out you would get rho_i = 0 for every point.

    Parameters
    ----------
    images : torch.Tensor
        Raw image batch of shape (B, 1, 28, 28). The DataLoader hands you
        images in this layout and does NOT flatten them for you.

    Returns
    -------
    torch.Tensor
        Shape (B,). rho_i for each point in the batch.

    Implementation notes
    --------------------
    * Flatten the spatial dimensions yourself: (B, 1, 28, 28) -> (B, 784).
      `Tensor.reshape` / `Tensor.view` / `torch.flatten` all work.
    * Build the full (B, B) distance matrix. `torch.cdist` is the direct route;
      you may also broadcast manually.
    * To exclude the diagonal, fill it with `float('inf')` BEFORE taking the
      min. Use `Tensor.masked_fill` with a boolean `torch.eye` mask.
      Do NOT write `distances + torch.eye(B) * float('inf')` — the off-diagonal
      entries become `0 * inf = nan` and silently poison the whole matrix.
    """
    B = images.shape[0]
    flat_images = images.reshape(B, -1)
    distances = torch.cdist(flat_images, flat_images)
    mask = torch.eye(B, dtype=torch.bool, device=distances.device)
    distances_masked = distances.masked_fill(mask, float('inf'))
    rho, _ = torch.min(distances_masked, dim=1)
    return rho


def find_a_b(min_dist: float = 0.1, spread: float = 1.0) -> tuple[float, float]:
    r"""Fit the low-dimensional similarity kernel's (a, b) hyper-parameters.

    UMAP models similarity in the 2D embedding space with a smooth,
    differentiable curve:

        phi(d) = (1 + a * d^(2b))^(-1)

    That curve is fitted, once and offline, to the piecewise target that
    actually encodes what we want -- points closer than `min_dist` are treated
    as identical, and similarity decays exponentially beyond it:

                    | 1                                  if d <= min_dist
        Psi(d)  =   |
                    | exp(-(d - min_dist) / spread)       if d >  min_dist

    Parameters
    ----------
    min_dist : float
        Minimum separation between points in the embedding. Smaller values
        produce tighter, more clumped clusters.
    spread : float
        Scale over which the target similarity decays.

    Returns
    -------
    tuple[float, float]
        The fitted `(a, b)`, as plain Python floats.

    Implementation notes
    --------------------
    * This runs ONCE before training, not per batch, so `scipy.optimize.curve_fit`
      is allowed here (and only here).
    * Sample the target on a grid: `np.linspace(0, 3 * spread, 300)`.
      The grid must start at 0 and use exactly 300 points -- a different grid
      shifts the fit and will not match the expected values.
    * Use the default initial guess `p0=[1.0, 1.0]`.
    * Sanity check: `find_a_b(0.1, 1.0)` should give a ~= 1.5769, b ~= 0.8951.
    """
    import numpy as np
    x = np.linspace(0, 3 * spread, 300)
    y = np.where(x <= min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    
    def phi(d, a, b):
        return 1.0 / (1.0 + a * d ** (2 * b))
        
    popt, _ = curve_fit(phi, x, y, p0=[1.0, 1.0])
    return float(popt[0]), float(popt[1])


def find_sigma(
    distances: torch.Tensor,
    k: int = 15,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> torch.Tensor:
    r"""Binary-search the per-point bandwidth sigma_i.

    sigma_i is the local scale that normalises each point's neighbourhood, so
    that every point has the same *effective* number of neighbours regardless
    of how dense its region of the manifold is. It is chosen so the fuzzy
    weights around point i sum to a fixed budget:

        sum_{j != i} exp( -max(0, d_ij - rho_i) / sigma_i )  =  log2(k)

    The left side increases monotonically in sigma_i, which is exactly why a
    binary search works.

    Parameters
    ----------
    distances : torch.Tensor
        Symmetric (B, B) pairwise distance matrix with a zero diagonal.
    k : int
        Target number of neighbours. Sets the budget `log2(k)`.
    tol : float
        Convergence tolerance on the SUM (not on sigma).
    max_iter : int
        Hard cap on binary-search iterations.

    Returns
    -------
    torch.Tensor
        Shape (B,). sigma_i for each point.

    Implementation notes
    --------------------
    * Compute rho internally from `distances` (same rule as
      `compute_min_pairwise_distance`: row-wise min excluding the diagonal).
    * Exclude the diagonal from the sum, exactly as in the formula above.
    * Vectorise across the batch: all B searches advance together, each row
      carrying its own `lo` / `hi` / `mid`. Use `torch.where`, not a Python
      loop over rows.

    EXACT ALGORITHM -- follow it literally
    -------------------------------------
    `tol` and `max_iter` are fixed by the assignment so that every student
    converges on bit-identical floats. Binary search stops on a tolerance, so
    a different update schedule lands on a *slightly* different sigma that
    still satisfies the tolerance -- and grading compares to 4 decimals. A
    "morally equivalent" search will silently lose those marks, so do exactly
    this::

        lo   = 0.0            (per row)
        hi   = +inf           (per row)
        mid  = 1.0            (per row)

        repeat max_iter times:
            psum = sum_{j != i} exp(-max(0, d_ij - rho_i) / mid_i)

            if all |psum - log2(k)| < tol:
                break

            for each row i, in parallel:
                if psum_i > log2(k):        # neighbourhood too wide
                    hi_i  = mid_i
                    mid_i = (lo_i + hi_i) / 2
                else:                       # neighbourhood too narrow
                    lo_i  = mid_i
                    mid_i = mid_i * 2       if hi_i is +inf
                            (lo_i + hi_i)/2 otherwise

        return mid

    Note the rows are updated jointly but the *break* is global: iteration
    stops only once every row is within tolerance.
    """
    import math
    B = distances.shape[0]
    
    mask = torch.eye(B, dtype=torch.bool, device=distances.device)
    distances_masked = distances.masked_fill(mask, float('inf'))
    rho, _ = torch.min(distances_masked, dim=1)
    
    lo = torch.zeros(B, dtype=distances.dtype, device=distances.device)
    hi = torch.full((B,), float('inf'), dtype=distances.dtype, device=distances.device)
    mid = torch.ones(B, dtype=distances.dtype, device=distances.device)
    
    target = math.log2(k)
    
    for _ in range(max_iter):
        diff = distances - rho.unsqueeze(1)
        diff_clamped = torch.clamp(diff, min=0.0)
        p_ij = torch.exp(-diff_clamped / mid.unsqueeze(1))
        
        p_ij_masked = p_ij.masked_fill(mask, 0.0)
        psum = p_ij_masked.sum(dim=1)
        
        if torch.all(torch.abs(psum - target) < tol):
            break
            
        too_wide = psum > target
        
        hi = torch.where(too_wide, mid, hi)
        lo = torch.where(too_wide, lo, mid)
        
        hi_is_inf = torch.isinf(hi)
        mid_if_narrow = torch.where(hi_is_inf, mid * 2.0, (lo + hi) / 2.0)
        
        mid = torch.where(too_wide, (lo + hi) / 2.0, mid_if_narrow)
        
    return mid


def compute_asymmetric_p(
    distances: torch.Tensor,
    rho: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the directed (asymmetric) fuzzy membership strengths.

    Math
    ----
        p_{j|i} = exp( -max(0, d_ij - rho_i) / sigma_i )

    This is "how strongly does i believe j is its neighbour". It is asymmetric
    because rho_i and sigma_i are local to i: in general p_{j|i} != p_{i|j}.

    The `max(0, .)` clamp means the nearest neighbour of i (at distance rho_i)
    gets weight exactly exp(0) = 1.0, and nothing can exceed 1.0.

    Parameters
    ----------
    distances : torch.Tensor
        Symmetric (B, B) pairwise distance matrix.
    rho : torch.Tensor
        Shape (B,). Nearest-neighbour distances.
    sigma : torch.Tensor
        Shape (B,). Per-point bandwidths.

    Returns
    -------
    torch.Tensor
        Shape (B, B). Entry [i, j] is p_{j|i}. The diagonal is set to 0.0 --
        a point is not its own neighbour.

    Implementation notes
    --------------------
    * `rho` and `sigma` are indexed by i, which is the ROW. Broadcast them
      along rows with `rho.unsqueeze(1)` / `sigma.unsqueeze(1)`, not
      `unsqueeze(0)`. Getting this backwards transposes your whole graph and
      is the single most common bug in this assignment.
    * Clamp with `torch.clamp(x, min=0.0)`.
    * Zero the diagonal with `masked_fill` and a boolean `torch.eye` mask.
    """
    diff = distances - rho.unsqueeze(1)
    diff_clamped = torch.clamp(diff, min=0.0)
    p_asym = torch.exp(-diff_clamped / sigma.unsqueeze(1))
    
    mask = torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    p_asym = p_asym.masked_fill(mask, 0.0)
    return p_asym


def compute_symmetric_p(p_asym: torch.Tensor) -> torch.Tensor:
    r"""Symmetrise the directed graph via the probabilistic t-conorm.

    Math
    ----
        p_ij = p_{j|i} + p_{i|j} - p_{j|i} * p_{i|j}

    This is fuzzy-set union: "i considers j a neighbour OR j considers i a
    neighbour". Note it is NOT the average -- the union keeps an edge alive
    when either endpoint is confident about it, which is what preserves
    connections into sparse regions of the manifold.

    The result stays in [0, 1] and is symmetric by construction.

    Parameters
    ----------
    p_asym : torch.Tensor
        Shape (B, B). Entry [i, j] is p_{j|i}, from `compute_asymmetric_p`.

    Returns
    -------
    torch.Tensor
        Shape (B, B). The symmetric p_ij.

    Implementation notes
    --------------------
    * p_{i|j} is just the transpose of p_{j|i}: use `p_asym.t()` (or `.T`).
    * The whole thing is one line of elementwise tensor arithmetic.
    """
    p_i_j = p_asym.t()
    return p_asym + p_i_j - p_asym * p_i_j


def umap_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""Fuzzy-set cross-entropy between the high-D graph p and the low-D graph q.

    Math
    ----
        C = sum_ij [  p_ij * log(p_ij / q_ij)
                    + (1 - p_ij) * log((1 - p_ij) / (1 - q_ij))  ]

    The first term is ATTRACTIVE: it pulls together points that are neighbours
    in the original 784-D space. The second is REPULSIVE: it pushes apart
    points that are not. Their balance is what produces UMAP's characteristic
    well-separated clusters.

    Reduction is a plain `.sum()` over every entry of the inputs as given.

    !! SHAPE-AGNOSTIC -- REQUIRED !!
    --------------------------------
    `p` and `q` are two tensors OF THE SAME SHAPE holding one pair-membership
    each. That shape is not part of the contract: it may be `(B, B)` for a
    dense block of pairs, or `(E,)` for a flat list of sampled pairs. Training
    uses the flat form; you will be graded on both.

    So write this function PURELY ELEMENTWISE, plus the final `.sum()`:

    * NO indexing or slicing (`p[i, j]`, `p[0]`, `torch.diagonal`).
    * NO shape-dependent masking (`torch.eye`, `masked_fill`, `fill_diagonal_`).
    * NO `.reshape()`, `.view()`, `.t()`, or anything assuming 2-D.
    * NO `p.shape[0]` -- if you find yourself needing a dimension, stop.

    The clamps below are all the special-casing this function needs. Zeroing a
    diagonal is the caller's job, not yours: on an `(E,)` edge list there is no
    diagonal to zero.

    Parameters
    ----------
    p : torch.Tensor
        Target memberships from the high-dimensional space. Treated as a
        constant -- no gradient flows into `p`.
    q : torch.Tensor
        Same shape as `p`. Memberships computed from the 2D embedding, via
        q = (1 + a * d^(2b))^(-1). Gradients flow through this.

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) tensor. THIS IS THE TRAINING LOSS -- it must carry a
        live `grad_fn`.

    !! CLAMPING -- READ THIS !!
    --------------------------
    `q` can legitimately reach 0.0 or 1.0 in the embedding. If it does:

        log(p / 0)      -> +inf        loss becomes `nan`
        log((1-p) / 0)  -> +inf        gradients become `nan`

    and once a `nan` reaches the optimiser every weight in your network is
    destroyed on the very next step -- permanently, and silently.

    So clamp q into a safe interval BEFORE the logs, using eps = 1e-4::

        q = torch.clamp(q, 1e-4, 1.0 - 1e-4)

    Clamp `p` the same way. `p` legitimately contains exact 0.0 entries (the
    diagonal), which would otherwise give `0 * log(0) = nan` -- mathematically
    that term is 0, but PyTorch will not do that for you.

    !! DIFFERENTIABILITY -- STRICT CONSTRAINT !!
    --------------------------------------------
    This function is checked with `torch.autograd.gradcheck`. It must be
    written entirely in differentiable torch ops:

    * NO `.numpy()`, `.item()`, `.tolist()`, `.detach()`, `float(...)`.
    * NO `math.log` / `np.log`. Use `torch.log`.
    * NO `torch.no_grad()`, and no in-place writes into `q` (`q[mask] = ...`).
      `torch.clamp` returns a new tensor and is safely differentiable --
      prefer it over any manual masking.
    """
    p_clamped = torch.clamp(p, 1e-4, 1.0 - 1e-4)
    q_clamped = torch.clamp(q, 1e-4, 1.0 - 1e-4)
    
    term1 = p_clamped * torch.log(p_clamped / q_clamped)
    term2 = (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - q_clamped))
    
    return (term1 + term2).sum()
