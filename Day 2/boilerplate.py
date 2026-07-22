"""
Parametric UMAP — Boilerplate (fully implemented)
=================================================

Data loading, CNN encoder, the PyTorch Lightning module, and the training
entry point.

You do not need to modify this file to complete the assignment, with one
exception: the clearly marked TODO block inside `training_step`.

    python generate_global_probabilities.py    # once, first
    python boilerplate.py

The UMAP target graph `p_ij` is built ONCE over the whole 54k training subset
by `generate_global_probabilities.py`, which calls your `student_tasks.py`.
Training loads that sparse graph and, for each batch, looks up the (B, B)
sub-block for the samples it contains -- which is why the training DataLoader
yields `(image, label, index)` rather than just `(image, label)`.

The 60k MNIST train split is divided into 54k train / 6k validation. After
every epoch the validation loop embeds the held-out images and scores them
with the same KNN probe used for grading, logged as `val_knn_acc` — so you
get a live estimate of your final score while training, and the checkpoint
that gets kept is the one that scored best on it.

Determinism
-----------
Two runs with the same SEED produce the same numbers, so a change in your
score is a change you caused. Every source of randomness is pinned:

  * `L.seed_everything(SEED)`         python / numpy / torch global RNGs
  * `random_split(generator=...)`     which images land in the val split
  * `DataLoader(generator=...)`       batch order each epoch
  * `worker_init_fn=seed_worker`      numpy+random inside DataLoader workers
  * re-seed before model construction CNN weight init
  * `train_test_split(random_state=)` the KNN probe's reference/query halves
  * `Trainer(deterministic=True)`     deterministic cuDNN/CUDA kernels
  * `CUBLAS_WORKSPACE_CONFIG`         set at the top, required by the above

If you change SEED your numbers will move. That is expected — just report
which seed you used, and change it only deliberately.
"""

from __future__ import annotations

import math
import os
import random
import shutil

# Must be set BEFORE the first cuBLAS call, so before torch initialises a CUDA
# context. With `Trainer(deterministic=True)` and CUDA >= 10.2, cuBLAS refuses
# to run deterministically unless this is configured, and torch raises rather
# than silently going nondeterministic. Harmless on CPU/MPS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from torch.utils.data import DataLoader, Subset, random_split  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.datasets import MNIST  # noqa: E402
from student_tasks import umap_loss
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
FINAL_CHECKPOINT = f"{CHECKPOINT_DIR}/model.ckpt"

# The global probability graph, written by `generate_global_probabilities.py`.
ARTIFACT_DIR = "./artifacts"
GLOBAL_P_PATH = f"{ARTIFACT_DIR}/global_p.npz"
GLOBAL_P_META = f"{ARTIFACT_DIR}/global_p.json"

# Fallback batch size for the generic `get_dataloader` helper. Training does
# NOT use it -- it iterates edges, sized by EDGE_BATCH_SIZE below -- so raising
# this will not change your training run.
BATCH_SIZE = 256
VAL_BATCH_SIZE = 512  # no NxN graph is built at val time, so this can be big

# Edge sampling needs real mileage over the graph: each step scores only
# EDGE_BATCH_SIZE * (1 + N_NEGATIVE) pairs, so an epoch here is far less work
# than a dense-block epoch. Measured val_knn_acc on the reference solution:
#
#     10 epochs -> 0.40      40 epochs -> 0.53   (still climbing)
#
# 40 epochs takes about 4.5 minutes. Raise it further if you have the time.
MAX_EPOCHS = 80
LEARNING_RATE = 1e-3
SEED = 42

# Fraction of the 60k MNIST train split held out for validation.
# 0.1 -> 54,000 train / 6,000 val.
VAL_FRACTION = 0.1

# UMAP hyper-parameters.
N_NEIGHBORS = 15  # -> k, the neighbourhood budget log2(k)
MIN_DIST = 0.1  # -> feeds find_a_b
SPREAD = 1.0  # -> feeds find_a_b

# Edge sampling. Training iterates EDGES of the global graph, not images.
EDGE_BATCH_SIZE = 512  # positive edges per step
N_NEGATIVE = 5  # negative pairs drawn per positive edge
EDGES_PER_EPOCH = 100_000  # edges visited per epoch (graph has ~600k)

# KNN probe (the topology metric). Fixed at 5 by the assignment — this is the
# same k your work is graded with, so leave it alone.
PROBE_NEIGHBORS = 5


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class MNISTTrainOnly(MNIST):
    """MNIST restricted to the train split at the *download* level.

    `torchvision.datasets.MNIST` fetches all four archives (train AND t10k)
    even when `train=True`, because `download` iterates the full `resources`
    list. Trimming that list to the first two entries means the test-set
    archives are never pulled over the network at all.
    """

    resources = MNIST.resources[:2]  # train-images-idx3, train-labels-idx1

    @property
    def raw_folder(self) -> str:
        # torchvision derives this from the class name, which would give
        # `data/MNISTTrainOnly/raw`. Pin it to the conventional location so an
        # already-downloaded MNIST is reused instead of re-fetched.
        return os.path.join(self.root, "MNIST", "raw")


def get_datasets(
    data_dir: str = DATA_DIR,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> tuple[Subset, Subset]:
    """Split the MNIST train split into disjoint train / validation subsets.

    The MNIST *test* split is never downloaded or touched. We carve the
    validation set out of the 60k training images instead, so the held-out
    images are images the encoder has genuinely never been fitted on.

    The split is driven by a seeded generator, so it is identical on every
    machine and across every run -- your validation score is comparable to
    your classmates'.
    """
    dataset = MNISTTrainOnly(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),  # uint8 [0,255] -> float32 [0,1]
    )

    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val

    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def materialize_images(dataset) -> torch.Tensor:
    """Load a dataset into one `(N, 1, 28, 28)` float32 tensor.

    54,000 x 784 x 4 bytes is about 169 MB, so the training subset sits in RAM
    comfortably and edge lookups become plain tensor indexing.
    """
    images = torch.empty(len(dataset), 1, 28, 28, dtype=torch.float32)
    for i in range(len(dataset)):
        image, _label = dataset[i]
        images[i] = image
    return images


class EdgeDataset(torch.utils.data.Dataset):
    """Iterates the EDGES of the global graph, not the images.

    WHY EDGES -- this is the whole point
    ------------------------------------
    The graph is 0.04% dense. Sample 256 images at random and almost none of
    them are neighbours, so a dense (B, B) target block is ~99.95% zeros and
    `umap_loss` sees ~34 attractive pairs against ~65,500 repulsive ones. That
    600:1 imbalance makes training push everything apart and nothing together.

    Iterating edges inverts the problem. Every sample IS a real neighbour pair
    with a real p_ij, and repulsion is added deliberately by drawing
    `n_negative` random pairs per positive -- so the ratio becomes a knob you
    set (5:1) rather than an accident of graph density (600:1).

    Yields `(head_image, tail_image, p_ij)`.
    """

    def __init__(
        self,
        images: torch.Tensor,
        heads: np.ndarray,
        tails: np.ndarray,
        p_values: np.ndarray,
    ) -> None:
        self.images = images
        self.heads = torch.from_numpy(heads.astype(np.int64))
        self.tails = torch.from_numpy(tails.astype(np.int64))
        self.p_values = torch.from_numpy(p_values.astype(np.float32))

    @classmethod
    def from_sparse(cls, images: torch.Tensor, p_global) -> "EdgeDataset":
        """Build from the CSR graph, keeping each undirected edge once."""
        import scipy.sparse as sp

        # p is symmetric, so the upper triangle holds every edge exactly once.
        upper = sp.triu(p_global, k=1).tocoo()
        return cls(images, upper.row, upper.col, upper.data)

    def __len__(self) -> int:
        return len(self.heads)

    def __getitem__(self, edge: int):
        return (
            self.images[self.heads[edge]],
            self.images[self.tails[edge]],
            self.p_values[edge],
        )


def load_global_p(
    path: str = GLOBAL_P_PATH,
    expected_samples: int | None = None,
):
    """Load the sparse global probability graph from disk.

    Returns a SciPy CSR matrix. Kept sparse: dense would be 11.7 GB, while the
    k-NN graph holds well under a million edges.
    """
    import scipy.sparse as sp

    if not os.path.exists(path):
        raise SystemExit(
            f"\n  No global probability graph at '{path}'.\n\n"
            f"  Build it first:\n\n"
            f"      python generate_global_probabilities.py\n\n"
            f"  It needs a finished student_tasks.py, so run\n"
            f"  `python selfcheck.py` first if you are unsure.\n"
        )

    p_global = sp.load_npz(path).tocsr()

    if expected_samples is not None and p_global.shape[0] != expected_samples:
        raise SystemExit(
            f"\n  '{path}' covers {p_global.shape[0]} samples but the training "
            f"subset has {expected_samples}.\n"
            f"  This usually means it was generated with --limit, or that SEED "
            f"/ VAL_FRACTION changed.\n"
            f"  Re-run:  python generate_global_probabilities.py\n"
        )

    return p_global


def seed_worker(worker_id: int) -> None:
    """Re-seed a DataLoader worker process.

    Each worker inherits a fresh torch seed from the parent's generator, but
    numpy's and Python's RNGs are NOT seeded for it. Anything using them (a
    custom augmentation, say) would be nondeterministic across workers. This
    is a no-op at `num_workers=0`; it matters the moment you raise it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    seed: int = SEED,
) -> DataLoader:
    """Wrap a dataset in a DataLoader.

    The loader yields raw `(B, 1, 28, 28)` image tensors, untouched. Flattening
    to `(B, 784)` is deliberately left to you -- it is part of
    `compute_min_pairwise_distance` in `student_tasks.py`.

    Shuffling is driven by an explicitly seeded generator rather than the
    global torch RNG. That matters: with the global RNG, the batch order would
    depend on how much randomness happened to be consumed earlier in the
    process, so adding a single `torch.randn` anywhere above would silently
    change every epoch's batches.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
    )


def get_dataloaders(
    edge_batch_size: int = EDGE_BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    data_dir: str = DATA_DIR,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    edges_per_epoch: int = EDGES_PER_EPOCH,
) -> tuple[DataLoader, DataLoader]:
    """Build the (train, val) DataLoader pair used by `main()`.

    The train loader yields EDGES `(head_image, tail_image, p_ij)`; the val
    loader yields plain `(image, label)` pairs, since the KNN probe only ever
    needs to embed images.
    """
    train_ds, val_ds = get_datasets(data_dir, val_fraction, seed)

    global_p = load_global_p(expected_samples=len(train_ds))
    edge_ds = EdgeDataset.from_sparse(materialize_images(train_ds), global_p)

    # A full pass over ~600k edges per epoch is more compute than this
    # workshop needs, so each epoch visits a uniform random subsample. Uniform,
    # not p-weighted: p_ij is already an explicit factor inside `umap_loss`,
    # and sampling proportional to it as well would count it twice.
    sampler = torch.utils.data.RandomSampler(
        edge_ds,
        replacement=True,
        num_samples=min(edges_per_epoch, len(edge_ds)),
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        edge_ds,
        batch_size=edge_batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    val_loader = get_dataloader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------
# The KNN topology probe
# --------------------------------------------------------------------------


def knn_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = PROBE_NEIGHBORS,
    holdout: bool = True,
    seed: int = SEED,
) -> float:
    """Score a 2D embedding by how well its neighbourhoods agree on labels.

    This is the metric your submission is graded on, computed here exactly as
    it is computed at grading time. Leave it alone: changing it only changes
    the number you see, not the number you get.

    Parameters
    ----------
    holdout : bool
        True  -- fit the classifier on a stratified half of the points and
                 score it on the other half. An honest generalisation estimate.
        False -- fit and score on the same points. Optimistically biased:
                 `KNeighborsClassifier` counts a point as one of its own 5
                 neighbours, so one vote is guaranteed correct, and it also
                 gets twice the reference points. Expect it to read clearly
                 higher than the holdout number, especially early in training.
    """
    n_classes = len(np.unique(labels))
    if len(labels) < 2 * (n_neighbors + 1) or n_classes < 2:
        # Too few points to probe — happens during Lightning's 2-batch sanity
        # check before training starts.
        return float("nan")

    if not holdout:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(embeddings, labels)
        return float(knn.score(embeddings, labels))

    x_ref, x_query, y_ref, y_query = train_test_split(
        embeddings,
        labels,
        test_size=0.5,
        random_state=seed,
        stratify=labels,
    )
    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_ref, y_ref)
    return float(knn.score(x_query, y_query))


@torch.no_grad()
def embed_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the encoder over a loader and return `(embeddings (N, 2), labels (N,))`."""
    was_training = model.training
    model.eval()

    embeddings, labels = [], []
    for i, (images, batch_labels) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(next(model.parameters()).device)
        embeddings.append(model(images).cpu())
        labels.append(batch_labels.cpu())

    if was_training:
        model.train()

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """A deliberately small conv encoder: (B, 1, 28, 28) -> (B, 2).

    ######################################################################
    #                                                                    #
    #   THIS ARCHITECTURE IS A PLACEHOLDER -- REPLACE IT IF YOU WANT.    #
    #                                                                    #
    #   Nothing else in the codebase depends on what happens between     #
    #   the input and the output. Swap in a ResNet, add BatchNorm,       #
    #   go deeper, use a ViT -- whatever you like.                       #
    #                                                                    #
    #   The ONLY two contracts you must honour:                          #
    #     * input  is (B, 1, 28, 28)                                     #
    #     * output is (B, 2)   <- the 2D embedding coordinates           #
    #                                                                    #
    #   A better encoder is one of the easiest ways to improve your      #
    #   final KNN-probe score.                                           #
    #                                                                    #
    ######################################################################
    """

    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32,  7,  7)
        )

        self.head = nn.Sequential(
            nn.Flatten(),  # (B, 1568)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # (B, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Lightning module
# --------------------------------------------------------------------------


class ParametricUMAP(L.LightningModule):
    """Trains `SimpleCNN` so that its 2D output reproduces MNIST's topology."""

    def __init__(
        self,
        latent_dim: int = 2,
        lr: float = LEARNING_RATE,
        n_neighbors: int = N_NEIGHBORS,
        min_dist: float = MIN_DIST,
        spread: float = SPREAD,
        probe_neighbors: int = PROBE_NEIGHBORS,
        n_negative: int = N_NEGATIVE,
    ) -> None:
        super().__init__()
        # Records every __init__ arg into the checkpoint so the model can be
        # rebuilt with `ParametricUMAP.load_from_checkpoint(path)` and no
        # arguments. Grading relies on this — do not remove it.
        #
        self.save_hyperparameters()

        self.encoder = SimpleCNN(latent_dim=latent_dim)

        # (a, b) define the low-dimensional similarity kernel. They depend only
        # on `min_dist` / `spread`, so they are fitted ONCE here rather than
        # per batch. Requires `find_a_b` to be implemented.
        from student_tasks import find_a_b

        a, b = find_a_b(min_dist=min_dist, spread=spread)
        # Buffers, not parameters: they travel with the model to GPU and into
        # the checkpoint, but are never optimised.
        self.register_buffer("a", torch.tensor(float(a)))
        self.register_buffer("b", torch.tensor(float(b)))

        # Scratch buffers for the validation KNN probe. Cleared each epoch.
        self._val_embeddings: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, 2)."""
        return self.encoder(x)

    def sample_negatives(self, z_tail: torch.Tensor) -> torch.Tensor:
        """Draw `n_negative` negative partners for each positive edge.

        Fully implemented — just call it.

        Reshuffling the batch's tail embeddings is the standard trick: pairing
        head i with some other edge's tail gives a pair that is almost
        certainly NOT an edge of the graph (density is 0.04%), so its target
        membership is p = 0. No extra images are loaded and no lookup is
        needed.

        The shuffle is a random CYCLIC SHIFT rather than a full permutation,
        and that is not cosmetic. The backward pass of `z_tail[perm]` is an
        `index_put` with accumulation, which has no deterministic GPU kernel --
        under `Trainer(deterministic=True)` it raises rather than silently
        giving up reproducibility. `torch.roll` is a permutation too, and its
        backward is just another roll, so it stays deterministic everywhere.
        Since the batch is already a random sample of edges, pairing head i
        with tail i+k is as good a random negative as any.

        Shifts are drawn from [1, E) so a head is never paired back with its
        own true tail, which would be a false negative.

        Returns `(n_negative * E, 2)` — stack it against a repeated head.
        """
        n_edges = z_tail.shape[0]
        shifts = torch.randint(1, max(n_edges, 2), (self.hparams.n_negative,))
        return torch.cat(
            [torch.roll(z_tail, int(shift), dims=0) for shift in shifts]
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # One POSITIVE EDGE per row: two images that are neighbours in the
        # global graph, plus the strength p_ij of that edge. Labels are never
        # involved — this is unsupervised.
        head_images, tail_images, p_edge = batch

        z_head = self(head_images)  # (E, 2)
        z_tail = self(tail_images)  # (E, 2)

        # //////////////////////////////////////////////////////////////////
        # ///                                                            ///
        # ///                  TODO -- YOUR WORK GOES HERE               ///
        # ///                                                            ///
        # //////////////////////////////////////////////////////////////////
        #
        # Everything above and below this block is done. You must now build
        # the loss by IMPORTING AND CALLING the function you wrote in
        # `student_tasks.py`.
        #
        # Add this import at the top of the file:
        #
        #     from student_tasks import umap_loss
        #
        # ---- STEP 1: the low-dimensional similarity kernel ----------------
        #
        # Write a helper that turns two sets of 2D coordinates into their
        # membership strengths, using `self.a` and `self.b` (fitted for you in
        # __init__ by your `find_a_b`):
        #
        #     q(u, v) = (1 + a * ||u - v||^(2b))^(-1)
        #
        #   a) squared_distance = ((u - v) ** 2).sum(-1)          -> (E,)
        #   b) q = 1 / (1 + a * (squared_distance + 1e-8) ** b)   -> (E,)
        #
        #      The 1e-8 matters: d^(2b) has an infinite derivative at d = 0
        #      when b < 0.5, and a head can coincide with its tail.
        #      Note ||u-v||^(2b) == (squared_distance)^b, so you never need
        #      a square root.
        #
        # ---- STEP 2: positives and negatives ------------------------------
        #
        #   c) q_pos = q(z_head, z_tail)                          -> (E,)
        #
        #      These are real edges, so their targets are the p_edge values
        #      the DataLoader handed you.
        #
        #   d) z_negative = self.sample_negatives(z_tail)         -> (n*E, 2)
        #      q_neg      = q(z_head.repeat(n, 1), z_negative)    -> (n*E,)
        #
        #      where n = self.hparams.n_negative. These pairs are (almost
        #      certainly) not edges, so their target is p = 0.
        #
        # ---- STEP 3: the loss ---------------------------------------------
        #
        # Concatenate both groups into one flat list of pairs and score them
        # in a single call. `umap_loss` is elementwise, so it does not care
        # that these are (E,) vectors rather than a (B, B) matrix:
        #
        #   e) p_all = torch.cat([p_edge, torch.zeros_like(q_neg)])
        #      q_all = torch.cat([q_pos, q_neg])
        #      loss  = umap_loss(p_all, q_all)
        #
        # p_edge pulls true neighbours together; the zeros push everything
        # else apart. `n_negative` sets the balance between them.
        #
        # Then DELETE the `raise` below.
        #
        # //////////////////////////////////////////////////////////////////

        def q(u, v):
            squared_distance = ((u - v) ** 2).sum(-1)
            return 1.0 / (1.0 + self.a * (squared_distance + 1e-8) ** self.b)
            
        q_pos = q(z_head, z_tail)
        
        n = self.hparams.n_negative
        z_negative = self.sample_negatives(z_tail)
        q_neg = q(z_head.repeat(n, 1), z_negative)
        
        p_all = torch.cat([p_edge, torch.zeros_like(q_neg)])
        q_all = torch.cat([q_pos, q_neg])
        loss = umap_loss(p_all, q_all)

        # //////////////////////////////////////////////////////////////////
        # ///                  END OF YOUR WORK                          ///
        # //////////////////////////////////////////////////////////////////

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)  # noqa: F821
        return loss  # noqa: F821

    # ----------------------------------------------------------------------
    # Validation — the KNN topology probe
    # ----------------------------------------------------------------------
    #
    # This is fully implemented; you do not need to touch it.
    #
    # Validation deliberately does NOT compute the UMAP loss. It only runs the
    # encoder forward and scores the resulting 2D cloud with a KNN classifier.
    # Two reasons:
    #
    #   1. The UMAP loss is computed over a within-batch graph, so its value
    #      depends on batch composition and is a poor progress signal. KNN
    #      accuracy measures the thing you actually care about — whether the
    #      manifold's neighbourhood structure survived the projection to 2D.
    #
    #   2. It needs none of your `student_tasks.py` code, so the validation
    #      loop keeps working (and keeps telling you something useful) even
    #      while you are still debugging the loss.
    #
    # The val split is never trained on, so this is an honest estimate of the
    # score your submission will be graded with.

    def on_validation_epoch_start(self) -> None:
        self._val_embeddings.clear()
        self._val_labels.clear()

    def validation_step(self, batch, batch_idx: int) -> None:
        images, labels = batch
        self._val_embeddings.append(self(images).detach().cpu())
        self._val_labels.append(labels.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_embeddings:
            return

        embeddings = torch.cat(self._val_embeddings).numpy()
        labels = torch.cat(self._val_labels).numpy()

        accuracy = knn_probe(
            embeddings,
            labels,
            n_neighbors=self.hparams.probe_neighbors,
            holdout=True,
        )

        self._val_embeddings.clear()
        self._val_labels.clear()

        # nan during Lightning's pre-training sanity check, when only a couple
        # of batches are seen. Logging nan would poison the checkpoint monitor.
        if math.isnan(accuracy):
            return

        self.log("val_knn_acc", accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        ==================================================================
          FREE TO MODIFY -- tune this however you see fit.

          Adam at 1e-3 is a reasonable starting point, nothing more. You
          are welcome to switch optimiser (AdamW, SGD+momentum), add
          weight decay, or attach an LR scheduler by returning the usual
          {"optimizer": ..., "lr_scheduler": ...} dict.

          The loss is a `.sum()` over EDGE_BATCH_SIZE * (1 + N_NEGATIVE)
          pairs, so its magnitude -- and hence the gradient scale -- moves
          with both. Raise either one and expect to lower the LR.

          Measured on the reference solution at the shipped settings, 1e-3
          was the best of {1e-3, 5e-3, 1e-2, 3e-2}: the larger values all
          reached a similar peak but oscillated instead of converging. Worth
          re-tuning if you change the architecture or the batch geometry.
        ==================================================================
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    L.seed_everything(SEED, workers=True)

    # `get_dataloaders` loads the global graph and turns it into an edge list.
    train_loader, val_loader = get_dataloaders()

    # Re-seed immediately before constructing the model. `seed_everything`
    # above already fixed the global RNG, but CNN weight init draws from it,
    # so the initial weights would otherwise depend on how much randomness the
    # data pipeline happened to consume first. Re-seeding here pins the
    # initial weights to SEED alone, independent of anything above.
    L.seed_everything(SEED, workers=True)
    model = ParametricUMAP()

    print(
        f"graph: {len(train_loader.dataset):,} undirected edges | "
        f"{len(train_loader.sampler):,} sampled per epoch | "
        f"{N_NEGATIVE} negatives per positive"
    )
    print(f"val: {len(val_loader.dataset)} images (held out, never trained on)")

    # Keeps the epoch with the BEST held-out topology, not merely the last one
    # or the lowest training loss. The UMAP loss can keep falling while the
    # embedding degenerates (e.g. collapsing into a ball), so `val_knn_acc` is
    # the more trustworthy selection signal.
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        # Deliberately NOT "model": that is FINAL_CHECKPOINT's name, and the
        # publish step below copies onto it. Same name here would mean the
        # callback's best epoch and the published file race for one path.
        filename="best",
        monitor="val_knn_acc",
        mode="max",
        save_top_k=1,
        save_weights_only=False,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
        deterministic=True,
        # Run the KNN probe after every epoch. Raise this if validation feels
        # slow — it embeds 6k images and fits a KNN each time.
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # Publish the BEST epoch at the path that gets graded. SUBMIT THIS FILE.
    #
    # The best epoch is the ONLY checkpoint kept. There is deliberately no
    # fallback to `trainer.save_checkpoint`: that writes the weights as they
    # are right now -- the LAST epoch. val_knn_acc oscillates from epoch to
    # epoch, so the last epoch is routinely worse than the best one, and
    # publishing it would quietly cost you marks you had already earned.
    #
    # If no best epoch exists, nothing is written. A missing checkpoint is a
    # loud, obvious failure; a checkpoint holding the wrong weights is a
    # silent one.
    best_path = checkpoint_cb.best_model_path
    best_score = checkpoint_cb.best_model_score

    if not best_path or not os.path.exists(best_path):
        raise SystemExit(
            "\n  No best checkpoint was recorded, so nothing was saved.\n\n"
            "  val_knn_acc was never logged, which means validation did not\n"
            "  run or returned nan on every epoch. Check that the validation\n"
            "  loader is non-empty and that training completed at least one\n"
            "  full epoch, then run again.\n"
        )

    shutil.copyfile(best_path, FINAL_CHECKPOINT)
    os.remove(best_path)
    print(
        f"\nSaved BEST checkpoint to: {FINAL_CHECKPOINT}\n"
        f"  val_knn_acc = {best_score.item():.4f}  "
        f"(epoch with the highest held-out score, not the last epoch)"
    )
    print("Submit this checkpoint together with your student_tasks.py.")


if __name__ == "__main__":
    main()
