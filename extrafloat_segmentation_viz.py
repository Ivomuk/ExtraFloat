"""
extrafloat_segmentation_viz.py
================================
Visualization helpers for agent cluster analysis.

Matplotlib and seaborn are SOFT dependencies — they are imported inside each
function body so that the rest of the segmentation package remains importable
in headless / server environments where those libraries are not installed.

Market: Uganda (UG) — MTN Mobile Money agent segmentation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE_CATEGORICAL_10: str = "tab10"
_PALETTE_CATEGORICAL_20: str = "tab20"
_DEFAULT_HEATMAP_CMAP: str = "Blues"

# ─────────────────────────────────────────────────────────────────────────────
# SUBSAMPLE RANDOM SEED  (used in all point-cloud plots for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

_SUBSAMPLE_SEED: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _soft_import() -> tuple:
    """
    Import soft-dependency plotting libraries.

    Raises
    ------
    ImportError
        With an actionable install message if any library is missing.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Visualization helpers require matplotlib, seaborn, numpy, and pandas. "
            "Install them with: pip install matplotlib seaborn numpy pandas"
        ) from e
    return plt, sns, np, pd


def _subsample_indices(
    n_total: int,
    max_points: int,
    rng: "np.random.RandomState",
) -> "np.ndarray":
    """Return an index array of length min(n_total, max_points), shuffled."""
    import numpy as np  # noqa: F401 — already guarded by _soft_import callers

    if n_total <= max_points:
        return rng.permutation(n_total)
    return rng.choice(n_total, size=max_points, replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_pca_scatter(
    X_2d: "np.ndarray",
    labels: "pd.Series | np.ndarray",
    title: str = "PCA Scatter",
    max_points: int = 50_000,
    point_size: int = 5,
    ax: object | None = None,
) -> object:
    """
    2D scatter of PCA-reduced data, coloured by cluster labels.

    Subsamples to *max_points* if needed using a reproducible seed (42).
    Unique label values are mapped to a ``tab10`` / ``tab20`` categorical
    palette.

    Parameters
    ----------
    X_2d : array of shape (n_samples, 2) — the first two PCA dimensions.
    labels : integer or string cluster labels, length n_samples.
    title : figure title string.
    max_points : subsample cap; set to 0 to disable subsampling.
    point_size : matplotlib ``s`` scatter marker size.
    ax : optional existing Axes to draw on.  If None a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, sns, np, pd = _soft_import()

    rng = np.random.RandomState(_SUBSAMPLE_SEED)
    labels_arr = np.asarray(labels)
    n = len(labels_arr)

    if max_points and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X_plot = X_2d[idx]
        labels_plot = labels_arr[idx]
        logger.debug(
            "plot_pca_scatter: subsampled %d → %d points", n, max_points
        )
    else:
        X_plot = X_2d
        labels_plot = labels_arr

    unique_labels = sorted(set(labels_plot))
    palette = _PALETTE_CATEGORICAL_10 if len(unique_labels) <= 10 else _PALETTE_CATEGORICAL_20
    colour_list = sns.color_palette(palette, n_colors=len(unique_labels))
    colour_map = {lbl: colour_list[i] for i, lbl in enumerate(unique_labels)}
    point_colours = [colour_map[lbl] for lbl in labels_plot]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.get_figure()

    ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=point_colours,
        s=point_size,
        alpha=0.6,
        linewidths=0,
    )

    # Legend patches
    import matplotlib.patches as mpatches  # noqa: PLC0415
    patches = [
        mpatches.Patch(color=colour_map[lbl], label=str(lbl))
        for lbl in unique_labels
    ]
    ax.legend(handles=patches, title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)

    plt.tight_layout()
    logger.info("plot_pca_scatter: rendered %d points, %d clusters", len(labels_plot), len(unique_labels))
    return fig


def plot_cluster_distribution(
    df: "pd.DataFrame",
    cluster_col: str,
    title: str = "Agent Counts by Cluster",
    top_n: int | None = None,
) -> object:
    """
    Bar chart showing agent counts per cluster label.

    Parameters
    ----------
    df : DataFrame containing the cluster column.
    cluster_col : name of the column holding cluster labels.
    title : figure title string.
    top_n : if provided, display only the top_n most-populated clusters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, sns, np, pd = _soft_import()

    if cluster_col not in df.columns:
        raise ValueError(
            f"plot_cluster_distribution: column '{cluster_col}' not found in df. "
            f"Available columns: {list(df.columns)}"
        )

    counts: pd.Series = df[cluster_col].value_counts().sort_values(ascending=False)

    if top_n is not None:
        counts = counts.head(top_n)

    n_clusters = len(counts)
    palette = _PALETTE_CATEGORICAL_10 if n_clusters <= 10 else _PALETTE_CATEGORICAL_20

    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 0.9), 5))
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        palette=palette,
        ax=ax,
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Agent Count")
    ax.set_title(title)

    # Annotate bar tops
    for bar, val in zip(ax.patches, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + counts.max() * 0.01,
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    logger.info(
        "plot_cluster_distribution: %d clusters plotted from column '%s'",
        n_clusters,
        cluster_col,
    )
    return fig


def plot_purity_heatmap(
    crosstab_df: "pd.DataFrame",
    title: str = "Cluster × Category (row-normalised)",
    cmap: str = "Blues",
    annotate: bool = False,
) -> object:
    """
    Seaborn heatmap of a row-normalised crosstab matrix.

    Intended for inspecting cluster purity against a known categorical label
    (e.g. ``agent_category``).  The input crosstab is row-normalised to
    proportions so each row sums to 1.0.

    Parameters
    ----------
    crosstab_df : raw count crosstab; index = cluster labels, columns = categories.
    title : figure title string.
    cmap : seaborn/matplotlib colourmap name.
    annotate : if True, overlay cell values as text annotations.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, sns, np, pd = _soft_import()

    row_sums = crosstab_df.sum(axis=1).replace(0, np.nan)
    normed = crosstab_df.div(row_sums, axis=0).fillna(0.0)

    nrows, ncols = normed.shape
    fig, ax = plt.subplots(figsize=(max(6, ncols * 1.1), max(4, nrows * 0.7)))

    sns.heatmap(
        normed,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Cluster")

    plt.tight_layout()
    logger.info(
        "plot_purity_heatmap: %d clusters × %d categories", nrows, ncols
    )
    return fig


def plot_gmm_hdbscan_comparison(
    X_2d: "np.ndarray",
    gmm_labels: "pd.Series | np.ndarray",
    hdb_labels: "pd.Series | np.ndarray",
    max_points: int = 100_000,
) -> object:
    """
    Side-by-side (1 × 2) PCA scatter comparison of GMM vs HDBSCAN labels.

    Useful for validating that the two complementary clustering approaches
    agree on high-level structure.  Both panels are subsampled to the same
    random indices so the point positions match exactly.

    Parameters
    ----------
    X_2d : PCA-reduced array of shape (n_samples, 2).
    gmm_labels : cluster labels from the GMM step.
    hdb_labels : tier / cluster labels from the HDBSCAN step.
    max_points : per-panel subsample cap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, sns, np, pd = _soft_import()

    rng = np.random.RandomState(_SUBSAMPLE_SEED)
    gmm_arr = np.asarray(gmm_labels)
    hdb_arr = np.asarray(hdb_labels)
    n = len(gmm_arr)

    if max_points and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X_plot = X_2d[idx]
        gmm_plot = gmm_arr[idx]
        hdb_plot = hdb_arr[idx]
        logger.debug(
            "plot_gmm_hdbscan_comparison: subsampled %d → %d points", n, max_points
        )
    else:
        X_plot = X_2d
        gmm_plot = gmm_arr
        hdb_plot = hdb_arr

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, labels_plot, panel_title in [
        (axes[0], gmm_plot, "GMM Clusters"),
        (axes[1], hdb_plot, "HDBSCAN Tiers"),
    ]:
        unique_labels = sorted(set(labels_plot))
        palette = _PALETTE_CATEGORICAL_10 if len(unique_labels) <= 10 else _PALETTE_CATEGORICAL_20
        colour_list = sns.color_palette(palette, n_colors=len(unique_labels))
        colour_map = {lbl: colour_list[i] for i, lbl in enumerate(unique_labels)}
        point_colours = [colour_map[lbl] for lbl in labels_plot]

        ax.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=point_colours,
            s=4,
            alpha=0.5,
            linewidths=0,
        )

        import matplotlib.patches as mpatches  # noqa: PLC0415
        patches = [
            mpatches.Patch(color=colour_map[lbl], label=str(lbl))
            for lbl in unique_labels
        ]
        ax.legend(handles=patches, title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(panel_title)

    plt.tight_layout()
    logger.info(
        "plot_gmm_hdbscan_comparison: %d GMM clusters, %d HDBSCAN tiers",
        len(set(gmm_plot)),
        len(set(hdb_plot)),
    )
    return fig


def plot_segment_profiles(
    profile_df: "pd.DataFrame",
    title: str = "Segment Profiles (z-scored)",
    figsize: tuple[int, int] = (14, 6),
) -> object:
    """
    Heatmap of z-scored segment KPI profiles.

    Each row of *profile_df* represents one named segment; each column is a
    KPI feature.  Values are z-scored across segments before plotting so that
    features with very different scales are visually comparable.

    Parameters
    ----------
    profile_df : DataFrame with index = segment names, columns = KPI features.
    title : figure title string.
    figsize : (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt, sns, np, pd = _soft_import()

    # Z-score each column across segments
    std = profile_df.std(axis=0).replace(0, np.nan)
    z_scored = (profile_df - profile_df.mean(axis=0)) / std
    z_scored = z_scored.fillna(0.0)

    nrows, ncols = z_scored.shape
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        z_scored,
        cmap="RdYlGn",
        center=0.0,
        linewidths=0.4,
        linecolor="white",
        annot=(nrows * ncols) <= 200,   # annotate only for small tables
        fmt=".2f",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("KPI Feature")
    ax.set_ylabel("Segment")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    logger.info(
        "plot_segment_profiles: %d segments × %d KPIs", nrows, ncols
    )
    return fig
