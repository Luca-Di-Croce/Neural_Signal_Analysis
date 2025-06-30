import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx

from scipy.stats import mode
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity


def build_wout_graph(Wout, threshold=0.7):
    """
    Build a graph from Wout where nodes are reservoir units and
    edges represent cosine similarity above a threshold.
    """
    similarities = cosine_similarity(Wout)
    np.fill_diagonal(similarities, 0)

    N = Wout.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))  # Add all nodes first

    for i in range(N):
        for j in range(i + 1, N):
            if similarities[i, j] >= threshold:
                G.add_edge(i, j, weight=similarities[i, j])
    
    return G


def compute_graph_distance(G1, G2, method='spectral'):
    """
    Compare two graphs using the specified method.
    Supported: 'spectral' (Laplacian spectral distance), 'edge_overlap'
    """
    if method == 'spectral':
        L1 = nx.normalized_laplacian_matrix(G1).todense()
        L2 = nx.normalized_laplacian_matrix(G2).todense()
        eigs1 = np.sort(np.linalg.eigvals(L1))
        eigs2 = np.sort(np.linalg.eigvals(L2))
        return np.linalg.norm(eigs1 - eigs2)
    
    elif method == 'edge_overlap':
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        return 1 - (intersection / union)  # 0 = same, 1 = totally disjoint

    else:
        raise ValueError("Unknown method")

def compare_wout_sets(wout_dict, threshold=0.7, method='spectral'):
    """
    Given a dict of Wout matrices (e.g., {'grasp': W1, 'rest': W2}),
    build graphs and compute pairwise distance matrix.
    """
    keys = list(wout_dict.keys())
    graphs = {k: build_wout_graph(wout_dict[k], threshold) for k in keys}
    n = len(keys)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = compute_graph_distance(graphs[keys[i]], graphs[keys[j]], method)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return keys, dist_matrix

def compute_statistics_per_timepoint(Wout_dict):
    """
    Computes descriptive statistics for each timepoint across all nodes per class.

    Parameters:
    - Wout_dict: dict of {class_label: np.ndarray of shape (nodes, timepoints)}

    Returns:
    - stats_dict: dict of {class_label: dict of statistics}
    """
    stats_dict = {}

    for label, Wout in Wout_dict.items():
        # shape: (nodes, timepoints) = (500, 508)
        stats = {
            'mean': np.mean(Wout, axis=0),
            'std': np.std(Wout, axis=0),
            'min': np.min(Wout, axis=0),
            'max': np.max(Wout, axis=0),
            'median': np.median(Wout, axis=0),
            'q25': np.percentile(Wout, 25, axis=0),
            'q75': np.percentile(Wout, 75, axis=0),
            'mode': mode(Wout, axis=0, keepdims=False).mode  # returns (timepoints,)
        }
        stats_dict[label] = stats

    return stats_dict

def plot_statistics(stats_dict, stat_name='mean'):
    """
    Plot one statistic (e.g., 'mean') across timepoints for all classes.

    Parameters:
    - stats_dict: dict from compute_statistics_per_timepoint
    - stat_name: 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'mode'
    """
    plt.figure(figsize=(14, 5))
    for label, stats in stats_dict.items():
        values = stats[stat_name]
        plt.plot(values, label=f'Class {label}')

    plt.title(f'{stat_name.upper()} of W_out Across Timepoints (per Class)')
    plt.xlabel('Timepoint')
    plt.ylabel(stat_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    

def create_reservoir_graph(W):
    """
    Create a directed graph representation of the reservoir weight matrix W.
    Nodes are labeled 'R{i}' for reservoir units.
    Edges are added where weights are non-zero.
    
    Args:
        W (np.ndarray): Reservoir weight matrix (NxN)
    
    Returns:
        G (nx.DiGraph): Directed graph with nodes and weighted edges
        edge_weights (list): Scaled absolute weights for edge thickness in plots
    """
    reservoir_size = W.shape[0]
    G = nx.DiGraph()
    nodes = [f'R{i}' for i in range(reservoir_size)]
    G.add_nodes_from(nodes)

    edge_weights = []
    for i in range(reservoir_size):
        for j in range(reservoir_size):
            weight = W[i, j]
            if weight != 0:
                # Note edge direction from Rj -> Ri per W[i,j]
                G.add_edge(f'R{j}', f'R{i}', weight=weight)
                # Store scaled absolute weight for edge width plotting
                edge_weights.append(abs(weight) * 0.3)
    return G, edge_weights


def draw_reservoir_graph(G, edge_weights, Wout, pos, ax,
                          cmap_name='coolwarm', title=None, show_colorbar=False):
    """
    Plot the reservoir graph on a given matplotlib axis.
    Nodes colored by mean absolute Wout values; edges are black with widths scaled by edge_weights.

    Args:
        G (nx.DiGraph): Reservoir graph
        edge_weights (list): Edge widths scaled for plotting
        Wout (np.ndarray): Output weights matrix for coloring nodes
        pos (dict): Node positions for plotting (from networkx layout)
        ax (matplotlib.axes.Axes): Axis to draw on
        cmap_name (str): Colormap name for node colors
        title (str|None): Optional plot title
        show_colorbar (bool): Whether to show colorbar for node colors
    """
    # Compute node colors as mean absolute value of Wout per node
    node_colors = np.mean(np.abs(Wout), axis=1)
    norm = Normalize(vmin=np.min(node_colors), vmax=np.max(node_colors))
    cmap = cm.get_cmap(cmap_name)
    node_color_mapped = cmap(norm(node_colors))

    # Draw nodes and edges with styles
    nx.draw_networkx_nodes(G, pos, node_color=node_color_mapped,
                           node_size=80, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black',
                           width=edge_weights, alpha=0.5, ax=ax)

    if title:
        ax.set_title(title, fontsize=14)
    ax.axis('off')

    # Add colorbar for node colors if requested
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean |Wout| per node')

def plot_comparison_subplots(Wout_dict, W, class_groups, ncols,
                              titles=None, save_dir='plots', cmap_name='coolwarm',
                              show=True, filename_suffix=''):
    """
    Helper function to plot reservoir graphs for multiple classes side-by-side.
    
    Args:
        Wout_dict (dict): Dictionary mapping class labels to Wout arrays
        W (np.ndarray): Reservoir weight matrix
        class_groups (list of tuples): Each tuple contains class labels to plot together
        ncols (int): Number of subplots per figure (length of each tuple in class_groups)
        titles (list|None): Titles for each subplot, if None uses class labels as titles
        save_dir (str): Directory to save plots
        cmap_name (str): Colormap for nodes
        show (bool): Whether to display plots interactively
        filename_suffix (str): Suffix for saved filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    G, edge_weights = create_reservoir_graph(W)
    pos = nx.spring_layout(G, seed=42, k=0.3, weight=None)

    for classes in class_groups:
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 9))
        if ncols == 1:
            axes = [axes]  # ensure axes is iterable
        for idx, class_label in enumerate(classes):
           draw_reservoir_graph(
                G, edge_weights, Wout_dict[class_label], pos, axes[idx],
                cmap_name=cmap_name,
                title=titles[idx] if titles else str(class_label),
                show_colorbar=(idx == ncols - 1)  # Show colorbar only on last subplot
            )
        plt.tight_layout()
        filename = os.path.join(save_dir,
                                f"{'_vs_'.join(map(str, classes))}{filename_suffix}.png")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


def plot_reservoir_graph_comparisons(Wout_dict, W, class_pairs, **kwargs):
    """
    Plot reservoir graph comparisons for pairs of classes.
    """
    plot_comparison_subplots(Wout_dict, W, class_pairs, ncols=2, **kwargs)


def plot_reservoir_graph_comparisons_3way(Wout_dict, W, class_triplets, **kwargs):
    """
    Plot reservoir graph comparisons for triplets of classes.
    """
    plot_comparison_subplots(Wout_dict, W, class_triplets, ncols=3, **kwargs)


def bin_Wout_heatmap(Wout, num_rows=50, num_cols=10):
    """
    Bin Wout array into a heatmap by averaging over fixed row and column bins.

    Args:
        Wout (np.ndarray): 2D array of output weights (nodes x timepoints)
        num_rows (int): Number of row bins
        num_cols (int): Number of column bins
    
    Returns:
        heatmap (np.ndarray): Binned 2D array of shape (num_rows, num_cols)
    """
    row_bins = np.array_split(Wout, num_rows, axis=0)
    heatmap = np.zeros((num_rows, num_cols))
    for i, row_block in enumerate(row_bins):
        col_blocks = np.array_split(row_block, num_cols, axis=1)
        for j, block in enumerate(col_blocks):
            heatmap[i, j] = np.mean(block)
    return heatmap


def plot_reservoir_heatmap_comparisons_binned(Wout_dict, class_triplets,
                                              cmap_name='coolwarm', save_dir='plots', show=True):
    """
    Plot side-by-side heatmaps of binned absolute Wout arrays for three classes.
    The color scale is consistent across all three heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)

    for class1, class2, class3 in class_triplets:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # Compute binned heatmaps and track global min/max for consistent color scaling
        binned_data_dict = {}
        vmin, vmax = float('inf'), -float('inf')
        for label in [class1, class2, class3]:
            heatmap =bin_Wout_heatmap(np.abs(Wout_dict[label]))
            binned_data_dict[label] = heatmap
            vmin = min(vmin, heatmap.min())
            vmax = max(vmax, heatmap.max())

        # Plot each heatmap
        for idx, label in enumerate([class1, class2, class3]):
            ax = axes[idx]
            im = ax.imshow(binned_data_dict[label], cmap=cmap_name,
                           aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f"{label}", fontsize=14)
            ax.set_xlabel("Time bin")
            if idx == 0:
                ax.set_ylabel("Node bin")
            else:
                ax.set_yticks([])

        # Add a single colorbar to the right of the last subplot
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Mean |Wout| in bin')

        plt.tight_layout()
        filename = os.path.join(save_dir,
                                f"{class1}_vs_{class2}_vs_{class3}_heatmap_binned.png")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

def plot_class_difference_heatmaps(Wout_5_dict, Wout_10_dict, save_dir='plots', show=True):
    """
    Plot difference heatmaps between class 5 and classes 1-5 & 6-10.
    Shows side-by-side heatmaps with color scale centered at zero.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(1, 6):
        label_5 = f"class_{i}"
        label_10_1 = f"class_{i}"
        label_10_2 = f"class_{i+5}"

        W5 = Wout_5_dict[label_5]
        W10_1 = Wout_10_dict[label_10_1]
        W10_2 = Wout_10_dict[label_10_2]

        # Compute difference heatmaps
        diff_1 = W5 - W10_1
        diff_2 = W5 - W10_2

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        diffs = [diff_1, diff_2]
        titles = [f"{label_5} - {label_10_1}", f"{label_5} - {label_10_2}"]

        # Normalize color scale around zero for difference visualization
        vmax = max(np.abs(diff_1).max(), np.abs(diff_2).max())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        # Plot heatmaps
        for ax, diff, title in zip(axes, diffs, titles):
            im = ax.imshow(diff, aspect='auto', cmap='bwr', norm=norm)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Timepoints")
            ax.set_ylabel("Nodes")

        # Create a new axis on the right for the colorbar spanning both heatmaps
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Difference (Class 5 - Class 10)")

        plt.tight_layout()
        filename = os.path.join(save_dir, f"class_5_vs_class_10_comparison_{i}.png")
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

def plot_distance_graph(keys, dist_matrix, threshold=1.5, layout="kamada_kawai", cmap_name="viridis"):
    """
    Create and plot a weighted undirected graph where edges represent distances below a threshold.
    Edge thickness and color indicate similarity (inverse of distance).
    
    Args:
        keys (list): Node labels
        dist_matrix (np.ndarray): Symmetric distance matrix (NxN)
        threshold (float): Max distance to include edge
        layout (str): Layout algorithm name ("circular_layout", "kamada_kawai", "spring_layout")
        cmap_name (str): Colormap for edge colors
    """
    G = nx.Graph()
    G.add_nodes_from(keys)

    # Add edges for distances <= threshold
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = dist_matrix[i, j]
            if dist <= threshold:
                G.add_edge(keys[i], keys[j], weight=dist)

    # Determine node positions based on layout choice
    if layout == "circular_layout":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring_layout":
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)  # default fallback

    # Extract edge weights and calculate inverse weights for thickness
    edges = list(G.edges(data=True))
    weights = [attr["weight"] for _, _, attr in edges]
    # Normalize weights for thickness and color
    max_dist = max(weights) if weights else 1
    inv_weights = [max_dist - w for w in weights]

    # Draw graph components
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Map edge colors to inverse distances
    norm = Normalize(vmin=min(inv_weights), vmax=max(inv_weights))
    cmap = cm.get_cmap(cmap_name)
    edge_colors = [cmap(norm(w)) for w in inv_weights]

    nx.draw_networkx_edges(G, pos, width=[w * 2 for w in inv_weights], edge_color=edge_colors, alpha=0.8)

    plt.title("Distance Graph (Edges with dist <= {:.2f})".format(threshold))
    plt.axis('off')
    plt.show()


def plot_heatmap(matrix, all_classes, save_dir, title, filename):
    plt.figure(figsize=(13, 11))

    # Mask 0.0 values (used for diagonals)
    mask = np.isclose(matrix, 0.0)
    nonzero_vals = matrix[~mask]
    vmin = np.min(nonzero_vals)
    vmax = np.max(nonzero_vals)

    # Exclude diagonal from row sums
    matrix_no_diag = matrix.copy()
    np.fill_diagonal(matrix_no_diag, 0.0)
    row_sums = np.sum(matrix_no_diag, axis=1)
    col_sums = np.sum(matrix_no_diag, axis=0)

    # Class labels with distance sums
    row_labels = [f"{label} ({row_sums[i]:.2f})" for i, label in enumerate(all_classes)]
    col_labels = [f"{label}\n({col_sums[i]:.2f})" for i, label in enumerate(all_classes)]

    ax = sns.heatmap(
        matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",  # Reversed colormap
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()

def plot_class_similarity_heatmaps(Wout_5, Wout_10, save_dir='heatmaps'):
    os.makedirs(save_dir, exist_ok=True)

    all_classes = [f"5_class_{i}" for i in range(1, 6)] + [f"10_class_{i}" for i in range(1, 11)]

    # Compute mean vectors for each class
    mean_vectors = {}
    for i in range(1, 6):
        mean_vectors[f"5_class_{i}"] = Wout_5[f"class_{i}"].mean(axis=1)
    for i in range(1, 11):
        mean_vectors[f"10_class_{i}"] = Wout_10[f"class_{i}"].mean(axis=1)

    n = len(all_classes)
    cosine_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec_i = mean_vectors[all_classes[i]]
            vec_j = mean_vectors[all_classes[j]]
            cosine_matrix[i, j] = cosine(vec_i, vec_j)
            euclidean_matrix[i, j] = euclidean(vec_i, vec_j)

    plot_heatmap(cosine_matrix, all_classes, save_dir, "Cosine Distance Between Class Means", "cosine_distance_heatmap.png")
    plot_heatmap(euclidean_matrix, all_classes, save_dir, "Euclidean Distance Between Class Means", "euclidean_distance_heatmap.png")


def compute_similarity_matrices(Wout_dict):
    """
    Given a dictionary of vectors (either time-compressed or reservoir-compressed),
    compute cosine and Euclidean distance matrices (15x15).
    """
    labels = list(Wout_dict.keys())
    n = len(labels)

    cosine_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec_i = Wout_dict[labels[i]]
            vec_j = Wout_dict[labels[j]]
            cosine_matrix[i, j] = cosine(vec_i, vec_j)
            euclidean_matrix[i, j] = euclidean(vec_i, vec_j)

    return cosine_matrix, euclidean_matrix, labels

def plot_class_similarity_heatmaps_compressed(W_Out_5, W_Out_10):
    # Create a merged Wout dict for time-compressed or reservoir-compressed

    Wout_time = {f"5_class_{i}": W_Out_5[f"class_{i}"].mean(axis=1) for i in range(1, 6)}
    Wout_time.update({f"10_class_{i}": W_Out_10[f"class_{i}"].mean(axis=1) for i in range(1, 11)})

    Wout_reservoir = {f"5_class_{i}": W_Out_5[f"class_{i}"].mean(axis=0) for i in range(1, 6)}
    Wout_reservoir.update({f"10_class_{i}": W_Out_10[f"class_{i}"].mean(axis=0) for i in range(1, 11)})

    # Compute and plot TIME-COMPRESSED
    cos_time, eucl_time, labels_time = compute_similarity_matrices(Wout_time)
    plot_heatmap(cos_time, labels_time, "heatmaps", "Cosine Distance (Time Compressed)", "cosine_time.png")
    plot_heatmap(eucl_time, labels_time, "heatmaps", "Euclidean Distance (Time Compressed)", "euclidean_time.png")

    # Compute and plot RESERVOIR-COMPRESSED
    cos_res, eucl_res, labels_res = compute_similarity_matrices(Wout_reservoir)
    plot_heatmap(cos_res, labels_res, "heatmaps", "Cosine Distance (Reservoir Compressed)", "cosine_reservoir.png")
    plot_heatmap(eucl_res, labels_res, "heatmaps", "Euclidean Distance (Reservoir Compressed)", "euclidean_reservoir.png")
