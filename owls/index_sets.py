import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def index_set(d, param, max_index = 100000000, weights=None, max_size=100000000, type="hc", p_norm=2):
    """
    Generate an index set in d dimensions with a given criterion:
      - Hyperbolic cross: Prod_{i=1}^d (1+n_i)^(1/weights[i]) <= param+1,
      - lp ball: ((Sum_{i=1}^d (1+n_i)^(p_norm/weights[i]))^(1/p_norm)) <= param.

    Parameters:
        d (int): Dimension of the index set.
        param (int): Parameter controlling the size of the index set:
                     for 'hc': corresponds to H_c (with threshold param+1),
                     for 'lp': corresponds to the lp ball radius (with threshold param).
        max_index (int): Maximum value allowed for any index coordinate.
        weights (array-like, optional): Anisotropic weights (each in (0, 1]). If None, defaults to ones.
        max_size (int): Maximum number of indices to generate.
        type (str): Either 'lp' for lp ball or 'hc' for hyperbolic cross.
        p_norm (float, optional): The norm exponent used for the lp ball; only used if type=='lp'. Default is 2.

    Returns:
        np.ndarray: Sorted array of index tuples in the generated index set.
    """
    if weights is None: 
        weights = np.ones(d)
    weights = np.array(weights, dtype=float)
    if weights.shape[0] != d:
        raise ValueError("Length of the weight array must match the dimension d")
    # Avoid zero weights (replace with a very small number)
    weights[weights == 0] = 1e-10
    if not np.all((weights > 0) & (weights <= 1)):
        raise ValueError("All weights must lie in (0,1]")

    index_set = set()
    visited = set()  # Track indices already queued to avoid duplicates
    queue = deque([(0,) * d])
    visited.add((0,) * d)
    
    if type == 'hc':
        threshold = param + 1  # Constant threshold for comparison
    elif type == 'lp':
        threshold = param

    while queue:
        current = queue.popleft()
        
        # Compute the criterion based on the chosen type.
        if type == "hc":
            prod = 1.0
            for i, n in enumerate(current):
                prod *= (1 + n) ** (1.0 / weights[i])
            criterion = prod
        elif type == "lp":
            s = 0.0
            for i, n in enumerate(current):
                s += (n) ** (p_norm / weights[i])
            criterion = s ** (1.0 / p_norm)
        else:
            raise ValueError("Unknown type specified: use 'lp' or 'hc'")

        # Prune if the current index fails the condition
        if criterion > threshold:
            continue

        # Add the current index (if below max_size)
        if len(index_set) < max_size:
            index_set.add(current)
        else:
            break

        # Enqueue neighbors (increment one coordinate at a time, bounded by max_index)
        for i in range(d):
            if current[i] < max_index:
                neighbor = list(current)
                neighbor[i] += 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return np.array(sorted(index_set))
if __name__ == "__main__":
    # ------------------ 2D Tests with Graphs ------------------
    # Parameters for 2D examples
    d = 2
    max_index = 40
    param_hc = 20
    param_lp = 30
    weights = [1, 1]
    p_norm = 0.3333

    # Generate index sets
    indices_hc = index_set(d, param_hc, max_index, weights, type="hc")
    indices_lp = index_set(d, param_lp, max_index, weights, type="lp", p_norm=p_norm)

    # Print some summary info
    print("2D Hyperbolic Cross: {} indices".format(len(indices_hc)))
    print("2D lp Ball: {} indices".format(len(indices_lp)))

    # Plotting the 2D index sets side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(indices_hc[:, 0], indices_hc[:, 1], s=10)
    plt.title(f"Hyperbolic Cross Index Set (2D)\nparam={param_hc}")
    plt.xlabel("Index 0")
    plt.ylabel("Index 1")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(indices_lp[:, 0], indices_lp[:, 1], s=10, color='red')
    plt.title(f"lp Ball Index Set (2D)\nparam={param_lp}, p_norm={p_norm}")
    plt.xlabel("Index 0")
    plt.ylabel("Index 1")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ------------------ 3D Tests with Graphs ------------------
    # Parameters for 3D examples
    d = 3
    max_index = 40
    param_hc_3d = 10
    param_lp_3d = 10
    weights = [1, 1, 1]

    # Generate index sets for 3D
    indices_hc_3d = index_set(d, param_hc_3d, max_index, weights, type="hc")
    indices_lp_3d = index_set(d, param_lp_3d, max_index, weights, type="lp", p_norm=p_norm)

    print("3D Hyperbolic Cross: {} indices".format(len(indices_hc_3d)))
    print("3D lp Ball: {} indices".format(len(indices_lp_3d)))

    # 3D plotting using matplotlib's mplot3d
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(indices_hc_3d[:, 0], indices_hc_3d[:, 1], indices_hc_3d[:, 2])
    ax1.set_title(f"Hyperbolic Cross Index Set (3D)\nparam={param_hc_3d}")
    ax1.set_xlabel("Index 0")
    ax1.set_ylabel("Index 1")
    ax1.set_zlabel("Index 2")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(indices_lp_3d[:, 0], indices_lp_3d[:, 1], indices_lp_3d[:, 2], color='red')
    ax2.set_title(f"lp Ball Index Set (3D)\nparam={param_lp_3d}, p_norm={p_norm}")
    ax2.set_xlabel("Index 0")
    ax2.set_ylabel("Index 1")
    ax2.set_zlabel("Index 2")

    plt.tight_layout()
    plt.show()
