"""
ArgmaxLSH: does argmax encoding preserve cosine similarity?

Pipeline:
  1. Build vectors B (random or clustered data projected randomly).
  2. D = L2-normalized B,  D_num = argmax of each vector.
  3. For every pair (i, j): if cosine(D[i], D[j]) > threshold -> "similar".
  4. Report:
       - P(same argmax index | similar pair)
       - P(different argmax index | dissimilar pair)
"""

##  Representation Learning / Embeddings Evaluation

import argparse
import numpy as np


def make_vectors(n_samples=100, d_in=64, k=16, n_clusters=0, noise=0.5, seed=0):
    """Create B (n_samples x k) by randomly projecting input data.

    n_clusters = 0  -> pure random Gaussian data
    n_clusters > 0  -> clustered data (gives more naturally similar pairs)
    """
    rng = np.random.default_rng(seed)

    if n_clusters > 0:
        centers = rng.normal(size=(n_clusters, d_in))
        labels = rng.integers(0, n_clusters, size=n_samples)
        X = centers[labels] + noise * rng.normal(size=(n_samples, d_in))
    else:
        X = rng.normal(size=(n_samples, d_in))

    W = rng.normal(size=(d_in, k))
    return X @ W


def normalize(B, eps=1e-12):
    """Row-wise L2 normalization."""
    return B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)


def evaluate(D, D_num, threshold=1 / np.sqrt(2)):
    """Compute the two proportions over all unique pairs (i < j)."""
    n = len(D)
    cos_matrix = D @ D.T                       # vectors are already unit length
    i_idx, j_idx = np.triu_indices(n, k=1)     # unique pairs only
    cos = cos_matrix[i_idx, j_idx]
    same_index = D_num[i_idx] == D_num[j_idx]

    similar = cos > threshold
    dissimilar = ~similar

    n_similar = int(similar.sum())
    n_similar_same = int((similar & same_index).sum())
    n_notsimilar = int(dissimilar.sum())
    n_notsimilar_dif = int((dissimilar & ~same_index).sum())

    p_sim_same = n_similar_same / n_similar if n_similar else float("nan")
    p_dis_diff = n_notsimilar_dif / n_notsimilar if n_notsimilar else float("nan")

    return {
        "n_pairs": len(cos),
        "n_similar": n_similar,
        "n_similar_same": n_similar_same,
        "n_notsimilar": n_notsimilar,
        "n_notsimilar_dif": n_notsimilar_dif,
        "p_similar_same_index": p_sim_same,
        "p_dissimilar_diff_index": p_dis_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Argmax vs cosine similarity test")
    parser.add_argument("--n", type=int, default=100, help="number of vectors")
    parser.add_argument("--d_in", type=int, default=64, help="input dimension")
    parser.add_argument("--k", type=int, default=16, help="vector dimension (number of argmax classes)")
    parser.add_argument("--clusters", type=int, default=0, help="number of clusters (0 = pure random)")
    parser.add_argument("--noise", type=float, default=0.5, help="cluster noise level")
    parser.add_argument("--threshold", type=float, default=1 / np.sqrt(2),
                        help="cosine threshold (default 1/sqrt(2) = cos 45 deg)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    B = make_vectors(args.n, args.d_in, args.k, args.clusters, args.noise, args.seed)
    D = normalize(B)               # D = B_normalized
    D_num = np.argmax(B, axis=1)   # D_num = C_argmax (argmax is unchanged by normalization)

    r = evaluate(D, D_num, args.threshold)

    print(f"Threshold (cosine): {args.threshold:.4f}")
    print(f"Total pairs: {r['n_pairs']}")
    print(f"Similar pairs: {r['n_similar']} | with same integer: {r['n_similar_same']}")
    print(f"Dissimilar pairs: {r['n_notsimilar']} | with different integers: {r['n_notsimilar_dif']}")
    print()
    print("Proportion of similar vectors with the same integer value assigned:      ",
          r["p_similar_same_index"])
    print("Proportion of dissimilar vectors with different integer values assigned: ",
          r["p_dissimilar_diff_index"])


if __name__ == "__main__":
    main()