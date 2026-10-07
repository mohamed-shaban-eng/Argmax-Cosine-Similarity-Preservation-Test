# ArgmaxLSH

**Does argmax preserve cosine similarity?**

ArgmaxLSH is a small experiment that tests whether encoding a vector as the
index of its largest component (`argmax`) keeps similar vectors together.
It works like a locality-sensitive hash (LSH): similar vectors should get the
same integer code, and dissimilar vectors should get different ones.

## Idea

1. Generate `n` vectors (random, or clustered for more natural similarity).
2. Normalize each vector to unit length.
3. Assign each vector an integer: `argmax` of its components.
4. For every pair of vectors, compute cosine similarity.
   If it is above `1/√2` (angle < 45°), the pair is **similar**.
5. Measure:
   - **P(same integer | similar pair)**: do similar vectors get the same code?
   - **P(different integer | dissimilar pair)**: do dissimilar vectors get different codes?

## Requirements

- Python 3.8+
- NumPy

```bash
pip install numpy
```

## Usage

```bash
# Pure random vectors
python argmax_lsh.py

# Clustered data (more similar pairs)
python argmax_lsh.py --clusters 5 --noise 0.3

# Custom settings
python argmax_lsh.py --n 200 --k 32 --threshold 0.8 --seed 42
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 100 | Number of vectors |
| `--d_in` | 64 | Input dimension |
| `--k` | 16 | Vector dimension (number of possible integers) |
| `--clusters` | 0 | Number of clusters (0 = pure random) |
| `--noise` | 0.5 | Noise inside clusters |
| `--threshold` | 0.7071 | Cosine threshold for "similar" |
| `--seed` | 0 | Random seed |

## Example output

```
Threshold (cosine): 0.7071
Total pairs: 4950
Similar pairs: ... | with same integer: ...
Dissimilar pairs: ... | with different integers: ...

Proportion of similar vectors with the same integer value assigned:       ...
Proportion of dissimilar vectors with different integer values assigned:  ...
```

## Notes

- With purely random high-dimensional vectors, very few pairs are "similar",
  so use `--clusters` for a more meaningful test.
- If there are no similar pairs, the first proportion prints `nan`
  (the original loop would crash with a division by zero).

## License

MIT