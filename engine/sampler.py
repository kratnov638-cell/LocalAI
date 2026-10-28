"""Sampling utilities for the NumPy inference engine."""
import numpy as np


def sample(logits, history=(), temperature=0.8, top_p=0.95, repetition_penalty=1.1, rng=None, top_k=0, penalty_exempt=()):
    # Float32 is enough for sampling model logits and avoids copying a full
    # 128k-vocabulary vector to float64 on every generated token.
    x = np.asarray(logits, dtype=np.float32).copy()
    if x.ndim != 1 or x.size == 0:
        raise ValueError('logits must be a non-empty one-dimensional array')
    if not np.isfinite(x).all():
        raise ValueError('model produced non-finite logits; refusing to sample corrupted output')
    if not np.isfinite(temperature):
        raise ValueError('temperature must be finite')
    if not 0.0 < top_p <= 1.0:
        raise ValueError('top_p must be in (0, 1]')
    if not np.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError('repetition_penalty must be finite and greater than zero')
    if isinstance(top_k, bool) or int(top_k) != top_k or top_k < 0:
        raise ValueError('top_k must be a non-negative integer')

    # Structural stop tokens often already occur in the prompt (for example,
    # Llama 3's <|eot_id|> closes the user's message). Penalizing that token
    # because it appeared in the prompt suppresses the model's own end-of-turn
    # signal and can make a short answer spill into hundreds of junk tokens.
    exempt = set(penalty_exempt)
    for token in set(history):
        if token in exempt:
            continue
        if 0 <= token < len(x):
            x[token] = x[token] * repetition_penalty if x[token] < 0 else x[token] / repetition_penalty
    if temperature <= 0:
        return int(np.argmax(x))

    x /= temperature
    if top_k and top_k < x.size:
        keep = np.argpartition(x, -int(top_k))[-int(top_k):]
        mask = np.ones(x.size, dtype=bool)
        mask[keep] = False
        x[mask] = -np.inf
    x -= np.max(x)
    weights = np.exp(x)
    total_weight = float(weights.sum())
    if top_p < 1.0:
        # Find the nucleus with adaptive partial sorting instead of sorting all
        # 128k vocabulary entries at every step. The retained set is exact for
        # the same top-p cutoff; k grows only if the current candidates do not
        # contain enough probability mass.
        k = min(256, x.size)
        while True:
            candidates = np.argpartition(x, -k)[-k:]
            candidates = candidates[np.argsort(x[candidates])[::-1]]
            candidate_weights = weights[candidates]
            cumulative = np.cumsum(candidate_weights, dtype=np.float64) / total_weight
            if cumulative[-1] >= top_p or k == x.size:
                keep = int(np.searchsorted(cumulative, top_p, side='left')) + 1
                candidates = candidates[:keep]
                candidate_weights = candidate_weights[:keep]
                probs = candidate_weights / candidate_weights.sum()
                rng = rng if rng is not None else np.random.default_rng()
                return int(candidates[rng.choice(len(candidates), p=probs)])
            k = min(k * 2, x.size)
    probabilities = weights / total_weight
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(probabilities), p=probabilities))
