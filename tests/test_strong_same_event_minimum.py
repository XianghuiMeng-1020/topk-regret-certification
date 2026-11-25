"""C_strong = min of valid nonexact same-event certificates remains valid."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    certificate_full_coupled_box,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.certified_risk_frontier import (
    normalize_certificate,
)


def test_strong_min_components_and_frontier():
    rng = np.random.default_rng(20260807)
    taus = [0.05, 0.1, 0.25, 0.5, 1.0]
    for _ in range(40):
        n = int(rng.integers(5, 9))
        k = int(rng.integers(2, min(5, n)))
        y = rng.normal(size=n)
        w = 1.0 / np.log2(np.arange(2, k + 2))
        t = float(rng.uniform(0.2, 2.5))
        full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
        strong, parts = strongest_nonexact_same_event(
            y, k=k, weights=w, half_width=t
        )
        assert abs(strong - min(parts.values())) <= 1e-12
        assert all(v + 1e-9 >= full for v in parts.values())
        assert strong + 1e-9 >= full
        S = max(float(np.sort(np.maximum(y, 0.0))[::-1][:k].sum()), 1e-6)
        ct_f = normalize_certificate(full, S)
        ct_s = normalize_certificate(strong, S)
        for tau in taus:
            # frontier inclusion: D_full contains D_strong? Wait: full <= strong so
            # {ct_full <= tau}\supseteq {ct_strong <= tau}
            in_strong = ct_s <= tau
            in_full = ct_f <= tau
            if in_strong:
                assert in_full
