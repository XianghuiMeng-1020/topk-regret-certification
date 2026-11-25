"""T7 categories are exhaustive and mutually exclusive."""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.certified_risk_frontier import (
    STAB,
    normalize_certificate,
)
from private_mobility_cert.certificates.coupling_diagnostics import diagnose_coupling
from private_mobility_cert.certificates.release_conditioned import deploy_ranking


def test_categories_partition():
    rng = np.random.default_rng(99)
    taus = [0.05, 0.1, 0.25, 0.5, 1.0]
    cats = {
        "A_structural_irrelevance",
        "B_easy_saturation",
        "B_hard_abstention",
        "C_threshold_crossing",
        "B_same_side_other",
    }
    for _ in range(60):
        n = int(rng.integers(5, 9))
        k = int(rng.integers(2, 5))
        y = rng.normal(size=n)
        w = 1.0 / np.log2(np.arange(2, k + 2))
        t = float(rng.uniform(0.2, 2.5))
        pi = deploy_ranking(y, k)
        S = max(float(np.dot(w, np.maximum(y[pi], 0.0))), STAB)
        d = diagnose_coupling(y, k=k, weights=w, half_width=t, S=S)
        ct_f = normalize_certificate(d.C_full, S)
        ct_s = normalize_certificate(d.C_strong, S)
        for tau in taus:
            c = d.threshold_category(tau, ct_f, ct_s)
            assert c in cats
            # mutual exclusivity of operational meanings
            if abs(d.G_strong) <= 1e-12:
                assert c == "A_structural_irrelevance"
            elif ct_f <= tau < ct_s:
                assert c == "C_threshold_crossing"
