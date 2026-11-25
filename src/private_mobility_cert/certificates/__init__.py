"""Ex-post Top-K regret certificates."""

from private_mobility_cert.certificates.capacity_contrast import (
    capacity_regret,
    capacity_transfer_bound_l2,
)
from private_mobility_cert.certificates.quantiles import (
    analytic_q_m_geometric,
    analytic_q_m_laplace,
    calibrated_q_m_table,
)
from private_mobility_cert.certificates.subset_swap import (
    certificate_c0_box,
    certificate_c1_pairwise,
    certificate_c2_subset_swap,
    deployable_selection_from_y_act,
    observed_multi_swap_deficit,
    oracle_certificate,
    q_m_order_statistic,
)
from private_mobility_cert.certificates.topk_regret import (
    subset_swap_regret_identity,
    topk_regret,
    topk_select_indices,
)

__all__ = [
    "analytic_q_m_geometric",
    "analytic_q_m_laplace",
    "calibrated_q_m_table",
    "capacity_regret",
    "capacity_transfer_bound_l2",
    "certificate_c0_box",
    "certificate_c1_pairwise",
    "certificate_c2_subset_swap",
    "deployable_selection_from_y_act",
    "observed_multi_swap_deficit",
    "oracle_certificate",
    "q_m_order_statistic",
    "subset_swap_regret_identity",
    "topk_regret",
    "topk_select_indices",
]
