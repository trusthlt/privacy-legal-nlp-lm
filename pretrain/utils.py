from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent


def equal(x1, x2, diff=0.000001):
    return abs(x1 - x2) <= diff


def compute_epsilon(steps, noise_multiplier, sample_rate, delta):
    """Computes epsilon value for given hyperparameters."""
    if noise_multiplier == 0.0:
        return float('inf')
    # orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])
    rdp = compute_rdp(q=sample_rate,
                      noise_multiplier=noise_multiplier,
                      steps=steps,
                      orders=orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps, opt_order


def compute_target_nm(target_eps, steps, sample_rate, delta, nm_lower=0.1, nm_upper=2.1):
    # check upper -> epsilon larger, nm lower
    eps, alpha = compute_epsilon(steps, nm_lower, sample_rate, delta)
    if equal(eps, target_eps):
        return nm_lower
    if not eps > target_eps:
        raise ValueError(f"Bounds are wrong please make the lower bound {nm_lower} lower.")

    # check lower -> epsilon lower, nm higher
    eps, alpha = compute_epsilon(steps, nm_upper, sample_rate, delta)
    if equal(eps, target_eps):
        return nm_upper
    if not eps < target_eps:
        raise ValueError(f"Bounds are wrong please make the upper bound {nm_upper} larger.")

    while nm_lower < nm_upper:
        mid = (nm_upper + nm_lower) / 2
        eps, alpha = compute_epsilon(steps, mid, sample_rate, delta)
        if equal(eps, target_eps):
            return mid
        elif eps > target_eps:
            nm_lower = mid
        else:
            nm_upper = mid
