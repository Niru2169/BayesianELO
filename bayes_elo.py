import numpy as np
from scipy.optimize import minimize


def fit_bayeselo(games, n_players, anchors=None, prior_rating=1500.0, prior_std=200.0, kappa_prior_std=1.0):
 
    anchors = anchors or {}
    scale = np.log(10) / 400
    prior_std_log = prior_std * scale
    free = [p for p in range(n_players) if p not in anchors]

    fixed_u = np.zeros(n_players)
    for p, rating in anchors.items():
        fixed_u[p] = (rating - prior_rating) * scale

    def neg_log_posterior(params):
        u = fixed_u.copy()
        u[free] = params[:len(free)]
        ln_kappa = params[-1]
        rho, kappa = np.exp(u), np.exp(ln_kappa)
        ll = 0.0
        for i, j, w_i, w_j, d in games:
            denom = rho[i] + rho[j] + kappa * np.sqrt(rho[i] * rho[j])
            ll += (w_i * np.log(rho[i] / denom) +
                   w_j * np.log(rho[j] / denom) +
                   d * np.log(kappa * np.sqrt(rho[i] * rho[j]) / denom))
        prior_ratings = -np.sum(u[free] ** 2) / (2 * prior_std_log ** 2)
        prior_kappa = -(ln_kappa ** 2) / (2 * kappa_prior_std ** 2)
        return -(ll + prior_ratings + prior_kappa)

    result = minimize(neg_log_posterior, np.zeros(len(free) + 1), method='BFGS')

    u = fixed_u.copy()
    u[free] = result.x[:len(free)]
    return prior_rating + u / scale, float(np.exp(result.x[-1]))
