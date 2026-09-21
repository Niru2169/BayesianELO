import chess.pgn
import numpy as np
from scipy.optimize import minimize

player_names = []
player_index = {}

def get_index(name):
    if name not in player_index:
        player_index[name] = len(player_names)
        player_names.append(name)
    return player_index[name]

pair_tally = {}

with open("engine_tournament.pgn") as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        white = get_index(game.headers["White"])
        black = get_index(game.headers["Black"])
        result = game.headers["Result"]

        i, j = min(white, black), max(white, black)
        if (i, j) not in pair_tally:
            pair_tally[(i, j)] = [0, 0, 0] 

        if result == "1-0":
            winner = white
        elif result == "0-1":
            winner = black
        else:
            winner = None

        if winner is None:
            pair_tally[(i, j)][2] += 1
        elif winner == i:
            pair_tally[(i, j)][0] += 1
        else:
            pair_tally[(i, j)][1] += 1

games = [(i, j, w_i, w_j, d) for (i, j), (w_i, w_j, d) in pair_tally.items()]
n_players = len(player_names)

print("Parsed players:", player_names)
print("Aggregated matchups:")
for (i, j, w_i, w_j, d) in games:
    print(f"  {player_names[i]} vs {player_names[j]}: {w_i}-{w_j}-{d} (W-L-D for {player_names[i]})")
print()

def fit_bayeselo(games, n_players, prior_rating=1500, prior_std=200, kappa_prior_std=1.0):
    prior_std_log = prior_std / 400 * np.log(10)

    def neg_log_posterior(params):
        u, ln_kappa = params[:n_players], params[n_players]
        rho, kappa = np.exp(u), np.exp(ln_kappa)
        ll = 0.0
        for i, j, w_i, w_j, d in games:
            denom = rho[i] + rho[j] + kappa * np.sqrt(rho[i] * rho[j])
            ll += (w_i * np.log(rho[i] / denom) +
                   w_j * np.log(rho[j] / denom) +
                   d   * np.log(kappa * np.sqrt(rho[i] * rho[j]) / denom))
        prior_ratings = -np.sum(u**2) / (2 * prior_std_log**2)
        prior_kappa   = -(ln_kappa**2) / (2 * kappa_prior_std**2)
        return -(ll + prior_ratings + prior_kappa)   

    x0 = np.zeros(n_players + 1)
    result = minimize(neg_log_posterior, x0, method='L-BFGS-B')

    u, ln_kappa = result.x[:n_players], result.x[n_players]
    ratings = prior_rating + 400 * u / np.log(10)
    kappa = np.exp(ln_kappa)
    return ratings, kappa

ratings, kappa = fit_bayeselo(games, n_players, prior_rating=2700, prior_std=150)

print("Fitted vs true ratings")
print(f"{'Engine':<10}{'Fitted':<10}{'True':<10}")

true_vals = [2900, 2820, 2760, 2700, 2650, 2600]
for name, r, t in zip(player_names, ratings, true_vals):
    print(f"{name:<10}{r:<10.1f}{t:<10.1f}")
