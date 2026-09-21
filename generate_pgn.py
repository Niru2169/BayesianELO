
import random
import chess
import chess.pgn
import numpy as np

random.seed(42)
np.random.seed(42)

true_ratings = {
    "EngineA": 2900,
    "EngineB": 2820,
    "EngineC": 2760,
    "EngineD": 2700,
    "EngineE": 2650,
    "EngineF": 2600,
}
true_kappa = 0.9   
names = list(true_ratings.keys())
n = len(names)

def rho(r):
    return 10 ** (r / 400)

def outcome_probs(r_i, r_j, kappa):
    ri, rj = rho(r_i), rho(r_j)
    denom = ri + rj + kappa * np.sqrt(ri * rj)
    return ri/denom, rj/denom, kappa*np.sqrt(ri*rj)/denom  


GAMES_PER_PAIR = 20
games_data = [] 

pgn_games = []
game_counter = 0
for a in range(n):
    for b in range(a+1, n):
        for g in range(GAMES_PER_PAIR):
            white, black = (a, b) if g % 2 == 0 else (b, a)
            p_white, p_black, p_draw = outcome_probs(true_ratings[names[white]],
                                                      true_ratings[names[black]],
                                                      true_kappa)
            r = random.random()
            if r < p_white:
                result = "1-0"
            elif r < p_white + p_black:
                result = "0-1"
            else:
                result = "1/2-1/2"

            game_counter += 1
            game = chess.pgn.Game()
            game.headers["Event"] = "Synthetic Engine Round Robin"
            game.headers["Site"] = "Local"
            game.headers["Round"] = str(game_counter)
            game.headers["White"] = names[white]
            game.headers["Black"] = names[black]
            game.headers["Result"] = result

            board = game.board()
            node = game
            for _ in range(random.randint(6, 16)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = random.choice(legal)
                node = node.add_variation(move)
                board.push(move)
            pgn_games.append(game)
            games_data.append((white, black, result))

with open("engine_tournament.pgn", "w") as f:
    for g in pgn_games:
        print(g, file=f, end="\n\n")

print(f"Wrote {len(pgn_games)} games to engine_tournament.pgn")
print(f"Players: {names}")
print(f"True hidden ratings: {true_ratings}")
print(f"True kappa: {true_kappa}")
