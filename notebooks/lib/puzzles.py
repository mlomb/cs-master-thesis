from concurrent.futures import ThreadPoolExecutor
import threading
import chess
import chess.engine
from tqdm import tqdm
import random

engines: dict[chess.engine.SimpleEngine] = {}

def init_engine(engine_cmd: str | list[str]):
    engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)
    engines[threading.current_thread()] = engine


def solve_puzzle(board: chess.Board, solution: list[chess.Move], themes) -> tuple[int, int]:
    board = board.copy()
    solution = solution.copy() # clone
    engine = engines[threading.current_thread()]

    correct_moves = 0
    total_moves = 0

    while len(solution) > 0:
        opp_move, *solution = solution
        expected_move, *solution = solution

        # play opponent move
        board.push(opp_move)

        res = engine.play(board, chess.engine.Limit(time=50/1000))

        # play engine move
        board.push(expected_move)

        # Mate in 1 puzzles can have more than one solution (Lichess)
        if res.move == expected_move or board.is_checkmate():
            correct_moves += 1
        total_moves += 1

    return (correct_moves, total_moves)


class PuzzleAccuracy:
    def __init__(self, puzzles_csv: str, elo_bucket_size=200):
        self.elo_bucket_size = elo_bucket_size
        self.puzzles = []

        # read and store puzzles
        with open(puzzles_csv, 'r') as f:
            for line in f:
                fen, moves, rating, themes = line.strip().split(',')
                board = chess.Board(fen)
                moves = [chess.Move.from_uci(m) for m in moves.split(' ')]
                rating = int(rating)
                themes = themes.split(' ')
                if rating >= 3000: # rating too high
                    continue
                
                bucket = rating // self.elo_bucket_size
                themes.append('rating' + str(bucket * self.elo_bucket_size) + 'to' + str((bucket + 1) * self.elo_bucket_size - 1))

                self.puzzles.append((board, moves, themes))

        random.Random(42).shuffle(self.puzzles) # puzzles should be already sorted but just in case...
        #self.puzzles = self.puzzles[:100]
    
    def measure(self, engine_cmd: str | list[str]):
        def f(puzzle):
            board, moves, themes = puzzle
            return solve_puzzle(board, moves, themes)

        correct_moves = 0
        total_moves = 0

        themes_solved = {}
        themes_total = {}

        with ThreadPoolExecutor(max_workers=5, initializer=init_engine, initargs=(engine_cmd,)) as executor:
            results = list(tqdm(executor.map(f, self.puzzles), total=len(self.puzzles)))
            
            for (_, _, themes), (pz_correct_moves, pz_total_moves) in zip(self.puzzles, results):
                solved = (pz_correct_moves == pz_total_moves)

                correct_moves += pz_correct_moves
                total_moves += pz_total_moves

                for theme in themes:
                    themes_total[theme] = themes_total.get(theme, 0) + 1
                    themes_solved[theme] = themes_solved.get(theme, 0) + (1 if solved else 0)

        # exit all engines
        global engines
        for _, engine in engines.items():
            engine.quit()
        engines = {}

        result = []
        for theme in themes_total:
            result.append((theme, themes_solved[theme] / themes_total[theme]))

        return result, (correct_moves / total_moves)

if __name__ == '__main__':

    puzzle_acc = PuzzleAccuracy('../data/puzzles.csv')
    #a, b = puzzle_acc.measure('/home/mlomb/engines/stockfish-ubuntu-x86-64-avx2')
    a, b = puzzle_acc.measure('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/engine/target/release/engine')
    print(a, b)
