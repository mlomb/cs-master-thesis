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


def solve_puzzle(board: chess.Board, solution: list[chess.Move]) -> tuple[int, int]:
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

        # Mate in 1 puzzles in Lichess can have more than one solution
        if res.move == expected_move or board.is_checkmate():
            correct_moves += 1
        total_moves += 1

        # play engine move
        board.push(expected_move)
    
    return (correct_moves, total_moves)


class PuzzleAccuracy:
    def __init__(self, puzzles_csv: str, elo_bucket_size=200):
        self.elo_bucket_size = elo_bucket_size
        self.puzzles = []

        # read and store puzzles
        with open(puzzles_csv, 'r') as f:
            for line in f:
                fen, moves, rating = line.strip().split(',')
                board = chess.Board(fen)
                moves = [chess.Move.from_uci(m) for m in moves.split(' ')]
                rating = int(rating)
                if rating >= 3000: # rating too high
                    continue

                self.puzzles.append((board, moves, rating))

        random.Random(42).shuffle(self.puzzles)
        self.puzzles = self.puzzles[:1000]
    
    def measure(self, engine_cmd: str | list[str]):
        def f(puzzle):
            board, moves, _ = puzzle
            return solve_puzzle(board, moves)

        NUM_BUCKETS = 3000 // self.elo_bucket_size
        buckets_solved = [0] * NUM_BUCKETS
        buckets_total = [0] * NUM_BUCKETS
        correct_moves = 0
        total_moves = 0

        with ThreadPoolExecutor(max_workers=5, initializer=init_engine, initargs=(engine_cmd,)) as executor:
            results = list(tqdm(executor.map(f, self.puzzles), total=len(self.puzzles)))
            
            for (_, _, rating), (pz_correct_moves, pz_total_moves) in zip(self.puzzles, results):
                bucket = rating // self.elo_bucket_size
                solved = (correct_moves == total_moves)

                correct_moves += pz_correct_moves
                total_moves += pz_total_moves

                buckets_total[bucket] += 1
                if solved:
                    buckets_solved[bucket] += 1

        # exit all engines
        global engines
        for _, engine in engines.items():
            engine.quit()
        engines = {}

        result = []
        for b in range(0, NUM_BUCKETS):
            if buckets_total[b] > 0:
                result.append((
                    b * self.elo_bucket_size,
                    (b+1) * self.elo_bucket_size - 1,
                    buckets_solved[b] / buckets_total[b]
                ))

        return result, (correct_moves / total_moves)

if __name__ == '__main__':

    puzzle_acc = PuzzleAccuracy('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/puzzles.csv')
    res = puzzle_acc.measure('/home/mlomb/engines/stockfish-ubuntu-x86-64-avx2')
    #res = puzzle_acc.measure('/mnt/c/Users/mlomb/OneDrive/Escritorio/cs-master-thesis/bot/target/release/bot')
    print(res)
