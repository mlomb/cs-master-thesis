from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import chess
import chess.engine
from tqdm import tqdm
import random

engines: dict[chess.engine.SimpleEngine] = {}

def init_engine(engine_cmd: str):
    engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)
    engines[threading.current_thread()] = engine


def solve_puzzle(board: chess.Board, solution: list[chess.Move]) -> bool:
    board = board.copy()
    solution = solution.copy() # clone
    engine = engines[threading.current_thread()]

    while len(solution) > 0:
        opp_move, *solution = solution
        expected_move, *solution = solution

        # play opponent move
        board.push(opp_move)

        res = engine.play(board, chess.engine.Limit(time=100/1000))

        if res.move != expected_move:
            # Mate in 1 puzzles in Lichess can have more than one solution
            return board.is_checkmate()

        # play engine move
        board.push(expected_move)
    
    return True


class PuzzleAccuracy:
    def __init__(self):
        self.puzzles = []

        # read and store puzzles
        with open('/mnt/c/Users/mlomb/OneDrive/Escritorio/cs-master-thesis/puzzles.csv', 'r') as f:
            for line in f:
                fen, moves, rating = line.strip().split(',')
                board = chess.Board(fen)
                moves = [chess.Move.from_uci(m) for m in moves.split(' ')]
                rating = int(rating)

                self.puzzles.append((board, moves, rating))

        random.Random(42).shuffle(self.puzzles)
        self.puzzles = self.puzzles[:40]
    
    def measure(self, engine_cmd: str):
        def f(puzzle):
            board, moves, _ = puzzle
            return solve_puzzle(board, moves)

        ELO_BUCKET_SIZE = 200
        NUM_BUCKETS = 3000 // ELO_BUCKET_SIZE
        buckets_solved = [0] * NUM_BUCKETS
        buckets_total = [0] * NUM_BUCKETS

        with ThreadPoolExecutor(max_workers=5, initializer=init_engine, initargs=(engine_cmd,)) as executor:
            solved = list(tqdm(executor.map(f, self.puzzles), total=len(self.puzzles)))
            
            for (_, _, rating), solved in zip(self.puzzles, solved):
                b = rating // ELO_BUCKET_SIZE
                if b >= NUM_BUCKETS: # too high rating
                    continue

                buckets_total[b] += 1
                if solved:
                    buckets_solved[b] += 1
        
        # exit all engines
        global engines
        for _, engine in engines.items():
            engine.quit()
        engines = {}

        result = []
        for b in range(0, NUM_BUCKETS):
            if buckets_total[b] > 0:
                result.append((b * ELO_BUCKET_SIZE, buckets_solved[b] / buckets_total[b]))

        return result

if __name__ == '__main__':

    puzzle_acc = PuzzleAccuracy()
    res = puzzle_acc.measure('/mnt/c/Users/mlomb/OneDrive/Escritorio/cs-master-thesis/bot/target/release/bot')
    #res = puzzle_acc.measure('/home/mlomb/engines/stockfish-ubuntu-x86-64-avx2')
    print(res)
