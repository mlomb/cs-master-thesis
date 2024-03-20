from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import chess
import chess.engine
from tqdm import tqdm

engines: dict[chess.engine.SimpleEngine] = {}
exit_stack = ExitStack()

def init_engine(engine_cmd: str):
    engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)
    engines[threading.current_thread()] = exit_stack.enter_context(engine)


def solve_puzzle(board: chess.Board, solution: list[chess.Move]) -> bool:
    solution = solution.copy() # clone
    engine = engines[threading.current_thread()]

    while len(solution) > 0:
        opp_move, *solution = solution
        expected_move, *solution = solution

        # play opponent move
        board.push(opp_move)

        res = engine.play(board, chess.engine.Limit(time=50/1000))

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
        with open('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/puzzles.csv', 'r') as f:
            for line in f:
                fen, moves, rating = line.strip().split(',')
                board = chess.Board(fen)
                moves = [chess.Move.from_uci(m) for m in moves.split(' ')]
                rating = int(rating)

                self.puzzles.append((board, moves, rating))
    
    def measure(self, engine_cmd: str):
        def f(puzzle):
            board, moves, rating = puzzle
            if solve_puzzle(board, moves):
                return rating
            else:
                return -rating

        with ThreadPoolExecutor(max_workers=5, initializer=init_engine, initargs=(engine_cmd,)) as executor:
            results = list(tqdm(executor.map(f, self.puzzles), total=len(self.puzzles)))
            
            print(results)

        return 0

        ELO_BUCKET_SIZE = 200
        NUM_BUCKETS = 3000 // ELO_BUCKET_SIZE
        buckets_solved = [0] * NUM_BUCKETS
        buckets_total = [0] * NUM_BUCKETS

        for (board, moves, rating) in tqdm(self.puzzles):
            bucket = rating // ELO_BUCKET_SIZE
            if bucket >= NUM_BUCKETS: # too high rating
                continue

            buckets_total[bucket] += 1

            if self.solve_puzzle(engine, board, moves):
                buckets_solved[bucket] += 1

        print(buckets_total)

        engine.quit()

        return [s / t if t > 0 else 0 for s, t in zip(buckets_solved, buckets_total)]

if __name__ == '__main__':

    puzzle_acc = PuzzleAccuracy()
    res = puzzle_acc.measure('/mnt/c/Users/mlomb/Desktop/Tesis/cs-master-thesis/bot/target/release/bot')
    #res = puzzle_acc.measure('/home/mlomb/engines/stockfish-ubuntu-x86-64-avx2')
    print(res)
