from concurrent.futures import ThreadPoolExecutor
import chess
import chess.engine
from tqdm import tqdm
from paths import PUZZLES_DATA, ENGINE_BIN

def solve_puzzle(engine_cmd: str, board: chess.Board, solution: list[chess.Move]) -> tuple[int, int]:
    board = board.copy()
    solution = solution.copy() # clone
    engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)

    correct_moves = 0
    total_moves = 0

    while len(solution) > 0:
        opp_move, *solution = solution
        expected_move, *solution = solution

        # play opponent move
        board.push(opp_move)

        # ask engine for best move
        res = engine.play(board, chess.engine.Limit(nodes=30_000))

        # play expected move
        board.push(expected_move)

        # Mate in 1 puzzles can have more than one solution (Lichess)
        if res.move == expected_move or board.is_checkmate():
            correct_moves += 1
        total_moves += 1

    engine.close()
    
    return (correct_moves, total_moves)


class Puzzles:
    def __init__(self, puzzles_csv: str = PUZZLES_DATA, elo_bucket_size=200):
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

                bucket = rating // self.elo_bucket_size
                themes.append('rating' + str(bucket * self.elo_bucket_size) + 'to' + str((bucket + 1) * self.elo_bucket_size - 1))

                if self.should_skip(board, moves, themes, rating):
                    continue

                self.puzzles.append((board, moves, themes))

        #self.puzzles = self.puzzles[:100]
    
    def measure(self, engine_cmd: str | list[str]):
        def f(puzzle):
            board, moves, _ = puzzle
            return solve_puzzle(engine_cmd, board, moves)

        correct_moves = 0
        total_moves = 0

        themes_solved = {}
        themes_total = {}

        with ThreadPoolExecutor(max_workers=None) as executor:
            results = list(tqdm(executor.map(f, self.puzzles), total=len(self.puzzles), desc="Running puzzles"))

            for (_, _, themes), (pz_correct_moves, pz_total_moves) in zip(self.puzzles, results):
                solved = (pz_correct_moves == pz_total_moves)

                correct_moves += pz_correct_moves
                total_moves += pz_total_moves

                for theme in themes:
                    themes_total[theme] = themes_total.get(theme, 0) + 1
                    themes_solved[theme] = themes_solved.get(theme, 0) + (1 if solved else 0)

        result = []
        for theme in themes_total:
            result.append((theme, themes_solved[theme] / themes_total[theme]))

        return result, (correct_moves / total_moves)
    
    def should_skip(self, board: chess.Board, solution: list[chess.Move], themes: list[str], rating: int) -> bool:
        if rating >= 3000: # rating too high
            return True
        if rating < 1000: # rating too low
            return True

        board = board.copy()
        solution = solution.copy()
        while len(solution) > 0:
            opp_move, *solution = solution
            expected_move, *solution = solution
            
            board.push(opp_move)

            if len(list(board.legal_moves)) <= 2:
                # if there is only one legal move in a step of the puzzle
                # we skip it
                return True

            board.push(expected_move)
        
        if board.is_checkmate():
            # skip puzzles that end in checkmate
            return True

        # skip easy themes
        SKIP_THEMES = [
            "mate",
            "oneMove",
        ]
        return any(theme in SKIP_THEMES for theme in themes)

if __name__ == '__main__':
    p = Puzzles()
    #a, b = p.measure('/mnt/c/datasets/stockfish-ubuntu-x86-64-avx2')
    a, b = p.measure([ENGINE_BIN])
    print(a, b)
