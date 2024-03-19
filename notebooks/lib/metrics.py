import chess
import chess.engine

class PuzzleAccuracy:
    def __init__(self, engine_cmd: str):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)

        board = chess.Board()
        
        pr = self.engine.play(board, chess.engine.Limit(depth=10))
    
        print('PuzzleAccuracy initialized')
        print(pr)

        self.engine.quit()


if __name__ == '__main__':

    puzzle_acc = PuzzleAccuracy('/home/mlomb/engines/stockfish-ubuntu-x86-64-avx2')
