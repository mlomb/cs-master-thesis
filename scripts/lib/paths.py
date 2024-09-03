import os

def rel_to_abs(path: str):
    """
    Takes a relative path from this script (paths.py) and converts it to an absolute path
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

# engines
ENGINE_BIN = rel_to_abs("../../engine/target/release/engine")
STOCKFISH_BIN = rel_to_abs("../bin/stockfish-ubuntu-x86-64-avx2")

# data
DEFAULT_DATASET = "/mnt/c/datasets/compact.plain"
PUZZLES_DATA = rel_to_abs("../data/puzzles.csv")
OPENING_BOOK = rel_to_abs("../data/noob_3moves.epd")

# tools
TOOLS_BIN = rel_to_abs("../../tools/target/release/tools")
CUTECHESS_CLI_BIN = rel_to_abs("../bin/cutechess-cli")
ORDO_BIN = rel_to_abs("../bin/ordo")

if __name__ == '__main__':
    print("ENGINE_BIN", ENGINE_BIN)
    print("STOCKFISH_BIN", STOCKFISH_BIN)
    print("DEFAULT_DATASET", DEFAULT_DATASET)
    print("PUZZLES_DATA", PUZZLES_DATA)
    print("OPENING_BOOK", OPENING_BOOK)
    print("TOOLS_BIN", TOOLS_BIN)
    print("CUTECHESS_CLI_BIN", CUTECHESS_CLI_BIN)
    print("ORDO_BIN", ORDO_BIN)
