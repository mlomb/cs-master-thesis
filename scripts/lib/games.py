from dataclasses import dataclass
import re
import subprocess
import tempfile
from tqdm import tqdm
from paths import CUTECHESS_CLI_BIN, OPENING_BOOK, ENGINE_BIN, ORDO_BIN, STOCKFISH_BIN


@dataclass
class Engine:
    """
    Runnable engine description with corresponding limits.
    """
    name: str
    cmd: str
    args: list[str] | None = None
    movetime: float | None = 0.05
    nodes: int | None = None
    depth: int | None = None
    elo: int | None = None # only works in Stockfish


def run_games(
    engines: list[Engine],
    n: int = 100,
    concurrency: int = 16,
    pgn_file: str = "out.pgn",
    seed: int = 42
):
    """
    Runs `n` games between each pair of engines in the list
    Stores the results in the given PGN file
    """
    command = []
    command += [
        # https://manpages.ubuntu.com/manpages/trusty/en/man6/cutechess-cli.6.html
        CUTECHESS_CLI_BIN,
        '-rounds', str(n),
        '-concurrency', str(concurrency),
        '-openings', f'file={OPENING_BOOK}', 'format=epd', 'order=random',
        '-srand', str(seed),
        '-games', '2', # due repeat
        '-repeat',
        '-pgnout', f'{pgn_file}',
        '-recover', # recover from crashes
        '-draw', 'movenumber=40', 'movecount=8', 'score=10', # adjudicate draws
        '-each', 'timemargin=2000',
    ]

    # add engines
    for engine in engines:
        command += ['-engine', f'name={engine.name}', f'cmd={engine.cmd}', 'proto=uci', 'restart=off']
        if engine.args:
            command += [f'arg={" ".join(engine.args)}']
        if engine.movetime:
            command += [f'st={engine.movetime}']
        if engine.nodes:
            command += [f'nodes={engine.nodes}']
        if engine.depth:
            command += [f'depth={engine.depth}']
        if engine.elo:
            command += [
                'option.Threads=1',
                'option.UCI_LimitStrength=true',
                'option.UCI_Elo=' + str(engine.elo),
            ]

    # print("Running:", " ".join(command))

    # run games
    with tqdm(total=2 * n, desc=f"Running games ({len(engines)} engines)") as pbar:
        program = subprocess.Popen(command, stdout=subprocess.PIPE)

        for line in program.stdout:
            # print(line.decode("utf-8").strip())
            if b"Finished game" in line:
                pbar.update(1)

        program.wait()
        pbar.close()

        assert program.returncode == 0, "Error running games"

def run_ordo(pgn_file: str):
    """
    Runs Ordo on the given PGN file
    Returns a dictionary with the results, where the key is the engine name and the value is
    a tuple with the elo, error margin, and the ratio of points (points / played)
    """
    command = [
        ORDO_BIN,
        '-q', # quiet
        '-J', # add confidence column
        '-p', f'{pgn_file}',
        '--draw-auto',
        '--white-auto',
        '-s', '100' # sims to calculate errors (same used in Stockfish trainer)
    ]

    # print("Running:", " ".join(command))
    output = subprocess.check_output(command).decode("utf-8")

    results = dict()

    # parse output
    for line in output.split("\n"):
        #   # PLAYER              :  RATING  ERROR  POINTS  PLAYED   (%)  CFS(%)
        #   1 engine              :  2367.2   81.6    13.0      20    65      95
        if line.startswith("   "):
            print(line)

            if not line.startswith("   #"):
                line = re.sub('[\s|\t]+', ' ', line).strip()
                parts = line.split(" ")
                
                name = parts[1]
                elo = float(parts[3])
                error = float(parts[4])
                points = float(parts[5])
                played = float(parts[6])

                results[name] = (elo, error, points / played)

    return results

def measure_perf_diff(
    engine1: Engine = Engine(name="engine", cmd=ENGINE_BIN),
    engine2: Engine = Engine(name="stockfish-elo2650", cmd=STOCKFISH_BIN, elo=2650), # reference engine
    n: int = 100,
    concurrency: int = 16,
) -> tuple[int, int, int]:
    """
    Measure the performance difference of one engine against another. The anchor is engine 2.
    Returns a tuple with the ELO difference, the error margin, and the ratio of points of engine 1 (points / played)
    """
    with tempfile.NamedTemporaryFile(suffix=".pgn") as tmp:
        pgn_file = tmp.name

        print(pgn_file)

        run_games(
            engines=[engine1, engine2],
            n=n,
            concurrency=concurrency,
            pgn_file=pgn_file
        )
        res = run_ordo(pgn_file=pgn_file)

        assert engine1.name in res, f"Engine {engine1.name} not found in results"
        assert engine2.name in res, f"Engine {engine2.name} not found in results"

        elo1, error1, ratio1 = res[engine1.name]
        elo2, error2, ratio2 = res[engine2.name]
        diff = elo1 - elo2

        assert abs(error1 - error2) < 0.00001, "Error margin should be the same for both engines"
        assert abs(ratio1 + ratio2 - 1) < 0.00001, "Sum of ratios should be 1"

        return diff, error1, ratio1


if __name__ == '__main__':
    # elo, error, points = measure_perf_diff(
    #     engine2=Engine(name="aaa", cmd="aaa"),
    #     n=100,
    # )
    # print(f"ELO: {elo} ± {error} Points: {points}")

    with open("elo.csv", "w") as f:
        f.write("n,elo_diff,error,points\n")
        f.flush()

        for n in [100, 200, 300, 400]:
            for rep in range(5):
                elo, error, points = measure_perf_diff(n=n)
                print(f"ELO: {elo} ± {error} Points: {points}")

                f.write(f"{n},{elo},{error},{points}\n")
                f.flush()
