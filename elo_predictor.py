"""
Chess ELO prediction pipeline - v3 (perf pass)

Changes from v2:
  - FIXED: crash-path return tuple no longer drifts out of sync with the
    real feature tuple (it's now derived from len(FEATURE_NAMES_FROM_ENGINE),
    so it can't silently mismatch again).
  - FIXED: the "drop placeholder ELOs" filter now applies whether you load
    from cache or extract fresh, instead of only on the cached path.
  - Opening/midgame/endgame splits are now percentage-of-game-length based
    instead of fixed move counts, so short and long games get comparable
    phase boundaries.
  - New features: top3_move_rate, avg_sharpness, punishable_blunder_rate
    (requires MultiPV=3 analysis instead of MultiPV=1).
  - Two-pass adaptive depth: every game gets a fast depth-13 pass first;
    games a quick screening model predicts as high-rated get a second,
    deeper depth-19 pass, because depth 13 can't reliably separate a 2200
    from a 2800 player. The screening predictions are out-of-fold (via
    cross_val_predict), so nothing here leaks test labels.
  - Wider top-end bucket resolution when sampling (2000-2300 / 2300-2600 /
    2600-3000 instead of one wide 2000-2500 / 2500-3000).
  - Monotonic constraints on features with an unambiguous direction
    (more blunders -> never higher predicted ELO, etc.), built
    programmatically from a name->direction dict so the constraint list
    can't drift out of order the way a hand-typed one could.
  - Post-hoc isotonic calibration fit on a held-out calibration slice,
    to correct the regression-to-the-mean bias visible in the v2 scatter
    plot (weak players overpredicted, strong players underpredicted).
  - No clock/time-usage features - not available in this dataset.

Performance changes in this pass (no change to depth, multipv, sampling,
feature math, model, or calibration - so results should match the
previous version bit-for-bit, just faster):
  - Each worker process now keeps ONE Stockfish engine alive for its whole
    lifetime instead of spawning/killing a process per game. "Clear Hash"
    is still issued before every game, so games still can't influence each
    other's evaluations - only the process-spawn + UCI-handshake overhead
    is removed.
  - Engine hash table bumped from Stockfish's ~16MB default to
    ENGINE_HASH_MB (128MB). This only helps the engine reuse transposition
    entries to reach the SAME target depth faster - it doesn't change
    search depth, multipv, or which moves get evaluated.
  - Pass-1 and the deep re-analysis pass now share a single lazily-created
    ProcessPoolExecutor instead of each spinning up its own pool.
  - If a worker's engine process dies mid-game (rare), that one game is
    sacrificed (zero-features, same as the old "engine failed to start"
    fallback) and the worker transparently respawns a fresh engine for the
    next game, instead of silently degrading for the rest of the run.
"""

import concurrent.futures
import io
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.engine
import chess.pgn
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from xgboost import XGBRegressor

logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

# ---------------------------------------------------------
# CONFIG
# Must stay at true module level (outside `if __name__ == '__main__'`)
# so worker processes spawned by ProcessPoolExecutor can see them on import.
# ---------------------------------------------------------
STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"  # adjust for your OS
STANDARD_DEPTH = 13     # fast first pass, applied to every game
DEEP_DEPTH = 16         # second pass, applied only to games predicted as strong
DEEP_REANALYSIS_ELO_THRESHOLD = 1900
MULTIPV = 3             # needed for top3_move_rate + sharpness features
SAMPLE_SIZE = 9000      # set to a small int (e.g. 100) to sanity-check pass-1
                        # timing before committing to the full run; None = full run

# N_WORKERS x ENGINE_HASH_MB is the RAM budget for engine hash tables
# (8 x 128MB ~= 1GB here). If you have RAM to spare, raising ENGINE_HASH_MB
# helps the deep pass the most (it builds a much bigger search tree than
# depth 13 and thrashes a small hash table faster) - more so the higher
# DEEP_DEPTH is set.
# N_WORKERS: Stockfish scales poorly across hyperthreads, so pin this to
# physical core count rather than os.cpu_count() (which counts logical
# cores) if your CPU has hyperthreading/SMT.
N_WORKERS = 8
ENGINE_HASH_MB = 128

# Cache filenames encode SAMPLE_SIZE / DEEP_DEPTH on purpose: if you change
# either of those to iterate faster, you get a fresh (small, fast) cache
# instead of silently reloading a stale one built with the old settings -
# that mismatch is what made the SAMPLE_SIZE=200 edit look like it did
# nothing on the last run (the old 2000-game pass1 CSV was still on disk
# under the old fixed filename, so it got loaded as-is).
_sample_tag = f"n{SAMPLE_SIZE}" if SAMPLE_SIZE is not None else "full"
PASS1_CACHE = f"processed_chess_games_v3_pass1_{_sample_tag}.csv"
DEEP_CACHE = f"processed_chess_games_v3_deep_{_sample_tag}_d{DEEP_DEPTH}.csv"

FEATURE_NAMES_FROM_ENGINE = [
    'overall_acpl', 'opening_acpl', 'midgame_acpl', 'endgame_acpl',
    'cpl_std_dev', 'eval_variance', 'eval_swings',
    'blunders', 'mistakes', 'inaccuracies', 'total_moves',
    'blunders_per_move', 'mistakes_per_move', 'inaccuracies_per_move',
    'best_move_rate', 'perfect_move_rate',
    'top3_move_rate', 'avg_sharpness', 'punishable_blunder_rate',
]

# +1 = higher feature value -> should push predicted ELO up
# -1 = higher feature value -> should push predicted ELO down
#  0 = no assumed direction (left unconstrained)
MONOTONIC_DIRECTIONS = {
    'overall_acpl': -1, 'opening_acpl': -1, 'midgame_acpl': -1, 'endgame_acpl': -1,
    'cpl_std_dev': -1, 'eval_variance': 0, 'eval_swings': 0,
    'blunders': -1, 'mistakes': -1, 'inaccuracies': -1, 'total_moves': 0,
    'blunders_per_move': -1, 'mistakes_per_move': -1, 'inaccuracies_per_move': -1,
    'best_move_rate': 1, 'perfect_move_rate': 1,
    'top3_move_rate': 1, 'avg_sharpness': 0, 'punishable_blunder_rate': -1,
    'base_time': 0, 'deep_reanalyzed': 0,
}

# ---------------------------------------------------------
# PART 1: STOCKFISH FEATURE EXTRACTION
# ---------------------------------------------------------

# Persistent per-worker engine handle. Set once by _init_worker() when a
# ProcessPoolExecutor worker process starts, then reused for every game
# that worker processes (instead of popen_uci()/quit() per game).
_ENGINE = None


def _spawn_engine():
    """Starts a fresh configured Stockfish instance, or returns None on failure."""
    try:
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        eng.configure({"Threads": 1, "Hash": ENGINE_HASH_MB})
        return eng
    except Exception:
        return None


def _init_worker():
    """Runs once in each ProcessPoolExecutor worker process, before any games."""
    global _ENGINE
    _ENGINE = _spawn_engine()
    # Deliberately no explicit engine.quit()/atexit cleanup here. When this
    # worker process exits, the OS closes its file descriptors - including
    # the pipe to the Stockfish subprocess - so Stockfish sees EOF on
    # stdin and exits on its own (standard UCI engine behavior). Trying to
    # quit() gracefully from an atexit hook was observed to sometimes hang
    # executor.shutdown(wait=True) instead, since python-chess's engine
    # communication runs through a background thread/event loop that
    # doesn't reliably survive interpreter shutdown ordering.


def analyze_position(engine, board, depth, multipv):
    """Always returns a list of info dicts, even when multipv makes that a list of 1."""
    result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    if isinstance(result, dict):
        result = [result]
    return result


def process_single_game(args):
    """
    args: (pgn_moves, depth, multipv)
    depth/multipv are passed in (rather than hardcoded) so this same function
    serves both the fast first pass and the deep second pass.
    Uses this worker's persistent engine (see _init_worker) instead of
    spawning a new Stockfish process per game.
    """
    global _ENGINE
    pgn_moves, depth, multipv = args

    if _ENGINE is None:
        # Either the worker's engine failed at startup, or a previous game
        # killed it (see EngineTerminatedError handling below) - try once
        # to bring up a fresh one before giving up on this game too.
        _ENGINE = _spawn_engine()
    if _ENGINE is None:
        return tuple([0.0] * len(FEATURE_NAMES_FROM_ENGINE))
    engine = _ENGINE

    pgn_io = io.StringIO(str(pgn_moves))
    game = chess.pgn.read_game(pgn_io)

    if game is None:
        return tuple([0.0] * len(FEATURE_NAMES_FROM_ENGINE))

    board = game.board()
    centipawn_losses = []
    evaluations = []
    sharpness_values = []

    best_move_matches = 0
    top3_matches = 0
    perfect_moves = 0
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    eval_swings = 0
    punishable_blunders = 0

    try:
        # Cleared per game (cheap - in-memory only, no process restart) so
        # one game's search tree can never bleed into the next game's
        # evaluations, exactly like the old fresh-process-per-game did.
        engine.configure({"Clear Hash": True})

        info = analyze_position(engine, board, depth, multipv)

        for move in game.mainline_moves():
            current_turn = board.turn

            score_before = info[0]["score"].pov(current_turn).score(mate_score=10000)
            best_engine_move = info[0].get("pv", [None])[0]
            top_moves = [entry["pv"][0] for entry in info if entry.get("pv")]

            sharpness = 0
            if len(info) > 1:
                second_score = info[1]["score"].pov(current_turn).score(mate_score=10000)
                sharpness = max(0, score_before - second_score)

            board.push(move)

            info = analyze_position(engine, board, depth, multipv)
            score_after = info[0]["score"].pov(current_turn).score(mate_score=10000)

            loss = max(0, score_before - score_after)

            if current_turn == chess.WHITE:
                centipawn_losses.append(loss)
                evaluations.append(score_after)
                sharpness_values.append(sharpness)

                if move == best_engine_move:
                    best_move_matches += 1
                if move in top_moves:
                    top3_matches += 1
                if loss == 0:
                    perfect_moves += 1

                if loss > 300:
                    blunders += 1
                    if sharpness > 150:
                        punishable_blunders += 1
                elif loss >= 100:
                    mistakes += 1
                elif loss >= 50:
                    inaccuracies += 1

                if len(evaluations) > 1:
                    if abs(evaluations[-1] - evaluations[-2]) > 200:
                        eval_swings += 1

    except chess.engine.EngineTerminatedError:
        # The persistent engine itself died. The old code would just have
        # spawned a fresh process for the next game anyway, so mirror that:
        # this game is lost (zero-features), and the next call on this
        # worker will transparently respawn a new engine.
        _ENGINE = None
        return tuple([0.0] * len(FEATURE_NAMES_FROM_ENGINE))
    except Exception:
        # Same as the original: keep whatever partial data was gathered
        # before the error and compute features from that.
        pass

    total_moves = len(centipawn_losses)
    n = total_moves

    # Percentage-based phase boundaries instead of fixed move counts, so a
    # 20-move blitz game and an 80-move classical game get comparable splits.
    opening_losses = centipawn_losses[:max(1, int(n * 0.2))]
    midgame_losses = centipawn_losses[int(n * 0.2):int(n * 0.8)]
    endgame_losses = centipawn_losses[int(n * 0.8):]

    opening_acpl = sum(opening_losses) / len(opening_losses) if opening_losses else 0.0
    midgame_acpl = sum(midgame_losses) / len(midgame_losses) if midgame_losses else 0.0
    endgame_acpl = sum(endgame_losses) / len(endgame_losses) if endgame_losses else 0.0

    overall_acpl = sum(centipawn_losses) / len(centipawn_losses) if centipawn_losses else 0.0
    cpl_std_dev = float(np.std(centipawn_losses)) if centipawn_losses else 0.0
    eval_variance = float(np.std(evaluations)) if evaluations else 0.0
    avg_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0

    best_move_rate = best_move_matches / total_moves if total_moves > 0 else 0.0
    top3_move_rate = top3_matches / total_moves if total_moves > 0 else 0.0
    perfect_move_rate = perfect_moves / total_moves if total_moves > 0 else 0.0
    blunders_per_move = blunders / total_moves if total_moves > 0 else 0.0
    mistakes_per_move = mistakes / total_moves if total_moves > 0 else 0.0
    inaccuracies_per_move = inaccuracies / total_moves if total_moves > 0 else 0.0
    punishable_blunder_rate = punishable_blunders / total_moves if total_moves > 0 else 0.0

    return (
        overall_acpl, opening_acpl, midgame_acpl, endgame_acpl,
        cpl_std_dev, eval_variance, eval_swings,
        blunders, mistakes, inaccuracies, total_moves,
        blunders_per_move, mistakes_per_move, inaccuracies_per_move,
        best_move_rate, perfect_move_rate,
        top3_move_rate, avg_sharpness, punishable_blunder_rate,
    )


# ---------------------------------------------------------
# PART 2: EXECUTION & MACHINE LEARNING
# ---------------------------------------------------------

if __name__ == '__main__':

    # Lazily-created, shared across BOTH the pass-1 and deep-pass map()
    # calls below, so worker/engine spawn cost (N_WORKERS Stockfish
    # instances) is paid at most once per run, not once per phase - and
    # not at all if both caches already exist.
    executor = None

    def _get_executor():
        global executor
        if executor is None:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=N_WORKERS, initializer=_init_worker
            )
        return executor

    try:
        # ---- Load pass-1 features (cached or freshly extracted) ----
        if os.path.exists(PASS1_CACHE):
            print(f"Found existing '{PASS1_CACHE}'! Loading pre-extracted data...")
            df = pd.read_csv(PASS1_CACHE)
            df = df[~df['white_elo'].isin([1200, 1500, 2000])].reset_index(drop=True)
        else:
            print("Loading raw dataset 'chess_data.csv'...")
            df = pd.read_csv("chess_data.csv", sep=';')

            df['base_time'] = df['time_control'].astype(str).str.split('+').str[0]
            df['base_time'] = pd.to_numeric(df['base_time'], errors='coerce').fillna(600)

            df = df.dropna(subset=['moves', 'white_elo'])
            df = df[df['moves'].str.len() > 150]

            # Drop placeholder / default ratings (e.g. chess.com seeds new accounts
            # at round numbers like 1200) so they aren't treated as real skill
            # labels. Applied here too now, not just on the cached-load path.
            df = df[~df['white_elo'].isin([1200, 1500, 2000])]

            # Balanced sampling, with finer resolution at the top end -- that's
            # where depth-13 analysis has the least room to distinguish players,
            # so more examples there matters more than at the low end.
            buckets = [
                (100, 500), (500, 1000), (1000, 1500), (1500, 2000),
                (2000, 2300), (2300, 2600), (2600, 3001),
            ]
            per_bucket_cap = 1300
            sampled = []
            for lo, hi in buckets:
                b = df[(df['white_elo'] >= lo) & (df['white_elo'] < hi)]
                n_available = len(b)
                if n_available < per_bucket_cap:
                    print(f"  Bucket [{lo},{hi}): only {n_available} games available (< {per_bucket_cap} cap)")
                sampled.append(b.sample(n=min(per_bucket_cap, n_available), random_state=42))
            df = pd.concat(sampled)

            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            if SAMPLE_SIZE is not None:
                df = df.iloc[:SAMPLE_SIZE].reset_index(drop=True)
                print(f"SAMPLE_SIZE is set -- using only {len(df)} games for this run.")

            moves_list = df['moves'].tolist()

            print(f"Extracting pass-1 (depth {STANDARD_DEPTH}, multipv {MULTIPV}) "
                  f"features for {len(moves_list)} games using {N_WORKERS} CPU cores...")

            pass1_args = [(m, STANDARD_DEPTH, MULTIPV) for m in moves_list]
            results = []
            # Scales to batch size instead of a fixed count, so a small
            # SAMPLE_SIZE run still prints ~10 updates instead of going
            # silent for the whole run (a fixed "every 500" never fires at
            # all once the batch is smaller than 500).
            print_every = max(1, len(pass1_args) // 10)
            for res in _get_executor().map(process_single_game, pass1_args):
                results.append(res)
                if len(results) % print_every == 0:
                    print(f"Processed {len(results)} / {len(moves_list)} games...")

            df[FEATURE_NAMES_FROM_ENGINE] = pd.DataFrame(
                results, index=df.index, columns=FEATURE_NAMES_FROM_ENGINE
            )
            df.to_csv(PASS1_CACHE, index=False)
            print(f"Pass-1 features backed up to '{PASS1_CACHE}'")

        # ---- Screening pass: out-of-fold rough ELO predictions for everyone ----
        # Used only to decide who gets deep re-analysis. Out-of-fold (via
        # cross_val_predict) so a game's own label doesn't leak into its own
        # screening prediction.
        print("Running 5-fold screening pass to flag high-rated games for deep re-analysis...")
        screen_features = FEATURE_NAMES_FROM_ENGINE + ['base_time']
        X_screen = df[screen_features]
        y_screen = df['white_elo']

        screening_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
        df['predicted_elo_pass1'] = cross_val_predict(
            screening_model, X_screen, y_screen,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )

        deep_mask = df['predicted_elo_pass1'] >= DEEP_REANALYSIS_ELO_THRESHOLD
        df['deep_reanalyzed'] = 0

        # ---- Deep re-analysis for games the screening pass thinks are strong ----
        if os.path.exists(DEEP_CACHE):
            print(f"Found existing '{DEEP_CACHE}'! Loading cached deep re-analysis...")
            deep_df = pd.read_csv(DEEP_CACHE, index_col=0)
        else:
            deep_indices = df[deep_mask].index.tolist()
            print(f"{len(deep_indices)} games predicted >= {DEEP_REANALYSIS_ELO_THRESHOLD} ELO. "
                  f"Re-analyzing at depth {DEEP_DEPTH}...")

            if deep_indices:
                deep_args = [(df.loc[i, 'moves'], DEEP_DEPTH, MULTIPV) for i in deep_indices]
                deep_results = []
                # Same reasoning as pass-1's print_every: this batch is
                # usually much smaller than 100, so a fixed "every 100"
                # print previously meant total silence for the whole
                # deep-reanalysis phase - easy to mistake for a hang.
                print_every = max(1, len(deep_args) // 10)
                for res in _get_executor().map(process_single_game, deep_args):
                    deep_results.append(res)
                    if len(deep_results) % print_every == 0:
                        print(f"Deep-reanalyzed {len(deep_results)} / {len(deep_indices)} games...")
                deep_df = pd.DataFrame(deep_results, columns=FEATURE_NAMES_FROM_ENGINE, index=deep_indices)
            else:
                deep_df = pd.DataFrame(columns=FEATURE_NAMES_FROM_ENGINE)

            deep_df.to_csv(DEEP_CACHE, index=True)
            print(f"Deep re-analysis backed up to '{DEEP_CACHE}'")

        if len(deep_df) > 0:
            df.loc[deep_df.index, FEATURE_NAMES_FROM_ENGINE] = deep_df[FEATURE_NAMES_FROM_ENGINE].values
            df.loc[deep_df.index, 'deep_reanalyzed'] = 1

    finally:
        # Tears down the pool exactly once, whether we used it for zero,
        # one, or both phases above. Worker processes exit normally here,
        # which closes their Stockfish pipes (see _init_worker) - no
        # explicit per-engine quit() needed.
        if executor is not None:
            executor.shutdown(wait=True)

    # ---- Extra "free" features from raw columns already in the cached CSV ----
    df['eco_class'] = df['ECO'].fillna('U').astype(str).str[0]
    eco_dummies = pd.get_dummies(df['eco_class'], prefix='eco')
    df[eco_dummies.columns] = eco_dummies

    result_dummies = pd.get_dummies(df['result'].astype(str), prefix='result')
    df[result_dummies.columns] = result_dummies

    start_ts = pd.to_datetime(df['UTC_date'].astype(str) + ' ' + df['start_time'].astype(str), errors='coerce')
    end_ts = pd.to_datetime(df['end_date'].astype(str) + ' ' + df['end_time'].astype(str), errors='coerce')
    df['game_duration_sec'] = (end_ts - start_ts).dt.total_seconds()
    df['game_duration_sec'] = df['game_duration_sec'].fillna(df['game_duration_sec'].median())

    extra_features = list(eco_dummies.columns) + list(result_dummies.columns) + ['game_duration_sec']
    for f in extra_features:
        MONOTONIC_DIRECTIONS.setdefault(f, 0)  # no assumed up/down direction for these

    # ---- Final feature set ----
    feature_columns = FEATURE_NAMES_FROM_ENGINE + ['base_time', 'deep_reanalyzed'] + extra_features
    X = df[feature_columns]
    y = df['white_elo']

    monotone_constraints = '(' + ','.join(str(MONOTONIC_DIRECTIONS[f]) for f in feature_columns) + ')'
    print("Splitting data (train / calibration / test)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

    print("Training final XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=monotone_constraints,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model("chess_elo_model_v3.json")
    print("Model saved successfully!")

    print("Fitting isotonic calibration on held-out calibration slice...")
    cal_predictions = xgb_model.predict(X_cal)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cal_predictions, y_cal)

    print("Evaluating on test set...")
    raw_test_predictions = xgb_model.predict(X_test)
    calibrated_test_predictions = calibrator.transform(raw_test_predictions)

    raw_mae = mean_absolute_error(y_test, raw_test_predictions)
    calibrated_mae = mean_absolute_error(y_test, calibrated_test_predictions)

    print(f"\nRaw model MAE:        ~{raw_mae:.0f} ELO points")
    print(f"Calibrated model MAE: ~{calibrated_mae:.0f} ELO points\n")

    print("MAE by actual ELO range (raw predictions):")
    eval_df = pd.DataFrame({'actual': y_test.values, 'predicted': raw_test_predictions})
    eval_df['range'] = pd.cut(eval_df['actual'], bins=[0, 500, 1000, 1500, 2000, 2500, 3001])
    for rng, group in eval_df.groupby('range', observed=True):
        print(f"  {rng}: MAE ~{mean_absolute_error(group['actual'], group['predicted']):.0f} (n={len(group)})")

    # ---------------------------------------------------------
    # PART 3: VISUALIZATIONS
    # ---------------------------------------------------------
    print("\nGenerating visualizations...")

    plt.figure(figsize=(12, 9))
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center', color='skyblue')
    plt.yticks(range(len(sorted_idx)), [feature_columns[i] for i in sorted_idx])
    plt.title('XGBoost V3 Feature Importance')
    plt.xlabel('Relative Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance_v3.png')
    plt.show(block=False)

    actual_elos = y_test.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, preds, title in [
        (axes[0], raw_test_predictions, 'Raw'),
        (axes[1], calibrated_test_predictions, 'Calibrated'),
    ]:
        ax.scatter(actual_elos, preds, alpha=0.5, color='royalblue', edgecolors='w', s=60)
        min_val = min(min(actual_elos), min(preds))
        max_val = max(max(actual_elos), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax.set_title(f'Actual vs. Predicted ELO ({title})')
        ax.set_xlabel('Actual ELO')
        ax.set_ylabel('Predicted ELO')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_elo_v3.png')
    plt.show()