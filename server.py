import os
import io
import re
import shutil
import numpy as np
import pandas as pd
import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor

from chessimg2pos import predict_fen 

# ==========================================
# 1. PYTORCH CNN ARCHITECTURE (Puzzle Solver)
# ==========================================
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        return x

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# ==========================================
# 2. START THE SERVER & LOAD BOTH MODELS
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading XGBoost v3 ELO model...")
xgb_model = XGBRegressor()
xgb_model.load_model("chess_elo_model_v3.json")
# Dynamically pull the exact features the model expects
EXPECTED_FEATURES = xgb_model.get_booster().feature_names

print("Loading PyTorch CNN model...")
cnn_model = ChessCNN()
cnn_model.load_state_dict(torch.load("chess_model.pth", map_location=torch.device('cpu')))
cnn_model.eval()
print("Both models loaded successfully! Waiting for requests...")

ENGINE_FEATURE_NAMES = [
    'overall_acpl', 'opening_acpl', 'midgame_acpl', 'endgame_acpl',
    'cpl_std_dev', 'eval_variance', 'eval_swings',
    'blunders', 'mistakes', 'inaccuracies', 'total_moves',
    'blunders_per_move', 'mistakes_per_move', 'inaccuracies_per_move',
    'best_move_rate', 'perfect_move_rate',
    'top3_move_rate', 'avg_sharpness', 'punishable_blunder_rate',
]

# ==========================================
# 3. EXTRACTION HELPER FOR V3 ELO MODEL
# ==========================================
def extract_v3_features(pgn_moves, engine, depth=13, multipv=3):
    pgn_io = io.StringIO(str(pgn_moves))
    game = chess.pgn.read_game(pgn_io)
    
    # Default values for metadata
    base_time = 600.0
    eco_class = "U"
    result = "*"
    game_duration_sec = 600.0

    if game is None:
        return [0.0] * len(ENGINE_FEATURE_NAMES), base_time, eco_class, result, game_duration_sec

    # Extract headers
    if "TimeControl" in game.headers:
        tc = str(game.headers["TimeControl"]).split('+')[0]
        try: base_time = float(tc)
        except ValueError: pass

    if "ECO" in game.headers:
        eco_class = str(game.headers["ECO"])[0]
        
    if "Result" in game.headers:
        result = str(game.headers["Result"])

    try:
        if all(k in game.headers for k in ["UTCDate", "StartTime", "EndDate", "EndTime"]):
            start = pd.to_datetime(f"{game.headers['UTCDate']} {game.headers['StartTime']}")
            end = pd.to_datetime(f"{game.headers['EndDate']} {game.headers['EndTime']}")
            game_duration_sec = (end - start).total_seconds()
    except Exception:
        pass

    board = game.board()
    centipawn_losses, evaluations, sharpness_values = [], [], []
    best_move_matches = top3_matches = perfect_moves = 0
    blunders = mistakes = inaccuracies = 0
    eval_swings = punishable_blunders = 0

    try:
        engine.configure({"Clear Hash": True})

        def analyze_position(b):
            res = engine.analyse(b, chess.engine.Limit(depth=depth), multipv=multipv)
            return [res] if isinstance(res, dict) else res

        info = analyze_position(board)

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
            info = analyze_position(board)
            score_after = info[0]["score"].pov(current_turn).score(mate_score=10000)
            loss = max(0, score_before - score_after)

            if current_turn == chess.WHITE:
                centipawn_losses.append(loss)
                evaluations.append(score_after)
                sharpness_values.append(sharpness)

                if move == best_engine_move: best_move_matches += 1
                if move in top_moves: top3_matches += 1
                if loss == 0: perfect_moves += 1

                if loss > 300:
                    blunders += 1
                    if sharpness > 150: punishable_blunders += 1
                elif loss >= 100: mistakes += 1
                elif loss >= 50: inaccuracies += 1

                if len(evaluations) > 1 and abs(evaluations[-1] - evaluations[-2]) > 200:
                    eval_swings += 1

    except Exception:
        pass

    total_moves = len(centipawn_losses)
    n = total_moves

    if n == 0:
        return [0.0] * len(ENGINE_FEATURE_NAMES), base_time, eco_class, result, game_duration_sec

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

    engine_features = [
        overall_acpl, opening_acpl, midgame_acpl, endgame_acpl,
        cpl_std_dev, eval_variance, eval_swings,
        blunders, mistakes, inaccuracies, total_moves,
        blunders_per_move, mistakes_per_move, inaccuracies_per_move,
        best_move_rate, perfect_move_rate,
        top3_move_rate, avg_sharpness, punishable_blunder_rate,
    ]

    return engine_features, base_time, eco_class, result, game_duration_sec


# ==========================================
# 4. HELPER FUNCTIONS FOR PUZZLE SOLVER
# ==========================================
def fix_bad_fen(bad_fen):
    return re.sub(r'1+', lambda match: str(len(match.group())), bad_fen)

def rotate_fen_180(base_fen):
    rows = base_fen.split('/')
    rotated_rows = [row[::-1] for row in reversed(rows)]
    return '/'.join(rotated_rows)

def auto_detect_orientation(base_fen):
    rows = base_fen.split('/')
    bottom_half = "".join(rows[4:8]) 
    white_count = sum(1 for char in bottom_half if char.isupper())
    black_count = sum(1 for char in bottom_half if char.islower())
    if black_count > white_count:
        return rotate_fen_180(base_fen)
    return base_fen

def fen_to_tensor(fen_string):
    tensor = torch.zeros((1, 12, 8, 8), dtype=torch.float32)
    board_part = fen_string.split(' ')[0]
    rows = board_part.split('/')
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for char in row:
            if char.isdigit():
                c_idx += int(char)
            else:
                if char in PIECE_TO_INDEX:
                    tensor[0, PIECE_TO_INDEX[char], r_idx, c_idx] = 1.0
                c_idx += 1
    return tensor

def evaluate_static_board(model, board_fen):
    tensor = fen_to_tensor(board_fen)
    with torch.no_grad():
        return model(tensor).item()

MATE_SCORE = 10000.0

def minimax(board, depth, alpha, beta, is_maximizing, model, ply_from_root=0):
    if board.is_game_over():
        if board.is_checkmate():
            return (-MATE_SCORE + ply_from_root) if is_maximizing else (MATE_SCORE - ply_from_root), None
        return 0.0, None 
    if depth == 0:
        return evaluate_static_board(model, board.fen()), None

    best_move = None
    if is_maximizing:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False, model, ply_from_root + 1)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True, model, ply_from_root + 1)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return min_eval, best_move

# ==========================================
# 5. FASTAPI ENDPOINTS
# ==========================================
class GameRequest(BaseModel):
    pgn_moves: str

@app.post("/predict_elo")
def predict_elo(request: GameRequest):
    stockfish_path = "stockfish" if shutil.which("stockfish") else "./stockfish/stockfish-windows-x86-64-avx2.exe"
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 128})
    
    # Extract features, including headers that were dynamically one-hot encoded in training
    engine_features, base_time, eco_class, result, game_duration_sec = extract_v3_features(
        request.pgn_moves, engine, depth=13, multipv=3
    )
    engine.quit()
    
    # Safely reconstruct the dictionary matching EXACTLY what the model expects
    data_dict = {name: val for name, val in zip(ENGINE_FEATURE_NAMES, engine_features)}
    data_dict['base_time'] = base_time
    data_dict['deep_reanalyzed'] = 0.0
    data_dict['game_duration_sec'] = game_duration_sec
    
    eco_col_name = f"eco_{eco_class}"
    result_col_name = f"result_{result}"

    final_row = {}
    for feature in EXPECTED_FEATURES:
        if feature in data_dict:
            final_row[feature] = data_dict[feature]
        elif feature == eco_col_name or feature == result_col_name:
            final_row[feature] = 1.0
        else:
            final_row[feature] = 0.0  # Fill missing one-hot variables with 0

    game_df = pd.DataFrame([final_row], columns=EXPECTED_FEATURES)
    predicted_elo = xgb_model.predict(game_df)[0]
    
    acpl = engine_features[0] 
    blunders = int(engine_features[7])
    mistakes = int(engine_features[8])
    
    return {
        "acpl": round(acpl, 1),
        "blunders": blunders,
        "mistakes": mistakes,
        "estimated_elo": int(predicted_elo)
    }

@app.post("/solve_position")
async def solve_position(image: UploadFile = File(...)):
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    try:
        raw_fen = predict_fen(temp_path)
        clean_fen = fix_bad_fen(raw_fen)
        oriented_fen = auto_detect_orientation(clean_fen)
        board = chess.Board(f"{oriented_fen} w - - 0 1")
        
        search_depth = 4
        score, best_move = minimax(
            board, depth=search_depth, alpha=-float('inf'), beta=float('inf'), 
            is_maximizing=(board.turn == chess.WHITE), model=cnn_model
        )
        
        if abs(score) > 9000.0:
            plies_to_mate = int(MATE_SCORE - abs(score))
            moves_to_mate = (plies_to_mate + 1) // 2
            winner = "White" if score > 0 else "Black"
            summary = f"MATE DETECTED! Mate in {moves_to_mate} for {winner}."
        else:
            leader = "White" if score > 0 else "Black"
            summary = f"{leader} leads by {abs(score):.2f} pawns."
            
        move_list = [best_move.uci()] if best_move else []
            
        return {
            "summary": summary,
            "moves": move_list
        }
        
    except Exception as e:
        return {"summary": f"Error parsing board: {str(e)}", "moves": []}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)