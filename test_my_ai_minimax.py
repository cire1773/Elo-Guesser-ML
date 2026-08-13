import torch
import torch.nn as nn
import re
import chess
from chessimg2pos import predict_fen

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
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

# ==========================================
# 2. HELPER FUNCTIONS & TENSOR CONVERSION
# ==========================================
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

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
                    channel = PIECE_TO_INDEX[char]
                    tensor[0, channel, r_idx, c_idx] = 1.0
                c_idx += 1
    return tensor

def evaluate_static_board(model, board_fen):
    """Uses your Neural Network to give a baseline score for a leaf board node."""
    tensor = fen_to_tensor(board_fen)
    with torch.no_grad():
        score = model(tensor).item()
    return score

# ==========================================
# 3. MINIMAX SEARCH ALGORITHM
# ==========================================
MATE_SCORE = 10000.0

def minimax(board, depth, alpha, beta, is_maximizing, model, ply_from_root=0):
    # Terminal check 1: Checkmate or Stalemate
    if board.is_game_over():
        if board.is_checkmate():
            # Subtract plies so the AI prefers faster mates
            return (-MATE_SCORE + ply_from_root) if is_maximizing else (MATE_SCORE - ply_from_root), None
        return 0.0, None # Draw/Stalemate

    # Terminal check 2: Reached depth limit -> Evaluate using your Neural Network
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
            if beta <= alpha:
                break # Prune branch
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
            if beta <= alpha:
                break # Prune branch
        return min_eval, best_move

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def evaluate_image_with_minimax(image_path, model_path="chess_model.pth", search_depth=4):
    # 1. Load Neural Network
    model = ChessCNN()
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("-> Successfully loaded Custom Neural Network!")
    except FileNotFoundError:
        print(f"Could not find {model_path}.")
        return

    # 2. Extract FEN from image
    print(f"-> Reading board from {image_path}...")
    try:
        raw_fen = predict_fen(image_path)
        clean_fen = fix_bad_fen(raw_fen)
        oriented_fen = auto_detect_orientation(clean_fen)
        print(f"-> Extracted FEN: {oriented_fen}")
    except Exception as e:
        print(f"Failed to read image: {e}")
        return

    # 3. Create Python-Chess Board
    full_fen = f"{oriented_fen} w - - 0 1"
    board = chess.Board(full_fen)

    print(f"-> Calculating {search_depth} steps ahead using Minimax + Neural Network...")
    
    # 4. Run Minimax Search
    score, best_move = minimax(board, depth=search_depth, alpha=-float('inf'), beta=float('inf'), 
                               is_maximizing=(board.turn == chess.WHITE), model=model)

    print("\n=== AI MINIMAX RESULT ===")
    
    # Check if a forced checkmate was discovered
    if abs(score) > 9000.0:
        # Calculate moves to mate from the score delta
        plies_to_mate = int(MATE_SCORE - abs(score))
        moves_to_mate = (plies_to_mate + 1) // 2
        winner = "White" if score > 0 else "Black"
        print(f"FORCED MATE DETECTED! Mate in {moves_to_mate} move(s) for {winner}.")
        if best_move:
            print(f"Best Move: {best_move.uci()}")
    else:
        leader = "White" if score > 0 else "Black"
        print(f"Position Evaluation: {leader} leads by {abs(score):.2f} pawns.")
        if best_move:
            print(f"Recommended Move: {best_move.uci()}")

if __name__ == "__main__":
    TEST_IMAGE = "5.jpeg"
    
    # search_depth=4 searches 4 half-moves ahead (2 full turns). 
    # Increase to 6 for deeper forced mate detection!
    evaluate_image_with_minimax(TEST_IMAGE, search_depth=4)
if __name__ == "__main__":
    KAGGLE_CSV_PATH = "chessData.csv"
