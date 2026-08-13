import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PREPARATION (FEN TO TENSORS)
# ==========================================

# Map chess characters to specific channels (0-11)
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_tensor(fen_string):
    """Converts a standard FEN string into an 8x8x12 PyTorch tensor."""
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    
    # We only care about the board layout part of the FEN (before the first space)
    board_part = fen_string.split(' ')[0]
    rows = board_part.split('/')
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for char in row:
            if char.isdigit():
                c_idx += int(char) # Skip empty squares
            else:
                if char in PIECE_TO_INDEX:
                    channel = PIECE_TO_INDEX[char]
                    tensor[channel, r_idx, c_idx] = 1.0
                c_idx += 1
    return tensor

class KaggleChessDataset(Dataset):
    def __init__(self, csv_file, max_rows):
        print(f"Loading Kaggle dataset from {csv_file} (Limited to {max_rows} rows)...")
        self.data = pd.read_csv(csv_file, nrows=max_rows)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        fen = self.data.iloc[idx]['FEN']
        eval_str = str(self.data.iloc[idx]['Evaluation'])
        
        X = fen_to_tensor(fen)
        
        # Scale evaluation to a normalized [-10.0, 10.0] pawn range
        if '#' in eval_str:
            score = 10.0 if '+' in eval_str else -10.0
        else:
            clean_str = eval_str.replace('+', '').replace('\ufeff', '')
            try:
                # Convert centipawns to pawns and clamp extreme values
                cp = float(clean_str) / 100.0
                score = max(-10.0, min(10.0, cp))
            except ValueError:
                score = 0.0 
            
        y = torch.tensor([score], dtype=torch.float32)
        return X, y

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================

class ChessCNN(nn.Module):
    """A Convolutional Neural Network designed to evaluate chess positions."""
    def __init__(self):
        super(ChessCNN, self).__init__()
        
        # 3 Convolutional layers to look for spatial patterns (like forks and pins)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers to calculate the final score
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1) # Output is exactly 1 number: the evaluation
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Prevents overfitting

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten the 3D grid into a 1D array for the dense layers
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        return x

# ==========================================
# 3. TRAINING LOOP AND GRAPHING
# ==========================================

model = ChessCNN()

# Check if a previously trained brain exists
if os.path.exists("chess_model.pth"):
    print("Loading existing trained brain from 'chess_model.pth' to continue learning...")
    model.load_state_dict(torch.load("chess_model.pth"))

def train_and_plot(csv_path, max_rows=100, epochs=10, batch_size=16):
    
    # 1. TELL PYTORCH TO USE YOUR MACBOOK'S POWER
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n---> SPEED BOOST: Training on Apple Hardware ({device}) <---")
    
    dataset = KaggleChessDataset(csv_path, max_rows=max_rows)
    
    # 2. USE ALL 8 CPU CORES TO LOAD DATA (num_workers=8)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    model = ChessCNN()
    
    # Load previous brain if it exists, so you don't lose progress!
    import os
    if os.path.exists("chess_model.pth"):
        print("-> Loading existing brain to continue learning...")
        model.load_state_dict(torch.load("chess_model.pth"))
    
    # Send the neural network to the Mac's high-speed chip
    model.to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_losses = []
    epoch_maes = []
    
    print(f"\nStarting training on {max_rows} rows for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_mae = 0.0
        
        for inputs, targets in dataloader:
            # Send the actual board data to the Mac's high-speed chip
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_mae += torch.abs(outputs - targets).mean().item()
            
        avg_loss = running_loss / len(dataloader)
        avg_mae = running_mae / len(dataloader)
        
        epoch_losses.append(avg_loss)
        epoch_maes.append(avg_mae)
        
        print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.2f} | Mean Error: {avg_mae:.2f} pawns")
        
    torch.save(model.state_dict(), "chess_model.pth")
    print("\nTraining complete! Neural Network saved to 'chess_model.pth'")
    
    # ... (Keep your exact same graphing code here using plt) ...
    print("Generating training graphs...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='b', label='Training Loss (MSE)')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Lower is better)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), epoch_maes, marker='o', color='g', label='Mean Error (Pawns)')
    plt.title('Prediction Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Average Error distance from Stockfish')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    KAGGLE_CSV_PATH = "chessData.csv"
    
    # CHANGE max_rows HERE to 50000
    train_and_plot(KAGGLE_CSV_PATH, max_rows=50000, epochs=15, batch_size=64)
