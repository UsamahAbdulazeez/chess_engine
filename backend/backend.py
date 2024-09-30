"""
Flask API for chess engine to predict chess moves based on FEN input.
"""
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from chess import Board
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Neural network model for chess move prediction.
class ChessModel(nn.Module):
    """
    A CNN-based neural network for chess move prediction.
    """
    def __init__(self, output_classes):  # Changed name to `output_classes`
        """
        Initializes the convolutional and fully connected layers.
        """
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, output_classes)  # Changed name to `output_classes`
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model and move mapping
model_dir = os.path.dirname(os.path.abspath(__file__))

# Load move_to_int dictionary
with open(f"{model_dir}/move_to_int.pkl", "rb") as file:
    move_to_int = pickle.load(file)

# Calculate number of classes dynamically based on move_to_int size
num_classes = len(move_to_int)

# Initialize the model
model = ChessModel(output_classes=num_classes)  # Passed `num_classes` as `output_classes`
model.load_state_dict(
    torch.load(f"{model_dir}/chess_model.pth", map_location=torch.device('cpu'))
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Converts the chess board into a matrix for model input.
def board_to_matrix(board: Board):
    """
    Converts the chess board to a 14-channel matrix representing the pieces and legal moves.
    """
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
        '.': 12  # Empty squares
    }

    matrix = np.zeros((14, 8, 8), dtype=np.int8)

    # Fill in the pieces
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(8 * i + j)
            if piece:
                plane = piece_to_plane.get(piece.symbol(), 12)  # Default to empty if not found
                matrix[plane][i][j] = 1
            else:
                matrix[12][i][j] = 1  # Mark the square as empty

    # Encode legal moves
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        to_square = move.to_square
        from_square = move.from_square
        row_to, col_to = divmod(to_square, 8)
        row_from, col_from = divmod(from_square, 8)
      
        # Mark the destination and source squares in the 13th and 14th channels
        matrix[12, row_to, col_to] = 1  # Destination square
        matrix[13, row_from, col_from] = 1  # Source square

    return matrix

# Predicts the best move for the given board position.
def predict_move(board: Board):
    """
    Predicts the next best move for a given chess board state using the neural network.
    """
    with torch.no_grad():
        x_tensor = torch.tensor(board_to_matrix(board), dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    int_to_move = {v: k for k, v in move_to_int.items()}
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]

    for move_index in sorted_indices:
        move = int_to_move.get(move_index, None)
        if move in legal_moves_uci:
            return move

    return None

# Flask route to handle POST requests for predictions
@app.route('/predict', methods=['POST'])
def get_prediction():
    """
    Flask route to get the best chess move based on FEN input.
    """
    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Invalid input. FEN string is required.'}), 400
 
    fen = data['fen']
    try:
        board = Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid FEN string.'}), 400
 
    move = predict_move(board)
    if move is None:
        return jsonify({'error': 'No valid move found.'}), 404

    return jsonify({'move': move})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
