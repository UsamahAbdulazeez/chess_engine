const backendUrl = config.BACKEND_URL;
let jwtToken = null;

// Initialize the chessboard and game
let board = null;
let game = new Chess();
let playerColor = null;
let selectedPiece = null;
let isMobile = false;

// Detect if the user is on a mobile device
function detectMobileDevice() {
    isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0;
}

// Piece images from chess.com theme
const pieceImages = {
    'wP': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png',
    'wN': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png',
    'wB': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png',
    'wR': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png',
    'wQ': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png',
    'wK': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png',
    'bP': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png',
    'bN': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png',
    'bB': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png',
    'bR': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png',
    'bQ': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png',
    'bK': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png'
};

// Helper to get the image for a piece
function getPieceImage(piece) {
    return pieceImages[piece] || '';
}

// Highlight legal moves for a selected piece
function highlightLegalMoves(square) {
    removeHighlights();
    const moves = game.moves({ square: square, verbose: true });

    moves.forEach(move => {
        $(`.square[data-square='${move.to}']`).addClass('highlight-move');
    });

    $(`.square[data-square='${square}']`).addClass('highlight-source');
}

// Remove move highlights
function removeHighlights() {
    $('.square').removeClass('highlight-move highlight-source');
}

// Handle drag start
function onDragStart(source, piece) {
    if (game.game_over() || (game.turn() === 'w' && piece.search(/^b/) !== -1) || (game.turn() === 'b' && piece.search(/^w/) !== -1) || (playerColor && piece.charAt(0) !== playerColor)) {
        return false;
    }

    highlightLegalMoves(source);
    return true;
}

// Handle piece drop
function onDrop(source, target) {
    removeHighlights();
    const move = game.move({ from: source, to: target, promotion: 'q' });

    if (move === null) return 'snapback';

    updateStatus();
    setTimeout(makeAIMove, 250);
}

// Handle square click (for touch devices)
function onClickSquare(square) {
    if (!selectedPiece) {
        const piece = game.get(square);
        if (piece && (playerColor === null || piece.color === playerColor)) {
            selectedPiece = square;
            highlightLegalMoves(square);
        }
    } else {
        const move = game.move({ from: selectedPiece, to: square, promotion: 'q' });
        if (move) {
            removeHighlights();
            selectedPiece = null;
            board.position(game.fen());
            updateStatus();
            setTimeout(makeAIMove, 250);
        } else {
            selectedPiece = null;
            removeHighlights();
            onClickSquare(square);  // Retry selection
        }
    }
}

// Mouseover highlight for legal moves
function onMouseoverSquare(square, piece) {
    if (isMobile) return;

    const moves = game.moves({ square: square, verbose: true });
    if (moves.length === 0) return;

    $(`.square[data-square='${square}']`).addClass('highlight-source');
    moves.forEach(move => {
        $(`.square[data-square='${move.to}']`).addClass('highlight-move');
    });
}

// Mouseout: Remove highlights
function onMouseoutSquare(square, piece) {
    if (isMobile) return;
    removeHighlights();
}

// Sync the board after a move
function onSnapEnd() {
    board.position(game.fen());
}

// Initialize the chessboard
function initBoard() {
    detectMobileDevice();

    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: getPieceImage,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onMouseoverSquare: onMouseoverSquare,
        onMouseoutSquare: onMouseoutSquare,
        onSnapEnd: onSnapEnd,
        onSquareClick: onClickSquare  // Touch handler
    };

    board = Chessboard('chessboard', config);

    updateStatus();
    customizeBoardColors();
}

// Fetch JWT token from backend and store it
async function fetchJWTToken() {
    try {
        const response = await fetch(backendUrl + '/login', { method: 'POST' });
        const data = await response.json();
        jwtToken = data.access_token;
        console.log("JWT Token acquired:", jwtToken);
    } catch (error) {
        console.error("Error fetching JWT token:", error);
    }
}

// AI move logic with JWT token and UCI notation handling
function makeAIMove() {
    if (game.game_over()) return;

    const fen = game.fen();

    fetch(backendUrl + '/predict', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${jwtToken}`,  // Send JWT token
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fen: fen })
    })
    .then(response => response.json())
    .then(data => {
        const uciMove = data.move;  // Backend returns move in UCI format (e.g., "e2e4")
        if (uciMove) {
            const move = game.move({ from: uciMove.slice(0, 2), to: uciMove.slice(2, 4), promotion: 'q' });
            if (move) {
                board.position(game.fen());
                updateStatus();
            } else {
                console.error('Invalid move from backend:', uciMove);
            }
        } else {
            console.error('No valid move received from backend');
        }
    })
    .catch(error => {
        console.error('Error fetching AI move:', error);
    });
}

// Update game status
function updateStatus() {
    let status = '';
    if (game.in_checkmate()) {
        status = `Checkmate! ${game.turn() === 'w' ? 'Black' : 'White'} wins.`;
    } else if (game.in_draw()) {
        status = 'Draw!';
    } else {
        status = `${game.turn() === 'w' ? 'White' : 'Black'} to move`;
        if (game.in_check()) {
            status += ', Check!';
        }
    }
    $('#status').text(status);
}

// Customize board colors
function customizeBoardColors() {
    $('.square').each(function() {
        if ($(this).hasClass('white-1e1d7')) {
            $(this).css('background-color', '#004445');
        } else {
            $(this).css('background-color', '#002222');
        }
    });
}

// Restart the game
function restartGame() {
    game.reset();
    board.start();
    playerColor = null;
    selectedPiece = null;
    updateStatus();
    $('#play-as').show();
    $('#chessboard').hide();
    $('#status').hide();
    $('.button-container').hide();
}

// Undo the last move
function undoMove() {
    if (game.history().length > 1) {
        game.undo();
        game.undo();
        board.position(game.fen());
        updateStatus();
    }
}

// Flip the chessboard
function flipBoard() {
    board.flip();
    customizeBoardColors();
}

// Set the player's color and start the game
function setPlayerColor(color) {
    playerColor = color;
    $('#play-as').hide();
    $('#chessboard').show();
    $('#status').show();
    $('.button-container').show();
    initBoard();

    if (color === 'b') {
        board.flip();
        customizeBoardColors();
        setTimeout(makeAIMove, 250);
    }
}

// Initialize everything when the page loads
$(document).ready(function() {
    detectMobileDevice();
    fetchJWTToken();  // Fetch JWT token when the page loads
    $('#play-as').show();
    $('#chessboard').hide();
    $('#status').hide();
    $('.button-container').hide();

    // Add touch event listeners for mobile devices
    if (isMobile) {
        $('#chessboard').on('touchstart', '.square-55d63', function(e) {
            const square = $(this).attr('data-square');
            onClickSquare(square);
        });
    }
});

// Handle board resizing
$(window).on('resize', function() {
    if (board !== null) {
        board.resize();
        customizeBoardColors();
    }
});
