let board = null;
let game = new Chess();
let playerColor = null;
let selectedPiece = null;

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

function getPieceImage(piece) {
    return pieceImages[piece] || '';
}

function highlightLegalMoves(square) {
    removeHighlights();
    var moves = game.moves({
        square: square,
        verbose: true
    });

    if (moves.length === 0) return;

    moves.forEach(move => {
        $(`#${move.to}`).addClass('highlight-move');
    });

    $(`#${square}`).addClass('highlight-square');
}

function removeHighlights() {
    $('.square-55d63').removeClass('highlight-square highlight-move');
}

function onDragStart(source, piece) {
    if (game.game_over()) return false;
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
    if (playerColor && piece.charAt(0) !== playerColor) return false;
    highlightLegalMoves(source);
}

function onDrop(source, target) {
    removeHighlights();
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';
    updateStatus();
    setTimeout(makeAIMove, 250);
}

function onSnapEnd() {
    board.position(game.fen());
}

function onClickSquare(square) {
    if (!selectedPiece) {
        const piece = game.get(square);
        if (piece && (playerColor === null || piece.color === playerColor)) {
            selectedPiece = square;
            highlightLegalMoves(square);
        }
    } else {
        const move = game.move({
            from: selectedPiece,
            to: square,
            promotion: 'q'
        });

        if (move) {
            removeHighlights();
            selectedPiece = null;
            board.position(game.fen());
            updateStatus();
            setTimeout(makeAIMove, 250);
        } else {
            selectedPiece = null;
            onClickSquare(square);
        }
    }
}

function initBoard() {
    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: function(piece) {
            return getPieceImage(piece);
        },
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        onSquareClick: onClickSquare
    };
    board = Chessboard('chessboard', config);
    updateStatus();
}

function makeAIMove() {
    if (game.game_over()) return;

    setTimeout(() => {
        const moves = game.moves();
        const move = moves[Math.floor(Math.random() * moves.length)];
        game.move(move);
        board.position(game.fen());
        updateStatus();
    }, 250);
}

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

function restartGame() {
    game.reset();
    board.start();
    playerColor = null;
    selectedPiece = null;
    updateStatus();
    $('#play-as').show();
}

function undoMove() {
    if (game.history().length > 1) {
        game.undo();
        game.undo();
        board.position(game.fen());
        updateStatus();
    }
}

function flipBoard() {
    board.flip();
}

function setPlayerColor(color) {
    playerColor = color;
    $('#play-as').hide();
    initBoard();
    
    // Flip the board if the user selects black
    if (color === 'b') {
        board.flip();
        setTimeout(makeAIMove, 250); // AI makes the first move
    }
}

$(document).ready(function() {
    $('#play-as').show();
});

$(window).on('resize', function() {
    if (board !== null) {
        board.resize();
    }
});
