let board = null;
let game = new Chess();
let playerColor = null;
let selectedPiece = null;
let isMobile = false;

// Detect if the user is on a mobile device
function detectMobileDevice() {
    isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0;
}

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
    const moves = game.moves({
        square: square,
        verbose: true
    });

    moves.forEach(move => {
        $(`.square[data-square='${move.to}']`).addClass('highlight-move');
    });

    $(`.square[data-square='${square}']`).addClass('highlight-source');
}

function removeHighlights() {
    $('.square').removeClass('highlight-move highlight-source');
}

function onDragStart(source, piece) {
    if (game.game_over() || 
        (game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1) ||
        (playerColor && piece.charAt(0) !== playerColor)) {
        return false;
    }

    highlightLegalMoves(source);
    return true;
}

function onDrop(source, target) {
    removeHighlights();
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateStatus();
    setTimeout(makeAIMove, 250);
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
            removeHighlights();
            onClickSquare(square);  // Retry selection
        }
    }
}

function onMouseoverSquare(square, piece) {
    if (isMobile) return; // Don't highlight on mobile devices

    const moves = game.moves({
        square: square,
        verbose: true
    });

    if (moves.length === 0) return;

    $(`.square[data-square='${square}']`).addClass('highlight-source');

    moves.forEach(move => {
        $(`.square[data-square='${move.to}']`).addClass('highlight-move');
    });
}

function onMouseoutSquare(square, piece) {
    if (isMobile) return; // Don't remove highlights on mobile devices
    removeHighlights();
}

function onSnapEnd() {
    board.position(game.fen());
}

function initBoard() {
    detectMobileDevice();

    const config = {
        draggable: true, // Enable drag-and-drop for all devices
        position: 'start',
        pieceTheme: getPieceImage,
        onDragStart: onDragStart,
        onDrop: onDrop,
        onMouseoverSquare: onMouseoverSquare,
        onMouseoutSquare: onMouseoutSquare,
        onSnapEnd: onSnapEnd,
        onSquareClick: onClickSquare // Add click handler for all devices
    };

    board = Chessboard('chessboard', config);

    updateStatus();
    customizeBoardColors();
}

// Additional functionality: AI moves and handling game status
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

function customizeBoardColors() {
    $('.square').each(function() {
        if ($(this).hasClass('white-1e1d7')) {
            $(this).css('background-color', '#004445');
        } else {
            $(this).css('background-color', '#002222');
        }
    });
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
    customizeBoardColors();
}

function setPlayerColor(color) {
    playerColor = color;
    $('#play-as').hide();
    initBoard();
    
    // Flip the board if the user selects black
    if (color === 'b') {
        board.flip();
        customizeBoardColors();
        setTimeout(makeAIMove, 250);
    }
}

$(document).ready(function() {
    detectMobileDevice();
    $('#play-as').show();
    initBoard();

    // Add touch event listeners for mobile devices
    if (isMobile) {
        $('#chessboard').on('touchstart', '.square-55d63', function(e) {
            const square = $(this).attr('data-square');
            onClickSquare(square);
        });
    }
});

$(window).on('resize', function() {
    if (board !== null) {
        board.resize();
        customizeBoardColors();
    }
});
