"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O


def actions(board):
    possible = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible.add((i, j))
    return possible


def result(board, action):
    if action not in actions(board):
        raise Exception("Invalid action")
    i, j = action
    new_board = [row[:] for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    for player_turn in [X, O]:
        for row in board:
            if all(cell == player_turn for cell in row):
                return player_turn
        for j in range(3):
            if all(board[i][j] == player_turn for i in range(3)):
                return player_turn
        if all(board[i][i] == player_turn for i in range(3)):
            return player_turn
        if all(board[i][2-i] == player_turn for i in range(3)):
            return player_turn
    return None


def terminal(board):
    if winner(board) is not None:
        return True
    return all(board[i][j] != EMPTY
               for i in range(3)
               for j in range(3))


def utility(board):
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    if terminal(board):
        return None

    current = player(board)

    if current == X:
        best_val = -math.inf
        best_action = None
        for action in actions(board):
            val = min_value(result(board, action))
            if val > best_val:
                best_val = val
                best_action = action
        return best_action
    else:
        best_val = math.inf
        best_action = None
        for action in actions(board):
            val = max_value(result(board, action))
            if val < best_val:
                best_val = val
                best_action = action
        return best_action


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
