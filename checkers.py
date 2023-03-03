import argparse
import copy
import sys
import time
import math

cache = {}  # you can use this to implement state caching!
DEPTH_LIMIT = 8


class Piece:
    """
    Represents a piece on the checkers board.
    """

    def __init__(self, color: str, is_king: bool or False, coord_x: int,
                 coord_y: int):
        """

        :param color: b or w
        :param is_king: True if this piece is a king. False otherwise
        :param coord_x: The x-coordinate of this piece
        :param coord_y: The y-coordinate of this piece.
        """
        self.colour = color
        self.is_king = is_king
        self.x = coord_x
        self.y = coord_y

    def move_piece(self, x, y):
        """
        Moves the piece to (self.x + x, self.y + y)
        :param x: The ammount to move in the x direction.
        :param y: The ammount to move in the y direction.
        >>> pieces = read_from_file("checkers_starting_state.txt")
        >>> state = State(pieces)
        >>> state.display()
        .b.b.b.b
        b.b.b.b.
        .b.b.b.b
        ........
        ........
        r.r.r.r.
        .r.r.r.r
        r.r.r.r.
        >>> piece = state.get_piece_at_coords(1, 2)
        >>> piece.move_piece(1, 1)
        >>> state.display()
        .b.b.b.b
        b.b.b.b.
        ...b.b.b
        ..b.....
        ........
        r.r.r.r.
        .r.r.r.r
        r.r.r.r.
        """
        self.x += x
        self.y += y


class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    def __init__(self, pieces: list[Piece]):

        self.width = 8
        self.height = 8

        self.pieces = pieces

        self.curr_turn = 'r'

    def display_print(self):
        """
        >>> board = read_from_file("checkers_starting_state.txt")
        >>> state = State(board)
        >>> state.display()
        .b.b.b.b
        b.b.b.b.
        .b.b.b.b
        ........
        ........
        r.r.r.r.
        .r.r.r.r
        r.r.r.r.
        """
        for i in range(self.height):
            for k in range(self.width):
                if self.get_piece_at_coords(k, i):
                    print(self.get_piece_at_coords(k, i).colour, end='')
                else:
                    print(".", end='')
            print("")

    def display(self, file):
        """
        >>> board = read_from_file("checkers_starting_state.txt")
        >>> state = State(board)
        >>> state.display()
        .b.b.b.b
        b.b.b.b.
        .b.b.b.b
        ........
        ........
        r.r.r.r.
        .r.r.r.r
        r.r.r.r.
        """
        for i in range(self.height):
            for k in range(self.width):
                if self.get_piece_at_coords(k, i):
                    print(self.get_piece_at_coords(k, i).colour, end="", file=file)
                else:
                    print(".", end="", file=file)
            print("", file=file)

    def construct_grid(self):
        """
        >>> board = read_from_file("checkers_starting_state.txt")
        >>> state = State(board)
        >>> state.construct_grid()
        """

        result = []
        for i in range(self.height):
            row = []
            for k in range(self.width):
                if self.get_piece_at_coords(k, i):
                    row.append(self.get_piece_at_coords(k, i).colour)
                else:
                    row.append(".")
            result.append(row)
        return str(result)

    def get_piece_at_coords(self, x, y):
        """
        :param x: The x-coordinate
        :param y: The y-coordinate
        :return: the piece at (x, y)
        >>> state = State(read_from_file("checkers_starting_state.txt"))
        >>> piece = state.get_piece_at_coords(1, 0)
        >>> piece.colour == "b"
        True
        >>> piece.is_king == False
        True
        """
        for piece in self.pieces:
            if piece.x == x and piece.y == y:
                return piece
        return None

    def delete_piece(self, x, y):
        """
        Removes a piece from the board (since it has been eaten)
        :param x: The x-coordinate of the piece
        :param y: The y-coordinate of the piece
        """
        for piece in self.pieces:
            if piece.x == x and piece.y == y:
                self.pieces.remove(piece)


def black_moves(state: State, piece: Piece):
    """
    Returns the directions that a black piece could move, or an empty list
    otherwise.
    :param state: The current state
    :param piece: The piece to check
    :return:
    """
    directions = []
    # Single move to the right
    if not state.get_piece_at_coords(piece.x + 1, piece.y + 1) and \
            check_within_board(piece, 1, 1):
        directions.append([1, 1])
    # Single move to the left
    if not state.get_piece_at_coords(piece.x - 1, piece.y + 1) and \
            check_within_board(piece, -1, 1):
        directions.append([-1, 1])
    return directions


def black_jump_moves(state: State, piece: Piece):
    """
    Returns the directions of jump moves for a black piece, or an empty list
    otherwise.
    :param state: The current state
    :param piece: The piece to check
    :return:
    """
    directions = []
    # Jump move to the right
    if state.get_piece_at_coords(piece.x + 1, piece.y + 1) and \
            state.get_piece_at_coords(piece.x + 1, piece.y + 1).colour in ['r', 'R'] \
            and not state.get_piece_at_coords(piece.x + 2, piece.y + 2) and \
            check_within_board(piece, 2, 2):
        directions.append([2, 2])
    # Jump move to the left
    if state.get_piece_at_coords(piece.x - 1, piece.y + 1) and \
            state.get_piece_at_coords(piece.x - 1, piece.y + 1).colour in ['r', 'R'] \
            and not state.get_piece_at_coords(piece.x - 2, piece.y + 2) and \
            check_within_board(piece, -2, 2):
        directions.append([-2, 2])
    return directions


def black_jump_recurse(state: State, piece: Piece):
    queue = [[state, piece]]
    results = []

    while queue:
        state = queue.pop()
        directions = black_jump_moves(state[0], state[1])

        for direction in directions:
            state_copy = copy.deepcopy(state[0])
            piece_to_move = state_copy.get_piece_at_coords(state[1].x,
                                                           state[1].y)
            if direction == [2, 2]:
                state_copy.delete_piece(state[1].x + 1,
                                        state[1].y + 1)
            else:
                state_copy.delete_piece(state[1].x - 1,
                                        state[1].y + 1)
            piece_to_move.move_piece(direction[0], direction[1])
            if piece_to_move.y == 7:
                piece_to_move.colour = 'B'

            if not black_jump_moves(state_copy, piece_to_move):
                state_copy.curr_turn = get_next_turn(state_copy.curr_turn)
                results.append(state_copy)
            else:
                queue.append([state_copy, piece_to_move])

    return results


def red_moves(state: State, piece: Piece):
    """
    Returns the directions that a red piece could move, or an empty list
    otherwise.
    :param state: The current state
    :param piece: The piece to check
    :return:
    """
    directions = []
    # Single move to the right
    if not state.get_piece_at_coords(piece.x + 1, piece.y - 1) and \
            check_within_board(piece, 1, -1):
        directions.append([1, -1])
    # Single move to the left
    if not state.get_piece_at_coords(piece.x - 1, piece.y - 1) and \
            check_within_board(piece, -1, -1):
        directions.append([-1, -1])
    return directions


def red_jump_moves(state: State, piece: Piece):
    """
    Returns the directions of jump moves for a red piece, or an empty list
    otherwise.
    :param state: The current state
    :param piece: The piece to check
    :return:
    >>> state = State(read_from_file("test_jump.txt"))
    >>> red = state.get_piece_at_coords(6,6)
    >>> red_jump_moves(state, red)
    """
    directions = []
    # Jump move to the right
    if state.get_piece_at_coords(piece.x + 1, piece.y - 1) and \
            state.get_piece_at_coords(piece.x + 1, piece.y - 1).colour in ['b', 'B'] \
            and not state.get_piece_at_coords(piece.x + 2, piece.y - 2) and \
            check_within_board(piece, 2, -2):
        directions.append([2, -2])
    # Jump move to the left
    if state.get_piece_at_coords(piece.x - 1, piece.y - 1) and \
            state.get_piece_at_coords(piece.x - 1, piece.y - 1).colour in ['b', 'B'] \
            and not state.get_piece_at_coords(piece.x - 2, piece.y - 2) and \
            check_within_board(piece, -2, -2):
        directions.append([-2, -2])
    return directions


def red_jump_recurse(state: State, piece: Piece):
    queue = [[state, piece]]
    results = []

    while queue:
        state = queue.pop()
        directions = red_jump_moves(state[0], state[1])

        for direction in directions:
            state_copy = copy.deepcopy(state[0])
            piece_to_move = state_copy.get_piece_at_coords(state[1].x,
                                                           state[1].y)
            if direction == [2, -2]:
                state_copy.delete_piece(state[1].x + 1,
                                        state[1].y - 1)
            else:
                state_copy.delete_piece(state[1].x - 1,
                                        state[1].y - 1)
            piece_to_move.move_piece(direction[0], direction[1])
            if piece_to_move.y == 0:
                piece_to_move.colour = 'R'

            if not black_jump_moves(state_copy, piece_to_move):
                state_copy.curr_turn = get_next_turn(state_copy.curr_turn)
                results.append(state_copy)
            else:
                queue.append([state_copy, piece_to_move])

    return results


def king_moves(state: State, piece: Piece):
    directions = []
    # Single move up-right
    if not state.get_piece_at_coords(piece.x + 1, piece.y + 1) and \
            check_within_board(piece, 1, 1):
        directions.append([1, 1])
    # Single move up-left
    if not state.get_piece_at_coords(piece.x - 1, piece.y + 1) and \
            check_within_board(piece, -1, 1):
        directions.append([-1, 1])
    # Single move down-right
    if not state.get_piece_at_coords(piece.x + 1, piece.y - 1) and \
            check_within_board(piece, 1, -1):
        directions.append([1, -1])
    # Single move down-left
    if not state.get_piece_at_coords(piece.x - 1, piece.y - 1) and \
            check_within_board(piece, -1, -1):
        directions.append([-1, -1])
    return directions


def king_jump_moves(state: State, piece: Piece):
    directions = []

    # Jump move up-right
    if state.get_piece_at_coords(piece.x + 1, piece.y - 1) and \
            state.get_piece_at_coords(piece.x + 1,
                                      piece.y - 1).colour in get_opp_char(piece.colour) \
            and not state.get_piece_at_coords(piece.x + 2, piece.y - 2) and \
            check_within_board(piece, 2, -2):
        directions.append([2, -2])
    # Jump move up-left
    if state.get_piece_at_coords(piece.x - 1, piece.y - 1) and \
            state.get_piece_at_coords(piece.x - 1,
                                      piece.y - 1).colour in get_opp_char(piece.colour) \
            and not state.get_piece_at_coords(piece.x - 2, piece.y - 2) and \
            check_within_board(piece, -2, -2):
        directions.append([-2, -2])
    # Jump move down-right
    if state.get_piece_at_coords(piece.x + 1, piece.y + 1) and \
            state.get_piece_at_coords(piece.x + 1,
                                      piece.y + 1).colour in get_opp_char(piece.colour) \
            and not state.get_piece_at_coords(piece.x + 2, piece.y + 2) and \
            check_within_board(piece, 2, 2):
        directions.append([2, 2])
    # Jump move down-left
    if state.get_piece_at_coords(piece.x - 1, piece.y + 1) and \
            state.get_piece_at_coords(piece.x - 1,
                                      piece.y + 1).colour in get_opp_char(piece.colour) \
            and not state.get_piece_at_coords(piece.x - 2, piece.y + 2) and \
            check_within_board(piece, -2, 2):
        directions.append([-2, 2])
    return directions


def king_jump_recurse(state: State, piece: Piece):
    queue = [[state, piece]]
    results = []

    while queue:
        state = queue.pop()
        directions = king_jump_moves(state[0], state[1])

        for direction in directions:
            state_copy = copy.deepcopy(state[0])
            piece_to_move = state_copy.get_piece_at_coords(state[1].x,
                                                           state[1].y)
            # Up-right
            if direction == [2, -2]:
                state_copy.delete_piece(state[1].x + 1,
                                        state[1].y - 1)
            # Up-left
            elif direction == [-2, -2]:
                state_copy.delete_piece(state[1].x - 1,
                                        state[1].y - 1)
            # Down-right
            elif direction == [2, 2]:
                state_copy.delete_piece(state[1].x + 1,
                                        state[1].y + 1)
            # Down-left
            else:
                state_copy.delete_piece(state[1].x - 1,
                                        state[1].y + 1)
            piece_to_move.move_piece(direction[0], direction[1])

            if not king_jump_moves(state_copy, piece_to_move):
                state_copy.curr_turn = get_next_turn(state_copy.curr_turn)
                results.append(state_copy)
            else:
                queue.append([state_copy, piece_to_move])
    return results


def get_successors(state: State) -> list[State]:
    """
    Finds all the successors to the current state.
    >>> state = State(read_from_file("test2.txt"))
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    >>> state = alpha_beta_search(state)
    >>> state.display_print()
    """
    result = []
    jump_moves = []
    moves = []

    for piece in state.pieces:
        if piece.colour == 'b' and state.curr_turn == 'b':
            if black_jump_moves(state, piece):
                jump_moves.extend(black_jump_recurse(state, piece))

            if black_moves(state, piece):
                directions = black_moves(state, piece)

                for direction in directions:
                    board_copy = copy.deepcopy(state)
                    piece_to_move = board_copy.get_piece_at_coords(piece.x,
                                                                   piece.y)
                    piece_to_move.move_piece(direction[0], direction[1])
                    if piece_to_move.y == 7:
                        piece_to_move.colour = "B"
                    board_copy.curr_turn = get_next_turn(board_copy.curr_turn)

                    moves.append(board_copy)
        if piece.colour == 'B' and state.curr_turn == 'b':
            if king_moves(state, piece):
                directions = king_moves(state, piece)

                for direction in directions:
                    board_copy = copy.deepcopy(state)
                    piece_to_move = board_copy.get_piece_at_coords(piece.x,
                                                                   piece.y)
                    piece_to_move.move_piece(direction[0], direction[1])
                    board_copy.curr_turn = get_next_turn(board_copy.curr_turn)

                    moves.append(board_copy)

            if king_jump_moves(state, piece):
                jump_moves.extend(king_jump_recurse(state, piece))

        if piece.colour == 'r' and state.curr_turn == 'r':
            if red_moves(state, piece):
                directions = red_moves(state, piece)

                for direction in directions:
                    board_copy = copy.deepcopy(state)
                    piece_to_move = board_copy.get_piece_at_coords(piece.x,
                                                                   piece.y)
                    piece_to_move.move_piece(direction[0], direction[1])
                    if piece_to_move.y == 0:
                        piece_to_move.colour = "R"

                    board_copy.curr_turn = get_next_turn(board_copy.curr_turn)

                    moves.append(board_copy)
            if red_jump_moves(state, piece):
                jump_moves.extend(red_jump_recurse(state, piece))

        if piece.colour == 'R' and state.curr_turn == 'r':
            if king_moves(state, piece):
                directions = king_moves(state, piece)

                for direction in directions:
                    board_copy = copy.deepcopy(state)
                    piece_to_move = board_copy.get_piece_at_coords(piece.x,
                                                                   piece.y)
                    piece_to_move.move_piece(direction[0], direction[1])
                    board_copy.curr_turn = get_next_turn(board_copy.curr_turn)

                    moves.append(board_copy)
            if king_jump_moves(state, piece):
                jump_moves.extend(king_jump_recurse(state, piece))
    if jump_moves:
        result.extend(jump_moves)
    else:
        result.extend(moves)
    return result


def check_within_board(piece: Piece, x: int, y: int) -> bool:
    """
    Return false if a move would go off of the board.

    True if the move is on the board
    :param piece: The piece in mind
    :param x: The x-component of the move
    :param y: The y-component of the move
    >>> state = State(read_from_file("checkers_starting_state.txt"))
    >>> piece = state.get_piece_at_coords(0, 1)
    >>> check_within_board(piece, -1, 1)
    False
    """
    if piece.x + x > 7 or piece.x + x < 0:
        return False
    if piece.y + y > 7 or piece.y + y < 0:
        return False
    return True


def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']


def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'


def is_terminal_state(state: State):
    """
    Return true if this state is a terminal state, false otherwise
    :param state:
    :return:
    >>> pieces = read_from_file("checkers_starting_state.txt")
    >>> state = State(pieces)
    >>> is_terminal_state(state)
    False
    """
    red = 0
    black = 0

    for piece in state.pieces:
        if piece.colour in ['r', 'R']:
            red += 1
        else:
            black += 1
    if red and not black:
        return True
    elif black and not red:
        return True
    return False


def evaluation(state: State):
    """
    Estimates the utility of the current state
    :param state:
    :return:
    """
    # Check if the state is a terminal state
    if is_terminal_state(state):
        red = 0
        black = 0

        for piece in state.pieces:
            if piece.colour in ['r', 'R']:
                red += 1
            else:
                black += 1
        if state.curr_turn == 'r' and red and not black:
            return math.inf
        elif state.curr_turn == 'r' and black and not red:
            return -math.inf
        elif state.curr_turn == 'b' and black and not red:
            return math.inf
        elif state.curr_turn == 'b' and red and not black:
            return -math.inf


        # elif state.curr_turn == 'r' and black and not red:
        #     return -math.inf
        # elif state.curr_turn == 'b' and red and not black:
        #     return -math.inf

    # Else, we know this state is not a terminal state, we provide an
    # estimate of its utility.
    else:

        total = 0

        for piece in state.pieces:
            if piece.colour == 'r':
                total += (1 * compute_piece_usefulness(piece))
            elif piece.colour == "R":
                total += (2 * compute_piece_usefulness(piece))
            elif piece.colour == 'b':
                total -= (1 * compute_piece_usefulness(piece))
            elif piece.colour == 'B':
                total -= (2 * compute_piece_usefulness(piece))
        return total


def compute_piece_usefulness(piece: Piece) -> float:
    """
    Compute the usefullness (position wise) of the piece on the board.
    :param state: A state
    :param piece: The piece in question
    :return: an int between 0 and 1
    """
    total = 0.25

    # If we have a red piece, it's trying to move upwards. Value moves that move red pieces towards the top of the
    # board.
    if piece.colour == 'r':
        total += (-0.05 * piece.y) + 0.35
    # If we have a black piece, it's trying to move downwards. Value moves that move black pieces towards the bottom
    # of the board.
    if piece.colour == 'b':
        total += (0.05 * piece.y)
    # Pieces on the edge of the board are generally stronger. Value moves that move pieces towards the sides.
    if piece.x == 0 or piece.x == 7:
        total += 0.15
    return total


def read_from_file(filename):
    """
    Creates a state from a text file contianing the pieces.

    :param filename: the state
    :return: a list of pieces
    >>> pieces = read_from_file("checkers_starting_state.txt")
    >>> state = State(pieces)
    >>> state.display()
    """
    f = open(filename)
    lines = f.readlines()

    pieces = []
    k = 0

    for line in lines:
        i = 0
        for x in line.rstrip():
            if str(x) == "b":
                pieces.append(Piece("b", False, i, k))
            elif str(x) == "B":
                pieces.append(Piece("B", True, i, k))
            elif str(x) == "r":
                pieces.append(Piece("r", False, i, k))
            elif str(x) == "R":
                pieces.append(Piece("R", True, i, k))
            i += 1
        k += 1

    f.close()

    return pieces


def alpha_beta_search(state: State):
    result = alphabeta_max(state, -math.inf, math.inf, 0)
    return result[1]


def alphabeta_max(state: State, alpha: float, beta: float, depth: int):
    # if state.construct_grid() in cache.keys() and cache[state.construct_grid()][2] <= depth:
    #     return cache[state.construct_grid()][0], cache[state.construct_grid()][1]
    if depth == DEPTH_LIMIT:
        return [evaluation(state), None]

    best = None
    v = -math.inf
    for successor in get_successors(state):
        tempval, tempstate = alphabeta_min(successor, alpha, beta, depth + 1)
        if tempval >= v:
            v = tempval
            best = successor
        if tempval > beta:
            # cache[state.construct_grid()] = [v, successor, depth]
            return [v, successor]
        alpha = max(alpha, tempval)
    return [v, best]


def alphabeta_min(state: State, alpha: float, beta: float, depth: int):
    # if state.construct_grid() in cache.keys() and cache[state.construct_grid()][2] <= depth:
    #     return cache[state.construct_grid()][0], None
    if depth == DEPTH_LIMIT:
        return [evaluation(state), None]

    best = None


    v = math.inf
    for successor in get_successors(state):
        tempval, tempstate = alphabeta_max(successor, alpha, beta, depth + 1)
        if tempval <= v:
            v = tempval
            best = successor
        if tempval < alpha:
            # cache[state.construct_grid()] = [v, successor, depth]
            return [v, successor]
        beta = min(beta, tempval)
    return [v, best]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board)
    ctr = 0

    file = open(args.outputfile, 'w')
    print(state.display(file))
    print("", file=file)
    while not is_terminal_state(state) and get_successors(state):
        state = alpha_beta_search(state)
        print(state.display(file))
        print("", file=file)
    file.close()

    # while not is_terminal_state(state) and get_successors(state):
    #     state = alpha_beta_search(state)
    #     state.display()
    #     print()
