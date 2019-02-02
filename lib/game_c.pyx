# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np

cdef int BOARD_SIZE = 8
cdef int PLAYER_BLACK = 0
cdef int PLAYER_WHITE = 1

ctypedef unsigned long long uint64_t

cdef uint64_t bits_to_int(bits):
    cdef uint64_t res = 0
    cdef int i
    for i in range(len(bits)):
        res *= 2
        res += <uint64_t>bits[i]
    return res

cdef int_to_bits(uint64_t num):
    cdef int size = BOARD_SIZE ** 2
    res = [0] * size
    cdef int j
    for j in reversed(range(size)):
        res[j] = num % 2
        num //= 2
    return res

cpdef encode_list(field_list, player):
    """
    Encode lists representation into the binary numbers
    :param field_list: list of BOARD_SIZE ** 2 with 0s and 1s
    :return: integer number with encoded game state
    """
    bits = field_list
    mask_bits = bits == 2
    bits[mask_bits] = 0

    return tuple([bits_to_int(bits), bits_to_int(mask_bits), player])

tmp = 2 * np.ones(BOARD_SIZE ** 2).astype(int)
tmp[3 + 4 * BOARD_SIZE] = tmp[4 + 3 * BOARD_SIZE] = PLAYER_BLACK
tmp[3 + 3 * BOARD_SIZE] = tmp[4 + 4 * BOARD_SIZE] = PLAYER_WHITE
INITIAL_STATE = encode_list(tmp, PLAYER_BLACK)

cpdef decode_binary(state_ints):
    """
    Decode binary representation into the list view
    :param state_ints: list of integer representing the field
    :return: list, int
    """
    res = np.array(int_to_bits(state_ints[0]), dtype=np.int32)
    mask_bits = np.array(int_to_bits(state_ints[1]))
    res[mask_bits == 1] = 2
    return res, state_ints[2]

cdef directions = [-BOARD_SIZE - 1, -BOARD_SIZE, -BOARD_SIZE + 1,
                   -1, 1,
                   BOARD_SIZE - 1, BOARD_SIZE, BOARD_SIZE + 1]

# hard coded for speed
cdef _get_n_moves_list(int pos):
    col = pos % BOARD_SIZE
    row = pos // BOARD_SIZE
    return [min(col, row), row, min(BOARD_SIZE - 1 - col, row),
            col, BOARD_SIZE - 1 - col,
            min(col, BOARD_SIZE - 1 - row),
            BOARD_SIZE - 1 - row,
            min(BOARD_SIZE - 1 - col, BOARD_SIZE - 1 - row)]

cpdef is_possible_move(states, int pos):
    field, player = decode_binary(states)
    return is_possible_move_f(field, player, pos)

cpdef bint is_possible_move_f(int[:] field, int player, int pos):
    # pass move
    if pos == BOARD_SIZE ** 2:
        return True

    if field[pos] != 2:
        return False

    n_moves_list = _get_n_moves_list(pos)
    cdef int i, j, cur_pos, d, n_moves
    cdef bint flag

    for j in range(len(n_moves_list)):
        d = directions[j]
        n_moves = <int>n_moves_list[j]
        flag = False
        cur_pos = pos
        for i in range(n_moves):
            cur_pos += d
            if field[cur_pos] == 1 - player:
                flag = True
            else:
                break
        if field[cur_pos] != player:
            continue
        if flag:
            return True
    return False

cpdef get_reverse_list_f(int[:] field, int player, int pos):
    # pass move
    if pos == BOARD_SIZE ** 2:
        return []

    if field[pos] != 2:
        return []

    reverse_list = []
    n_moves_list = _get_n_moves_list(pos)
    cdef int i, j, cur_pos, d, n_moves
    cdef bint flag

    for j in range(len(n_moves_list)):
        d = directions[j]
        n_moves = <int>n_moves_list[j]
        tmp_list = []

        cur_pos = pos
        for i in range(n_moves):
            cur_pos += d
            if field[cur_pos] == 1 - player:
                tmp_list.append(cur_pos)
            else:
                break
        if field[cur_pos] != player:
            continue
        reverse_list.extend(tmp_list)
    return reverse_list

cpdef int calc_result(state_ints, int player):
    """
    calculate winner
    :param state_int: current state
    :param player: player index (PLAYER_WHITE or PLAYER_BLACK)
    :return: result of the game from player's perspective
    """
    field, _ = decode_binary(state_ints)
    return calc_result_f(field, player)

cpdef int calc_result_f(np.ndarray field, int player):
    num_player = len(field[field == player])
    num_opponent = len(field[field == (1 - player)])

    if num_player > num_opponent:
        return 1
    elif num_player < num_opponent:
        return -1
    else:
        return 0

cpdef is_full(state_ints):
    field, _ = decode_binary(state_ints)
    return (field != 2).all()


cpdef is_one_color(state_ints):
    field, _ = decode_binary(state_ints)
    return (field != 0).all() or (field != 1).all()

cpdef move(state_ints, int pos):
    """
    Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
    :param state_int: current state
    :param pos: int position to make a move
    :param player: player index (PLAYER_WHITE or PLAYER_BLACK)
    :return: tuple of (state_new, won). Value won is bool, True if this move lead
    to victory or False otherwise (but it could be a draw)
    """
    field, player = decode_binary(state_ints)
    return move_f(field, player, pos)

cpdef move_f(field, int player, int pos):
    assert len(field) == BOARD_SIZE ** 2
    assert isinstance(pos, int)
    assert pos < BOARD_SIZE ** 2 + 1

    if pos < BOARD_SIZE ** 2:
        reverse_list = get_reverse_list_f(field, player, pos)
        field[pos] = player
        for pos in reverse_list:
            field[pos] = player
    state_new = encode_list(field.copy(), 1 - player)

    return state_new, field
