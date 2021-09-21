# -*- coding: utf-8 3-*-
import numpy as np
from lib import game_c

BOARD_SIZE = 8
PLAYER_BLACK = 0
PLAYER_WHITE = 1


#def bits_to_int(bits):
#    res = 0
#    for b in bits:
#        res *= 2
#        res += b
#    return res
#
#
#def int_to_bits(num):
#    size = 2 * BOARD_SIZE ** 2 + 1
#    res = [0] * size
#    for j in reversed(range(size)):
#        res[j] = num % 2
#        num //= 2
#    return res
#
#
def pos_tuple_to_int(pos):
    assert isinstance(pos, tuple)
    assert isinstance(pos[0], int)
    assert isinstance(pos[1], int)
    assert 0 <= pos[0] < BOARD_SIZE
    assert 0 <= pos[1] < BOARD_SIZE
    return pos[0] + pos[1] * BOARD_SIZE
#
#
#def encode_list(field_list, player):
#    """
#    Encode lists representation into the binary numbers
#    :param field_list: list of BOARD_SIZE ** 2 with 0s and 1s
#    :return: integer number with encoded game state
#    """
#    assert isinstance(field_list, np.ndarray)
#    assert len(field_list) == BOARD_SIZE ** 2
#    assert player == PLAYER_BLACK or player == PLAYER_WHITE
#
#    bits = field_list
#    mask_bits = bits == 2
#    bits[mask_bits] = 0
#    bits = np.append(bits, mask_bits)
#    bits = np.append(bits, player)
#    bits = list(map(int, bits))
#
#    return bits_to_int(bits)
#    bits = bits.astype(bool)
#
#    return tuple([bits_to_int(bits), bits_to_int(mask_bits), player])


#tmp = 2 * np.ones(BOARD_SIZE ** 2).astype(int)
#tmp[3 + 4 * BOARD_SIZE] = tmp[4 + 3 * BOARD_SIZE] = PLAYER_BLACK
#tmp[3 + 3 * BOARD_SIZE] = tmp[4 + 4 * BOARD_SIZE] = PLAYER_WHITE
#INITIAL_STATE = encode_list(tmp, PLAYER_BLACK)


#def decode_binary(state_int):
#    """
#    Decode binary representation into the list view
#    :param state_ints: list of integer representing the field
#    :return: list, int
#    """
#    assert isinstance(state_int, int)
#    bits = int_to_bits(state_int)
#    res = np.array(bits[:BOARD_SIZE ** 2], dtype=np.int32)
#    mask_bits = np.array(bits[BOARD_SIZE ** 2:-1])
#    res[mask_bits == 1] = 2
#    return res, bits[-1]


def transformation(mat, i):
    if i % 2 == 0:
        return mat[::-1]
    else:
        return mat.T


def multiple_transform(mat, i, reverse=False):
    if reverse:
        for j in range(i, 8):
            mat = transformation(mat, j)
    else:
        for j in range(i):
            mat = transformation(mat, j)
    return mat


def augment_data(state, probs):
    state_list = [state]
    probs_list = [probs]
    field, player = game_c.decode_binary(state)
    field = field.reshape([BOARD_SIZE, BOARD_SIZE])
    probs_without_pass = probs[:-1]
    probs_without_pass = probs_without_pass.reshape([BOARD_SIZE, BOARD_SIZE])
    pass_plob = probs[-1]
    for i in range(7):
        field = transformation(field, i)
        probs_without_pass = transformation(probs_without_pass, i)
        state_list.append(game_c.encode_list(field.flatten(), player))
        probs_list.append(np.concatenate([probs_without_pass.flatten(),
                                          [pass_plob]]))
    return state_list, probs_list


#def is_possible_move(state, pos):
#    assert isinstance(state, int)
#    field, player = decode_binary(state)
#    return game_c.is_possible_move_f(field, player, pos)


def empty_states(field):
    return list(map(int, np.where(field == 2)[0]))


#def calc_result(state_int, player):
#    """
#    calculate winner
#    :param state_int: current state
#    :param player: player index (PLAYER_WHITE or PLAYER_BLACK)
#    :return: result of the game from player's perspective
#    """
#    field, _ = decode_binary(state_int)
#    return game_c.calc_result_f(field, player)
#
#
#def is_full(state_int):
#    field, _ = decode_binary(state_int)
#    return (field != 2).all()
#
#
#def is_one_color(state_int):
#    field, _ = decode_binary(state_int)
#    return (field != 0).all() or (field != 1).all()
#
#
#def move(state_int, pos):
#    """
#    Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
#    :param state_int: current state
#    :param pos: int position to make a move
#    :param player: player index (PLAYER_WHITE or PLAYER_BLACK)
#    :return: tuple of (state_new, won). Value won is bool, True if this move lead
#    to victory or False otherwise (but it could be a draw)
#    """
#
#    assert isinstance(state_int, int)
#    field, player = decode_binary(state_int)
#
#    return move_f(field, player, pos)
#
#
#def move_f(field, player, pos):
#    assert len(field) == BOARD_SIZE ** 2
#    assert isinstance(pos, int)
#    assert pos < BOARD_SIZE ** 2 + 1
#
#    if pos < BOARD_SIZE ** 2:
#        reverse_list = game_c.get_reverse_list_f(field, player, pos)
#        field[pos] = player
#        for pos in reverse_list:
#            field[pos] = player
#    state_new = encode_list(field.copy(), 1 - player)
#
#    return state_new, field


def render(state_ints):
    state_list, _ = game_c.decode_binary(state_ints)
    data = [[' '] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for col_idx in range(BOARD_SIZE):
        for row_idx in range(BOARD_SIZE):
            cell = state_list[col_idx + row_idx * BOARD_SIZE]
            if cell != 2:
                data[row_idx][col_idx] = str(cell)
    return [''.join(row) for row in data]


def update_counts(counts_dict, key, counts):
    # second argument is default value
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res
