# -*- coding: utf-8 -*-
import time
# import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Queue

from lib import game, mcts, game_c
import colorama
from colorama import Fore, Back, Style

OBS_SHAPE = (3, game.BOARD_SIZE, game.BOARD_SIZE)
NUM_FILTERS = 64


class Net(nn.Module):
    def __init__(self, input_shape, actions_n, n_layers=20):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU()
        )

        # layers with residual
        convs = [nn.Sequential(
                nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
                nn.BatchNorm2d(NUM_FILTERS),
                nn.ReLU(),
                nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
                nn.BatchNorm2d(NUM_FILTERS)
        ) for _ in range(n_layers)]
        self.convs = nn.ModuleList(convs)

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        conv_val_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, actions_n)
        )

    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self.conv_in(x)
        for i in range(len(self.convs)):
            v = self.convs[i](v)
            v = F.relu(v)
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))
        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))
        return pol, val


def _encode_list_state(dest_np, state_list, who_move, rotate_num):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
    """
    assert dest_np.shape == OBS_SHAPE
    field = state_list.reshape([game.BOARD_SIZE, game.BOARD_SIZE])
    if rotate_num != -1:
        field = game.multiple_transform(field, rotate_num)
    dest_np[0][field == who_move] = 1
    dest_np[1][field == (1 - who_move)] = 1
    dest_np[2] += who_move


def state_lists_to_batch(state_lists, device="cpu", rotate_list=None):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, state in enumerate(state_lists):
        who_move = state[1]
        if rotate_list is not None:
            rotate_num = rotate_list[idx]
        else:
            rotate_num = -1
        _encode_list_state(batch[idx], state[0], who_move, rotate_num)
    return torch.tensor(batch).to(device)


def play_game(mcts_stores, replay_queue, probs_queue, net1, net2,
              steps_before_tau_0, mcts_searches, mcts_batch_size,
              device="cpu", status=""):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
#    assert isinstance(replay_queue, (Queue, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game_c.INITIAL_STATE
    nets = [net1, net2]
    cur_player = game.PLAYER_BLACK
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    pass_count = 0

    while result is None:
        t = time.perf_counter()
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size,
                                             state, cur_player,
                                             nets[cur_player], device=device)
        probs, _ = mcts_stores[cur_player].get_policy_value(state, tau=tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(game.BOARD_SIZE ** 2 + 1, p=probs)
        del probs
        if not game_c.is_possible_move(state, action):
            # in this case, game.move function raise AssertionError
            print("Impossible action selected")
        state, field = game_c.move(state, action)
        mcts_stores[0].clear_subtrees(state)
        mcts_stores[1].clear_subtrees(state)
        if action == game.BOARD_SIZE ** 2:
#            print("pass: player{}, #{}".format(cur_player, step + 1))
#            render(state)
            pass_count += 1
        else:
            pass_count = 0

#        render(state)
#        print(status)
#        print("{:.2f} [s/move]".format(time.perf_counter() - t))

        if pass_count == 2 or (field != 2).all() or \
            (field != 0).all() or (field != 1).all():
            result = game_c.calc_result_f(field, cur_player)
            net1_result = result if cur_player == 0 else -result

        cur_player = 1-cur_player
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_queue is not None:
        for state, cur_player, probs in reversed(game_history):
#            replay_queue.put((state, cur_player, result))
#            probs_queue.put(probs)
            augment_data = game.augment_data(state, probs)
            for arg_state, arg_probs in zip(*augment_data):
                replay_queue.put((arg_state, cur_player, result))
                probs_queue.put(arg_probs)
            augment_data[0].clear()
            augment_data[1].clear()
            del state, cur_player, probs

            result = -0.95 * result
    game_history.clear()
    mcts_stores[0].clear()
    mcts_stores[1].clear()
    del game_history
    del nets

    return net1_result, step


def play_game1(net1, net2, steps_before_tau_0,
               mcts_searches, mcts_batch_size,
               device="cpu"):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
#    assert isinstance(replay_queue, (Queue, type(None)))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0

    mcts_stores = [mcts.MCTS(), mcts.MCTS()]

    state = game_c.INITIAL_STATE
    nets = [net1, net2]
    cur_player = game.PLAYER_BLACK
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None
    n_trees = [[], []]

    pass_count = 0
    while result is None:
        mcts_stores[cur_player].search_batch(mcts_searches[cur_player],
                                             mcts_batch_size[cur_player],
                                             state, cur_player,
                                             nets[cur_player], device=device)
        n_trees[cur_player].append(len(mcts_stores[cur_player]))
        probs, _ = mcts_stores[cur_player].get_policy_value(state, tau=tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(game.BOARD_SIZE ** 2 + 1, p=probs)
        del probs
        if not game_c.is_possible_move(state, action):
            # in this case, game.move function raise AssertionError
            print("Impossible action selected")
        state, field = game_c.move(state, action)
        mcts_stores[0].clear_subtrees(state)
        mcts_stores[1].clear_subtrees(state)
        if action == game.BOARD_SIZE ** 2:
            pass_count += 1
        else:
            pass_count = 0

        if pass_count == 2 or (field != 2).all():
            result = game_c.calc_result_f(field, cur_player)
            net1_result = result if cur_player == 0 else -result

        cur_player = 1-cur_player
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    game_history.clear()
    mcts_stores[0].clear()
    mcts_stores[1].clear()
    del game_history
    del nets
    mcts_stores.clear()
    del mcts_stores

    return net1_result, step, n_trees


def render(state):
    colorama.init()
    rend = game.render(state)
    rend = ["{}{}|{}{}".format(Style.RESET_ALL, i, Back.GREEN,
                               rend[i].replace("0", Fore.BLACK + '●').replace("1", Fore.WHITE + "●").replace(" ", "  "))
            for i in range(game.BOARD_SIZE)]
    rend = "\n".join(rend)
    board = "  0 1 2 3 4 5 6 7\n  ----------------\n"\
            + rend + Style.RESET_ALL + "\n  ----------------\n  0 1 2 3 4 5 6 7\n"
#    os.system("cls")
    print(board)
