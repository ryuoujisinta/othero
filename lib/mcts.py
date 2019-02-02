# -*- coding: utf-8 -*-
"""
Monte-Carlo Tree Search
"""
import time
import numpy as np

from lib import game, model, game_c

import torch.nn.functional as F
import torch.multiprocessing as mp


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}
        self.for_time = 0
        self.subtrees = []

        self.visited_net_results = {}


    def clear_subtrees(self, state):
        retain_list = [state]
        new_subtrees = []
        for subtree in self.subtrees:
            if len(subtree) > 1:
                new_subtrees.append(subtree[1:])
            else:
                subtree.clear()
                del subtree
        self.subtrees.clear()
        self.subtrees = new_subtrees.copy()
        new_subtrees.clear()

        for subtree in self.subtrees:
            if subtree[0] == state:
                new_subtrees.append(subtree)
                retain_list.extend(subtree[1:])
            else:
                subtree.clear()
                del subtree

        retain_list = list(set(retain_list))
        self.subtrees.clear()
        self.subtrees = new_subtrees

        state_list = list(self.visit_count.keys()).copy()
        for s in state_list:
            if s not in retain_list:
                self.visit_count.pop(s)
                self.value.pop(s)
                self.value_avg.pop(s)
                self.probs.pop(s)

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()
        [subtree.clear() for subtree in self.subtrees]
        self.subtrees.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state_int, player, root_mask):
        """
        Traverse the tree until the end of game or leaf node
        :param state_int: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state_int of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None
        pass_count = 0

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = np.sqrt(counts.sum())
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * (game.BOARD_SIZE ** 2 + 1))
                probs = 0.75 * probs + 0.25 * noises
                score = values_avg + self.c_puct * probs * total_sqrt / (1 + counts)
                score += root_mask
            else:
                # select moves that maximise an upper confident bound
                score = values_avg + self.c_puct * probs * total_sqrt / (1 + counts)
            # suppress pass move
            score[-1] = -10.
            cur_field, _ = game_c.decode_binary(cur_state)
            score[:-1][cur_field != 2] = -np.inf
#            empty_states = game_c.empty_states(cur_field)
#
#            for action in empty_states:
#                if not game_c.is_possible_move_f(cur_field, cur_player, action):
#                    score[action] = -np.inf
            while True:
                action = score.argmax()
                if game_c.is_possible_move_f(cur_field, cur_player, action):
                    break
                score[action] = -np.inf

#            action = score.argmax()

            actions.append(action)
            cur_state, cur_field = game_c.move_f(cur_field, cur_player, action)
            if action == game.BOARD_SIZE ** 2:
                pass_count += 1
            else:
                pass_count = 0

            cur_player = 1-cur_player
            if pass_count == 2 or (cur_field != 2).all():
                value = game_c.calc_result_f(cur_field, cur_player)
                break

        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_ints):
        return state_ints not in self.probs

    def search_batch_test(self, count, batch_size, state_int, player, net, device="cpu"):
        cur_field, _ = game_c.decode_binary(state_int)
        root_mask = np.zeros(game.BOARD_SIZE ** 2 + 1)
        root_mask[:-1][cur_field != 2] = -np.inf
        empty_states = game.empty_states(cur_field)
        for action in empty_states:
            if not game_c.is_possible_move_f(cur_field, player, action):
                root_mask[action] = -np.inf

        for _ in range(count):
            backup_queue = []
            expand_queue = []
            planned = set()
            for i in range(count):
                value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player, root_mask)
                self.subtrees.append(states)

                # end of the game
                if value is not None:
                    backup_queue.append((value, states, actions))
                # encounter leaf node which is not end of the game
                else:
                    # avoid duplication of leaf state
                    if leaf_state not in planned:
                        planned.add(leaf_state)
                        expand_queue.append((leaf_state, states, actions))
                    else:
                        states.clear()
                        self.subtrees.pop()
            del planned

            # do expansion of nodes
            if expand_queue:
                expand_states = []
                keys = self.visited_net_results.keys()
                new_expand_queue = []
                existed_expand_queue = []
                value_list = []
                prob_list = []
                rotate_list = []
                new_rotate_list = []
                for leaf_state, states, actions in expand_queue:
                    rotate_num = np.random.randint(8)
                    if (leaf_state, rotate_num) in keys:
                        existed_expand_queue.append((leaf_state, states, actions))
                        rotate_list.append(rotate_num)
                        value, prob = self.visited_net_results[(leaf_state, rotate_num)]
                        value_list.append(value)
                        prob_list.append(prob)
                    else:
                        new_expand_queue.append((leaf_state, states, actions))
                        new_rotate_list.append(rotate_num)
                        leaf_state_lists = game_c.decode_binary(leaf_state)
                        expand_states.append(leaf_state_lists)
                expand_queue = [*existed_expand_queue, *new_expand_queue]
                rotate_list.extend(new_rotate_list)


                if len(new_rotate_list) == 0:
                    values = value_list
                    probs = prob_list
                else:
                    batch_v = model.state_lists_to_batch(expand_states,
                                                         device,
                                                         new_rotate_list)
                    logits_v, values_v = net(batch_v)
                    probs_v = F.softmax(logits_v, dim=1)
                    values = values_v.data.cpu().numpy()[:, 0]
                    probs = probs_v.data.cpu().numpy()

                    values = [*value_list, *list(values)]
                    probs = [*prob_list, *list(probs)]

                expand_states.clear()

                # create the nodes
                for (leaf_state, states, actions), value, prob, rotate_num in zip(expand_queue, values, probs, rotate_list):
                    self.visit_count[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.int32)
                    self.value[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.float32)
                    self.value_avg[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.float32)
                    prob_without_pass = prob[:-1].reshape([game.BOARD_SIZE, game.BOARD_SIZE])
                    prob_without_pass = game.multiple_transform(prob_without_pass, rotate_num, True)
                    self.probs[leaf_state] = np.concatenate([prob_without_pass.flatten(), [prob[-1]]])
                    backup_queue.append((value, states, actions))
                    self.visited_net_results[(leaf_state, rotate_num)] = (value, prob)
                rotate_list.clear()

            # perform backup of the searches
            for value, states, actions in backup_queue:
                # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
                cur_value = -value
                for state_int, action in zip(states[::-1], actions[::-1]):
                    self.visit_count[state_int][action] += 1
                    self.value[state_int][action] += cur_value
                    self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]
                    cur_value = -cur_value
                actions.clear()
            backup_queue.clear()

    def search_batch(self, count, batch_size, state_int, player, net,
                     device="cpu"):
        cur_field, _ = game_c.decode_binary(state_int)
        root_mask = np.zeros(game.BOARD_SIZE ** 2 + 1)
        root_mask[:-1][cur_field != 2] = -np.inf
        empty_states = game.empty_states(cur_field)
        for action in empty_states:
            if not game_c.is_possible_move_f(cur_field, player, action):
                root_mask[action] = -np.inf

        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net,
                                  root_mask, device)

    def search_minibatch(self, count, state_int, player, net, root_mask,
                         device="cpu"):
        """
        Perform several MCTS searches.
        """
        backup_queue = []
        expand_queue = []
        planned = set()
        for i in range(count):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player, root_mask)
            self.subtrees.append(states)

            # end of the game
            if value is not None:
                backup_queue.append((value, states, actions))
            # encounter leaf node which is not end of the game
            else:
                # avoid duplication of leaf state
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_queue.append((leaf_state, states, actions))
                else:
                    states.clear()
                    self.subtrees.pop()
        del planned

        # do expansion of nodes
        if expand_queue:
            expand_states = []
            keys = self.visited_net_results.keys()
            new_expand_queue = []
            existed_expand_queue = []
            value_list = []
            prob_list = []
            rotate_list = []
            new_rotate_list = []
            for leaf_state, states, actions in expand_queue:
                rotate_num = np.random.randint(8)
                if (leaf_state, rotate_num) in keys:
                    existed_expand_queue.append((leaf_state, states, actions))
                    rotate_list.append(rotate_num)
                    value, prob = self.visited_net_results[(leaf_state, rotate_num)]
                    value_list.append(value)
                    prob_list.append(prob)
                else:
                    new_expand_queue.append((leaf_state, states, actions))
                    new_rotate_list.append(rotate_num)
                    leaf_state_lists = game_c.decode_binary(leaf_state)
                    expand_states.append(leaf_state_lists)
            expand_queue = [*existed_expand_queue, *new_expand_queue]
            rotate_list.extend(new_rotate_list)

            if len(new_rotate_list) == 0:
                values = value_list
                probs = prob_list
            else:
                batch_v = model.state_lists_to_batch(expand_states,
                                                     device,
                                                     new_rotate_list)
                logits_v, values_v = net(batch_v)
                probs_v = F.softmax(logits_v, dim=1)
                values = values_v.data.cpu().numpy()[:, 0]
                probs = probs_v.data.cpu().numpy()

                values = [*value_list, *list(values)]
                probs = [*prob_list, *list(probs)]

            expand_states.clear()

            # create the nodes
            for (leaf_state, states, actions), value, prob, rotate_num in zip(expand_queue, values, probs, rotate_list):
#            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.int32)
                self.value[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.float32)
                self.value_avg[leaf_state] = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.float32)
                prob_without_pass = prob[:-1].reshape([game.BOARD_SIZE, game.BOARD_SIZE])
                prob_without_pass = game.multiple_transform(prob_without_pass, rotate_num, True)
                self.probs[leaf_state] = np.concatenate([prob_without_pass.flatten(), [prob[-1]]])
#                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))
                self.visited_net_results[(leaf_state, rotate_num)] = (value, prob)
            rotate_list.clear()

        expand_queue.clear()

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]
                cur_value = -cur_value
            actions.clear()
        backup_queue.clear()

    def get_policy_value(self, state_int, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = np.zeros(game.BOARD_SIZE ** 2 + 1, dtype=np.float32)
            probs[np.argmax(counts)] = 1.0
        else:
            counts = counts ** (1.0 / tau)
            total = counts.sum()
            probs = counts / total
        values = self.value_avg[state_int]
        return probs, values
