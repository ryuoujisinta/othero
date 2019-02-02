# -*- coding: utf-8 -*-
import os
import sys
import time
import gc
import ptan
import random
import argparse
import collections

from lib import game, model, mcts, game_c

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"

MCTS_SEARCHES = 50
MCTS_BATCH_SIZE = 16
REPLAY_BUFFER = 1500000
LEARNING_RATE = 0.01
BATCH_SIZE = 64

TRAIN_ROUNDS = 32
# MIN_REPLAY_TO_TRAIN = 80000
TRAIN_STEPS = 500
N_PROCESS = 15
SELF_PLAY_PERIOD = 20
N_STEPS = 14000

STEPS_BEFORE_TAU_0 = 20


#def evaluate(net1, net2, rounds, device="cpu"):
#    n1_win, n2_win = 0, 0
#    mcts_stores = [mcts.MCTS(), mcts.MCTS()]
#
#    for r_idx in range(rounds):
#        r, _ = model.play_game(mcts_stores=mcts_stores, replay_buffer=None,
#                               net1=net1, net2=net2, steps_before_tau_0=0,
#                               mcts_searches=20, mcts_batch_size=16,
#                               device=device)
#        if r < -0.5:
#            n2_win += 1
#        elif r > 0.5:
#            n1_win += 1
#    return n1_win / (n1_win + n2_win)


def self_play(tracker_queue, net, replay_queue, probs_queue, loop_count,
              device="cpu"):
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    for j in range(SELF_PLAY_PERIOD):
        i = loop_count * SELF_PLAY_PERIOD + j
        t = time.perf_counter()
        status = ""
        _, steps = model.play_game(mcts_stores, replay_queue, probs_queue,
                                   net, net,
                                   steps_before_tau_0=STEPS_BEFORE_TAU_0,
                                   mcts_searches=MCTS_SEARCHES,
                                   mcts_batch_size=MCTS_BATCH_SIZE,
                                   device=device, status=status)
        game_steps = steps
        dt = time.perf_counter() - t
        speed_steps = game_steps / dt
        status = "episode #{}, steps {:3d}, processing time {:5.2f} [s], steps/s {:5.2f}".format(i, steps, dt, speed_steps)
        tracker_queue.put(("speed_steps", speed_steps, i))
        print("episode #%d, steps %3d, steps/s %5.2f" % (
            i, game_steps, speed_steps))


def replay_buffer_size(replay_buffer):
    buffer_size = len(replay_buffer)
    size = sys.getsizeof(replay_buffer)
    if buffer_size > 0:
        for i in range(4):
            size += sys.getsizeof(replay_buffer[0][i]) * buffer_size
        size += sys.getsizeof(replay_buffer[0][3][0]) * buffer_size * 65
    return buffer_size, size / 10 ** 6


def training(tb_tracker, net, optimizer, scheduler, replay_buffer, probs_queue,
             saves_path, step, device=torch.device("cpu")):
    tmp_net = net.to(device)

#    while len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
#        time.sleep(10)
#        while not replay_queue.empty():
#            replay_buffer.append((*replay_queue.get(), probs_queue.get()))
#        print("replay buffer size: {}, {:.0f} MB".format(
#                                                    *replay_buffer_size(
#                                                            replay_buffer)))
    for i in range(TRAIN_STEPS):
        step_idx = TRAIN_STEPS * step + i + 1
#        while not replay_queue.empty():
#            if len(replay_buffer) == REPLAY_BUFFER:
#                replay_buffer.popleft()
#            replay_buffer.append((*replay_queue.get(), probs_queue.get()))
#        print("replay buffer size: {}, {:.0f} MB".format(
#                                                    *replay_buffer_size(
#                                                            replay_buffer)))
        # train
        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0
        t_train = time.time()

        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, _, batch_values, batch_probs = zip(*batch)
            batch_states_lists = [game_c.decode_binary(state) for state in batch_states]
            states_v = model.state_lists_to_batch(batch_states_lists, device)

            optimizer.zero_grad()
            probs_v = torch.FloatTensor(batch_probs).to(device)
            values_v = torch.FloatTensor(batch_values).to(device)
            out_logits_v, out_values_v = tmp_net(states_v)

            del batch
            del batch_states, batch_probs, batch_values
            del batch_states_lists
            del states_v

            loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
            # cross entropy loss
            loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
            sum_loss += loss_v.item()
            sum_value_loss += loss_value_v.item()
            sum_policy_loss += loss_policy_v.item()

            del probs_v, values_v, out_logits_v, out_values_v
            del loss_value_v, loss_policy_v, loss_v

#        scheduler.step(sum_loss / TRAIN_ROUNDS, step_idx)
        scheduler.step()
        tb_tracker.track("loss_total", sum_loss / TRAIN_ROUNDS, step_idx)
        tb_tracker.track("loss_value", sum_value_loss / TRAIN_ROUNDS, step_idx)
        tb_tracker.track("loss_policy", sum_policy_loss / TRAIN_ROUNDS, step_idx)

        print("Training step #{}: {:.2f} [s]".format(step_idx,
                                                     time.time() - t_train))
        t_train = time.time()

    # save net
    file_name = os.path.join(saves_path, "%06d.dat" % (step_idx))
    print("Model is saved as {}".format(file_name))
    torch.save(net.state_dict(), file_name)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("-m", "--model", help="The model to start from")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    net = model.Net(model.OBS_SHAPE, game.BOARD_SIZE ** 2 + 1)
    if args.model is None:
        step_idx = 0
        start = 0
    else:
#        fname = os.path.join(saves_path, args.model)
        fname = args.model
        if not os.path.exists(fname):
            print("{} does not exists!".format(fname))
            raise RuntimeError
        step_idx = int(os.path.basename(args.model)[:6])
        dir_name = os.path.dirname(args.model)
        start = step_idx / TRAIN_STEPS
        print("step_idx={}".format(step_idx))
        step_idx = step_idx - 6 * TRAIN_STEPS
    net.share_memory()

    print(net)

    track_queue = mp.Queue()
    replay_queue = mp.Queue(maxsize=REPLAY_BUFFER)
    probs_queue = mp.Queue(maxsize=REPLAY_BUFFER)
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)

    tmp_net = net.to(device)
    # add L2 regularisation (according to the original paper)
#    optimizer = optim.Adam(tmp_net.parameters(), weight_decay=1e-4)
    optimizer = optim.SGD(tmp_net.parameters(), lr=LEARNING_RATE,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=2500,
                                                     min_lr=1e-4)
#    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                               milestones=[500000, 1000000],
#                                               gamma=0.1)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=1) as tb_tracker:
        try:
            for i in range(step_idx // TRAIN_STEPS, N_STEPS):
                processes = []
                if i <= start and step_idx != 0:
                    fname = os.path.join(dir_name, "{:06d}.dat".format(i * TRAIN_STEPS))
                    net.load_state_dict(torch.load(fname,
                                                   map_location=lambda storage,
                                                   loc: storage))
                    print("{} is loaded.".format(fname))
                net = net.to(torch.device("cpu"))

                for _ in range(N_PROCESS):
                    p = mp.Process(target=self_play, args=(track_queue, net,
                                                           replay_queue,
                                                           probs_queue, i))
                    p.start()
                    processes.append(p)

                while True:
                    time.sleep(10)

                    while not track_queue.empty():
                        track_tuple = track_queue.get()
                        tb_tracker.track(*track_tuple)

                    while not replay_queue.empty():
                        if len(replay_buffer) == REPLAY_BUFFER:
                            replay_buffer.popleft()
                        replay_buffer.append((*replay_queue.get(), probs_queue.get()))
                    print("replay buffer size: {}, {:.0f} MB".format(
                                                                *replay_buffer_size(
                                                                        replay_buffer)))

                    end_processes = 0
                    for j in range(N_PROCESS):
                        if not processes[j].is_alive():
                            end_processes += 1
                    if end_processes == N_PROCESS:
                        break

                if i >= start:
                    training(tb_tracker, net, optimizer, scheduler,
                             replay_buffer, probs_queue, saves_path, i,
                             device)
        finally:
            for p in processes:
                p.join()
                p.terminate()