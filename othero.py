# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from glob import glob
import os.path
import numpy as np
import torch

from lib import game, model, mcts, game_c

LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)
CYAN = "#00ffff"


class app(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.make_display()
        self.make_menu(master)
        self.reset()
        self.tk_setPalette

        self.MCTS_SEARCHES = 20
        self.MCTS_BATCH_SIZE = 16
        self.n_mcts = 320

        self.model = model.Net(model.OBS_SHAPE, game.BOARD_SIZE ** 2 + 1)
        self.save_dir = "saves/mcts_800_with_gamma/"
        self.model_name = os.path.basename(sorted(glob(self.save_dir + "*.dat"))[-1])
        fname = self.save_dir + self.model_name
        self.model.load_state_dict(torch.load(fname,
                                              map_location=lambda storage,
                                              loc: storage))

        self.lock = True

    def make_display(self):
        self.frame1 = ttk.Frame(root, padding=5)
        self.grid()
        self.display_var = tk.StringVar()
        self.display_var.set("start")
        self.dispay = ttk.Label(self, textvariable=self.display_var,
                                font=LARGE_FONT)
        self.dispay.grid(column=0, row=0, columnspan=8)

        self.policy_val = tk.StringVar()
        self.policy_val.set("")
        self.dispay_policy_val = ttk.Label(self, textvariable=self.policy_val,
                                           font=LARGE_FONT)
        self.dispay_policy_val.grid(column=9, row=0, columnspan=2)

        tk.Label(self,
                 text="●",
                 font=LARGE_FONT).grid(column=9, row=4)
        tk.Label(self,
                 text="●",
                 fg="WHITE",
                 font=LARGE_FONT).grid(column=9, row=5)
        self.stone_vars = [tk.StringVar(), tk.StringVar()]
        self.stone_vars[0].set("")
        self.stone_vars[1].set("")
        ttk.Label(self,
                  textvariable=self.stone_vars[0],
                  font=LARGE_FONT).grid(column=10, row=4)
        ttk.Label(self,
                  textvariable=self.stone_vars[1],
                  font=LARGE_FONT).grid(column=10, row=5)

        self.make_board()

    def make_menu(self, master):
        menu_ROOT = tk.Menu(master)
        master.configure(menu=menu_ROOT)

        menu_file = tk.Menu(menu_ROOT)
        menu_ROOT.add_cascade(label='File', menu=menu_file, underline=0)
        menu_file.add_command(label="quit", under=0, command=master.destroy)

        menu_game = tk.Menu(menu_ROOT)
        menu_ROOT.add_cascade(label='game', menu=menu_game, underline=0)
        menu_game.add_command(label='New game', under=0,
                              command=self.choose_stone)
        menu_game.add_command(label='com level', under=0,
                              command=self.choose_com)

    def reset(self):
        self.state = game_c.INITIAL_STATE
        self.player = game.PLAYER_BLACK
        self.update_board()
        self.moves = []
        self.policy_val.set("")
        self.pass_count = 0
        self.mcts_store = mcts.MCTS()

    def update_count(self):
        field, _ = game_c.decode_binary(self.state)
        num_black = len(field[field == game.PLAYER_BLACK])
        num_white = len(field[field == game.PLAYER_WHITE])
        self.stone_vars[0].set(num_black)
        self.stone_vars[1].set(num_white)

    def make_board(self):
        self.state = game_c.INITIAL_STATE
        colors = self.__state_to_colors()
        self.buttons = []
        for i in range(game.BOARD_SIZE):
            self.buttons.append([])
            for j in range(game.BOARD_SIZE):
                button = tk.Button(
                    self,
                    bg='#00ff00',
                    fg=colors[i][j],
                    width=3,
                    command=lambda i=i, j=j: self.on_click(j, i),
                    text='●')
                button.grid(row=i+1, column=j+1)
                self.buttons[i].append(button)
        self.update_count()

    def update_board(self):
        colors = self.__state_to_colors()
        for i in range(game.BOARD_SIZE):
            for j in range(game.BOARD_SIZE):
                self.buttons[i][j].config(fg=colors[i][j])
        self.update_count()

    def __state_to_colors(self):
        state_list, _ = game_c.decode_binary(self.state)
        data = [["#00ff00"] * game.BOARD_SIZE for _ in range(game.BOARD_SIZE)]
        for col_idx in range(game.BOARD_SIZE):
            for row_idx in range(game.BOARD_SIZE):
                cell = state_list[col_idx + row_idx * game.BOARD_SIZE]
                if cell != 2:
                    data[row_idx][col_idx] = "#000000" if cell == 0 \
                                                        else "#ffffff"
        return data

    def on_click(self, i, j):
        if self.lock:
            return
        pos = game.pos_tuple_to_int((i, j))
        field, _ = game_c.decode_binary(self.state)
        if game_c.is_possible_move_f(field, self.player, pos):
            self.move_player(pos)
        else:
            for pos in game.empty_states(field):
                if game_c.is_possible_move_f(field, self.player, pos):
                    self.display_var.set("invalid action!")
                    return

            def quit_popup():
                popup.destroy()
                self.move_player(64)

            popup = tk.Toplevel()
            popup.wm_title("!")
            msg = "No possible moves."
            label = ttk.Label(popup, text=msg, font=NORM_FONT)
            label.pack(side="top", fill="x", pady=10)
            B1 = ttk.Button(popup, text="pass", command=quit_popup)
            B1.pack()

    def choose_stone(self):
        popup = tk.Toplevel()
        popup.wm_title("Choose stone")
        msg = "Choose your stone."
        label = ttk.Label(popup, text=msg, font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        def change_state():
            self.player_stone = v.get()

        def quit_popup():
            popup.destroy()
            self.play_game()

        v = tk.IntVar(popup)
        v.set(0)
        self.player_stone = 0

        radio1 = tk.Radiobutton(popup, text=u"BLACK", variable=v, value=0,
                                command=change_state)
        radio1.pack()

        radio2 = tk.Radiobutton(popup, text=u"WHITE", variable=v, value=1,
                                command=change_state)
        radio2.pack()

        B1 = ttk.Button(popup, text="OK", command=quit_popup)
        B1.pack()

    def choose_com(self):
        popup = tk.Toplevel()
        popup.wm_title("Choose com model")
        frame = tk.Frame(popup, pady=10)
        frame.pack(side="top", fill="x", pady=10)

        label1 = ttk.Label(frame, text="model", font=NORM_FONT)
        label1.grid(column=1, row=1)
        combo1 = ttk.Combobox(frame, state='readonly')
        combo1["values"] = tuple(map(os.path.basename,
                                     sorted(glob(self.save_dir + "*.dat"))))
        combo1.set(self.model_name)
        combo1.grid(column=2, row=1)

        label2 = ttk.Label(frame, text="#MCTS", font=NORM_FONT)
        label2.grid(column=1, row=2)
        combo2 = ttk.Combobox(frame, state='readonly')
        combo2["values"] = (80, 320, 800, 1600)
        combo2.set(self.n_mcts)
        combo2.grid(column=2, row=2)

        def quit_popup():
            self.model_name = combo1.get()
            fname = self.save_dir + self.model_name
            self.model.load_state_dict(torch.load(fname,
                                       map_location=lambda storage,
                                       loc: storage))
            self.n_mcts = int(combo2.get())
            if self.n_mcts == 80:
                self.MCTS_BATCH_SIZE = 8
                self.MCTS_SEARCHES = 10
            else:
                self.MCTS_BATCH_SIZE = 16
                self.MCTS_SEARCHES = self.n_mcts // 16
            popup.destroy()

        B1 = ttk.Button(popup, text="OK", command=quit_popup)
        B1.pack()

    def play_game(self):
        self.reset()
        if self.player_stone == 1:
            self.display_var.set("My turn")
            self.move_com()
        self.display_var.set("Your turn")
        self.lock = False

    def check_result(self, field):
        if self.pass_count == 2 or (field != 2).all() \
                or (field != 0).all() or (field != 1).all():
            self.lock = True
            result = game_c.calc_result_f(field, self.player_stone)
            if result == 1:
                self.display_var.set("You win!")
            elif result == -1:
                self.display_var.set("You lose!")
            else:
                self.display_var.set("Draw")

    def move_com(self):
        self.mcts_store.clear_subtrees(self.state)
        self.mcts_store.search_batch(self.MCTS_SEARCHES, self.MCTS_BATCH_SIZE,
                                     self.state, self.player, self.model)
        probs, values = self.mcts_store.get_policy_value(self.state, tau=0)
        action = np.random.choice(game.BOARD_SIZE ** 2 + 1, p=probs)
        self.value = values[action]
        self.moves.append(action)
        self.state, field = game_c.move(self.state, action)
        self.policy_val.set("{:.3f}".format(self.value))
        if action == game.BOARD_SIZE ** 2:
            self.pass_count += 1
        else:
            self.pass_count = 0
        self.update_board()
        self.display_var.set("Your turn")
        self.check_result(field)
        self.player = 1 - self.player
        self.lock = False

    def move_player(self, pos):
        self.state, field = game_c.move(self.state, pos)
        self.mcts_store.clear_subtrees(self.state)
        if pos == game.BOARD_SIZE ** 2:
            self.pass_count += 1
        else:
            self.pass_count = 0
        self.update_board()
        self.display_var.set("My turn")
        self.player = 1 - self.player
        self.lock = True
        self.check_result(field)
        self.move_com()


if __name__ == "__main__":
    root = tk.Tk()
    root.title(u"alpha othero")
    root.geometry("400x300")
#    root.configure(background=CYAN)
    app(root)
    root.mainloop()
