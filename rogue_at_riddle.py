import tqdm
import time
import random
import numpy as np
import tkinter as tk
import tk_qlearning_utils as tkqu


def create_tower(canvas, num_floor, canvas_width, canvas_height, floor_width, floor_height):
  pair = True if num_floor % 2 == 0 else False
  x_left = canvas_width // 2 - floor_width // 2
  x_right = canvas_width // 2 + floor_width // 2
  if pair:
    y_up = canvas_height // 2 - num_floor // 2 * floor_height
    y_down = canvas_height // 2 + num_floor // 2 * floor_height
  else:
    y_up = canvas_height // 2 - num_floor // 2 * floor_height - floor_height // 2
    y_down = canvas_height // 2 + num_floor // 2 * floor_height + floor_height // 2
  all_y = list(range(y_up, y_down, floor_height))

  recs = []
  i = num_floor - 1
  for y in all_y:
    recs.append(canvas.create_rectangle(x_left, y, x_right, y + floor_height))
    canvas.create_text(canvas_width // 2, y + floor_height // 2, text=str(i))
    i -= 1

  return recs


def update_water(canvas, floors, colors):
  for floor, color in zip(floors, colors):
    canvas.itemconfig(floor, fill=color)


def god_move(state):
  fetish = 7
  moves = [1, 3, 4]
  if state == 2:
    return 1, 'no other choice'  # we've already lost but we can only move by one

  div = state // fetish
  first_best_move = state - (div * fetish + 2)
  second_best_move = state - div * fetish
  if first_best_move == 0 or second_best_move == 0:
    return random.sample(moves, 1)[0], 'random'  # random because we've already lost
  elif first_best_move not in moves:
    return second_best_move, 'best'
  else:
    return first_best_move, 'best'


class RogueAtRiddle(object):
  def __init__(self, challenger=None, height=700, width=1250, num_floor=26, num_actions=3, num_states=25):
    self.force_best = False
    self.fix_qtable = False
    self.win, self.loose = 0, 0
    self.last_player = 'IA'
    self.end = False
    self.action_map = {1: 0, 3: 1, 4: 2}
    self.challenger = challenger
    self.text_tip = 'move: {} ({})'
    self.num_floor = 26
    self.window = tk.Tk()
    self.window.bind("<Key>", self.actions)

    self.canvas = tk.Canvas(self.window, width=width, height=height)
    self.recs = create_tower(self.canvas, 26, width // 3, height, 100, 20)

    self.initial_game_state_text = 'playing'
    self.game_state = self.canvas.create_text(width // 6, 20, text=self.initial_game_state_text)
    self.num_blue = 25

    initial_blue_states = [''] + ['blue' for _ in range(self.num_blue)]
    update_water(self.canvas, self.recs, initial_blue_states)

    move, move_type = god_move(self.num_blue)
    self.initial_tip_text = self.text_tip.format(move, move_type)
    self.tip = self.canvas.create_text(width // 6, 40, text=self.initial_tip_text)

    self.player_text = 'turn: {}'
    self.initial_player_text = self.player_text.format('you')
    self.player = self.canvas.create_text(width // 6, 60, text=self.initial_player_text)

    self.texts = {'player': self.player, 'game_state': self.game_state, 'tip': self.tip}

    # self.texts_pos_old_q, self.old_q_pos, centers = tkqu.draw_mat_struct(self.canvas, (750, 250), num_states,
    #                                                                      num_actions, 'Old Q-table', (750, 125))
    # tkqu.draw_labels_mat(self.canvas, centers, ['s{}'.format(i) for i in range(25, 0, -1)], ['a4', 'a3', 'a1'])
    self.texts_pos_new_q, self.new_q_pos, centers = tkqu.draw_mat_struct(self.canvas, (750, 350), num_states,
                                                                         num_actions, 'New Q-table', (750, 200))
    tkqu.draw_labels_mat(self.canvas, centers, ['s{}'.format(i) for i in range(25, 0, -1)], ['a1', 'a3', 'a4'])
    self.canvas.pack()

  def get_available_actions(self):
    if self.num_blue >= 4:
      return [1, 3, 4]
    elif self.num_blue == 3:
      return [1, 3]
    else:
      return [1]

  def checks(self, move):
    mouvement = True
    end = False
    if self.num_blue - move < 0:
      mouvement = False
    if self.num_blue - move == 0 or self.num_blue == 0:
      end = True
    return mouvement, end

  def god_action(self):
    move, gmove_type = god_move(self.num_blue)
    mov_ok, end = self.checks(move)
    self.num_blue -= move
    self.update_water_after_move()
    return mov_ok, end, move

  def update_water_after_move(self):
    filled_floor = ['blue' for _ in range(self.num_blue)]
    empty_floor = ['' for _ in range(self.num_floor - len(filled_floor))]
    update_water(self.canvas, self.recs, empty_floor + filled_floor)

  def update_texts(self, texts):
    for k, v in texts.items():
      if k in self.texts:
        self.canvas.itemconfig(self.texts[k], text=v)

  def end_update(self, move, god_end, end):
    # print('END UPDATE')
    self.end = True

    if god_end and not end:
      text = 'YOU LOOSE'
      reward = -1
      self.loose += 1
    else:
      text = 'YOU WIN'
      reward = 100
      self.win += 1

    self.update_texts({'tip': '', 'player': '', 'game_state': text})
    state = self.state_where_I_was
    if not self.fix_qtable:
      self.challenger.update_q_table(state, self.action_map[move], reward, None)
      self.updateQtable()

  def player_update(self, move):
    # print('PLAYER UPDATE')
    self.state_where_I_was = 25 - self.num_blue
    self.move_I_have_done = move
    self.num_blue -= move
    self.update_water_after_move()
    gmove, gmove_type = god_move(self.num_blue)
    self.update_texts({'tip': self.text_tip.format(gmove, gmove_type), 'player': self.player_text.format('IA')})
    self.last_player = 'Challenger'

  def god_ia_update(self):
    # print('GOD IA UPDATE')
    ok, god_end, move = self.god_action()
    gmove, gmove_type = god_move(self.num_blue)
    self.update_texts({'tip': self.text_tip.format(gmove, gmove_type), 'player': self.player_text.format('you')})
    self.state_after_IA_played = 25 - self.num_blue if 25 - self.num_blue < 25 else None
    if not self.fix_qtable:
      self.challenger.update_q_table(self.state_where_I_was, self.action_map[self.move_I_have_done], 0,
                                     self.state_after_IA_played)
      self.updateQtable()
    self.last_player = 'IA'
    return move, god_end

  def updateQtable(self):
    tkqu.fill_mat(self.canvas, self.texts_pos_new_q, self.challenger.Qtable)

  def actions(self, event):
    event_char = event.char
    if event_char == 'p':
      time.sleep(10)
    if event_char == 'r' or self.num_blue <= 0:
      self.end = False
      self.num_blue = 25  # restart game
      self.last_player = 'IA'
      self.update_water_after_move()
      self.update_texts({'tip': self.initial_tip_text, 'player': self.initial_player_text,
                         'game_state': self.initial_game_state_text})

    if event_char == 'a':
      if self.last_player == 'IA':
        event_char = ' '
      else:
        event_char = '\r'

    if event_char == '&':
      move = 1
    elif event_char == '"':
      move = 3
    elif event_char == "'":
      move = 4
    elif event_char == ' ':
      if self.challenger is not None:
        move = self.challenger.choose_action(self.get_available_actions, 25 - self.num_blue, force_best=self.force_best)

    if event_char in ['&', '"', "'", "\r", ' '] and not self.end:
      end = False
      god_end = False
      if event_char == '\r':
        move, god_end = self.god_ia_update()
      else:
        mov_ok, end = self.checks(move)
        if mov_ok:
          self.player_update(move)
      if end or god_end:
        self.end_update(move, god_end, end)


class Game(object):
  def __init__(self, initial_state=25):
    self.initial_state = initial_state
    self.state = initial_state

  def restart(self):
    self.state = self.initial_state

  def get_available_actions(self):
    if self.state >= 4:
      return [1, 3, 4]
    elif self.state == 3:
      return [1, 3]
    else:
      return [1]

  def game_bot_action(self):
    action, _ = god_move(self.state)
    self.state -= action

  def play(self, action):
    self.state -= action
    if self.state == 0:
      return 0, 100, True  # if you win
    self.game_bot_action()
    if self.state == 0:
      return 0, -1, True  # if bot win
    else:
      return self.state, 0, False  # if game is not finish yet


def speed_training(challenger, num_games):
  game = Game()
  state = game.state
  terminal = False
  map_action = {1: 0, 3: 1, 4: 2}
  for _ in tqdm.tqdm(range(num_games)):
    while not terminal:
      action = challenger.choose_action(game.get_available_actions, 25 - game.state)
      next_state, reward, terminal = game.play(action)

      qtable_state = 25 - state
      qtable_next_state = 25 - next_state if not terminal else None
      qtable_action = map_action[action]

      challenger.update_q_table(qtable_state, qtable_action, reward, qtable_next_state, borned=True)
      state = next_state
    terminal = False
    game.restart()
    state = game.state


class IAChallenger(object):
  def __init__(self, num_actions=3, num_states=25, epsilon=1, decay_rate=0.00001, alpha=0.5, gamma=0.9):
    self.epsilon = epsilon
    self.decay_rate = decay_rate
    self.alpha = alpha
    self.gamma = gamma
    self.Qtable = np.zeros((num_states, num_actions))
    self.action_map = {0: 1, 1: 3, 2: 4}

  def choose_action(self, env_get_action, state, force_best=False):
    action_available = env_get_action()
    if not force_best and np.random.rand() < self.epsilon:
      action = np.random.choice(action_available)
    else:
      action = self.action_map[np.argmax(self.Qtable[state])]
      if action not in action_available:
        action = np.random.choice(action_available)
    self.epsilon -= self.decay_rate if self.epsilon > 0 else 0
    return action

  def update_q_table(self, state, action, reward, next_state, borned=False):
    # print('state = {} | action = {} | reward = {} | next_state = {}'.format(state, action, reward, next_state))
    if next_state is None:
      self.Qtable[state][action] = self.Qtable[state][action] + reward
    else:
      self.Qtable[state][action] = self.Qtable[state][action] + self.alpha *\
                                 (reward + self.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state][action])
    if borned:
      if self.Qtable[state][action] < -10:
        self.Qtable[state][action] = -10
      elif self.Qtable[state][action] > 100:
        self.Qtable[state][action] = 100


class FakeEvent(object):
  def __init__(self):
    self.char = 'a'


def training(number_of_training=1000):
  speed_training(iac, number_of_training)
  rar.force_best, rar.win, rar.loose, rar.fix_qtable = True, 0, 0, True
  rar.updateQtable()

  while rar.win + rar.loose < 100:
    rar.actions(fe)
  print('After training:')
  print('Number of win = {} | loose = {}'.format(rar.win, rar.loose))


if __name__ == '__main__':
  fe = FakeEvent()

  iac = IAChallenger()
  rar = RogueAtRiddle(challenger=iac)

  training()

  rar.window.mainloop()
