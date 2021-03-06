import tqdm
import time
import random
import numpy as np
import tkinter as tk
import tk_qlearning_utils as tkqu


def draw_graph(canvas):
  canvas.create_line(1250, 500, 1250, 200, arrow=tk.LAST)
  canvas.create_line(1250, 500, 1650, 500, arrow=tk.LAST)
  canvas.create_text(1250, 175, text='% of win')
  canvas.create_text(1450, 550, text='Number of games played')

  labels_y = [i * 10 for i in range(11)]
  for i, y in enumerate(np.linspace(500, 215, 11)):
    canvas.create_line(1245, y, 1255, y)
    canvas.create_text(1230, y, text=str(labels_y[i]))

  labels_x = [i * 100 for i in range(11)]
  for i, x in enumerate(np.linspace(1250, 1635, 11)):
    canvas.create_line(x, 495, x, 505)
    canvas.create_text(x, 520, text=str(labels_x[i]))


def draw_curve(canvas, pts):
  x = np.linspace(1250, 1635, 11)
  y = np.linspace(500, 215, 11)
  coords = []
  if len(pts) <= len(x):
    for i, pt in enumerate(pts):
      y_center = y[-1] + (1 - pt) * (y[0] - y[-1])
      canvas.create_line(x[i]-5, y_center, x[i]+5, y_center)
      canvas.create_line(x[i], y_center-5, x[i], y_center+5)
      coords.append((x[i], y_center))
    for i, (xx, yy) in enumerate(coords):
      if len(coords) > i + 1:
        canvas.create_line(xx, yy, coords[i+1][0], coords[i+1][1], fill='red')


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

  recs, texts = [], []
  i = num_floor - 1
  for y in all_y:
    recs.append(canvas.create_rectangle(x_left, y, x_right, y + floor_height))
    texts.append(canvas.create_text(canvas_width // 2, y + floor_height // 2, text=str(i)))
    i -= 1

  for i, t in enumerate(texts[::-1]):
    if i % 7 == 0 or (i - 2) % 7 == 0:
      canvas.itemconfig(t, fill='red')

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
  def __init__(self, challenger=None, height=700, width=1750, num_floor=26, num_actions=3, num_states=25):
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
    # self.recs = create_tower(self.canvas, 26, width // 3, height, 100, 20)
    self.recs = create_tower(self.canvas, 26, width // 5, height, 100, 20)

    draw_graph(self.canvas)

    self.initial_game_state_text = 'playing'
    self.game_state = self.canvas.create_text(width // 10, 20, text=self.initial_game_state_text)
    self.num_blue = 25

    initial_blue_states = [''] + ['blue' for _ in range(self.num_blue)]
    update_water(self.canvas, self.recs, initial_blue_states)

    move, move_type = god_move(self.num_blue)
    self.initial_tip_text = self.text_tip.format(move, move_type)
    self.tip = self.canvas.create_text(width // 10, 40, text=self.initial_tip_text)

    self.player_text = 'turn: {}'
    self.initial_player_text = self.player_text.format('you')
    self.player = self.canvas.create_text(width // 10, 60, text=self.initial_player_text)

    self.game_played = self.canvas.create_text(750, 20, text='game played: 0')

    self.texts = {'player': self.player, 'game_state': self.game_state, 'tip': self.tip, 'game_played': self.game_played}

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
      self.challenger.update_q_table(state, self.action_map[move], reward, None, borned=True)
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
                                     self.state_after_IA_played, borned=True)
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
                         'game_state': self.initial_game_state_text,
                         'game_played': 'game played: {}'.format(self.win + self.loose)})

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


def speed_game(challenger, num_games):
  game = Game()
  terminal = False
  num_win = 0
  for _ in range(num_games):
    while not terminal:
      action = challenger.choose_action(game.get_available_actions, 25 - game.state, force_best=True)
      _, reward, terminal = game.play(action)
    if reward > 0:
      num_win += 1
    terminal = False
    game.restart()
  return num_win / num_games


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


def visual_training(fe, number_of_training=1000):
  pts = [0]
  memory_game = 1
  while rar.win + rar.loose < number_of_training:
    rar.actions(fe)
    num_games = rar.win + rar.loose
    # to update curve with win/loose rate
    if num_games % 100 == 0 and num_games > 0 and num_games > memory_game:
      percent_win = speed_game(rar.challenger, 100)
      memory_game = num_games
      pts.append(percent_win)
      draw_curve(rar.canvas, pts)

  rar.force_best, rar.fix_qtable = True, True


if __name__ == '__main__':
  random.seed(4)
  np.random.seed(4)
  fe = FakeEvent()

  iac = IAChallenger()
  rar = RogueAtRiddle(challenger=iac)

  visual_training(fe, number_of_training=1000)
  # training()

  rar.window.mainloop()
