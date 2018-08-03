import numpy as np
import tkinter as tk
import tk_qlearning_utils as u


class TwinQ(object):
  def __init__(self, width=1250, height=500, num_states=5, num_actions=5, greedy=False, reg=False,
               best_reward=10, decrease_eps=0.005, gamma=0.5):
    self.num_states = num_states
    self.num_actions = num_actions
    self.greedy = greedy
    self.reg = reg
    self.gamma = gamma
    self.epsilon = 1
    self.decrease_eps = decrease_eps
    self.s2w = {0: 'sos', 1: 'je', 2: 'vais', 3: 'bien', 4: 'eos'}
    self.master = tk.Tk()
    self.w = tk.Canvas(self.master, width=width, height=height)

    self.states = u.draw_states(self.w)
    u.draw_states_values(self.w)
    u.draw_actions(self.w)

    self.action_item = self.w.create_text(250, 65, text='action take ({}): None'.format(self.epsilon))

    self.sentence_item = u.draw_sentence(self.w)
    self.sentence_base = 'Sentence: {}'
    self.sentence = 'sos'

    self.texts_pos_old_q, self.old_q_pos, centers = u.draw_mat_struct(self.w, (750, 250), num_states,
                                                                      num_actions, 'Old Q-table', 750)
    u.draw_labels_mat(self.w, centers)

    self.texts_pos_new_q, self.new_q_pos, centers = u.draw_mat_struct(self.w, (1000, 250), num_states,
                                                                      num_actions, 'New Q-table', 1000)
    u.draw_labels_mat(self.w, centers)

    gr, nr, br = u.draw_reward_struct(self.w, 875, best_reward)
    self.rewards = {best_reward: gr, 0: nr, -1: br}

    self.old_qidx, self.old_state = 0, 0
    self.current_state, self.current_reward = 0, 0
    self.w.itemconfig(self.states[self.current_state], fill='green')
    self.w.itemconfig(self.sentence_item, text=self.sentence_base.format(self.sentence))
    self.w.itemconfig(self.rewards[self.current_reward], fill='green')

    self.master.bind("<Return>", self.update)
    self.w.pack()

    self.map_actions = {0: [1, 2, 3], 1: [0, 2, 4], 2: [0, 1, 3, 4], 3: [0, 2, 4], 4: [1, 2, 3, 0]}
    self.map_qtable = {1: 1, 2: 2, 3: 3, 10: 5, 12: 7, 14: 9, 20: 10, 21: 11, 23: 13, 24: 14,
                       30: 15, 32: 17, 34: 19, 41: 21, 42: 22, 43: 23, 40: 20, 0: 0}
    self.map_rewards = {1: best_reward, 2: -1, 3: -1, 10: -1, 12: best_reward, 14: -1, 20: -1, 21: -1,
                        23: best_reward, 24: -1, 30: -1, 32: -1, 34: best_reward, 41: 0, 42: 0, 43: 0, 40: 0, 0: 0}
    self.Q = np.zeros((num_states, num_actions))
    self.copyQ = np.zeros((num_states, num_actions))

  def update(self, event):
    # check if final state
    if self.current_state == 4:
      new_state = 0
    else:
      new_state = self.get_actions(self.current_state)

    self.update_sentence(new_state)
    self.change_state(self.current_state, new_state)
    self.update_reward(self.current_state, new_state)

    if self.current_state == 4:
      self.update_q_tables_item(new_state)
      self.old_qidx = 0
    else:
      self.update_q_tables(self.current_state, new_state)

    u.fill_mat(self.w, self.texts_pos_new_q, self.Q)
    u.fill_mat(self.w, self.texts_pos_old_q, self.copyQ)

    self.current_state = new_state
    self.w.update()
    np.copyto(self.copyQ, self.Q)

  def change_state(self, current_state, new_state):
    self.w.itemconfig(self.states[self.old_state], fill='')
    if current_state == 4:
      self.w.itemconfig(self.states[current_state], fill='')
    else:
      self.w.itemconfig(self.states[current_state], fill='red')
    self.w.itemconfig(self.states[new_state], fill='green')
    self.old_state = current_state

  def update_sentence(self, new_state):
    if self.current_state == 4:
      self.sentence = 'sos'
    else:
      self.sentence += ' {}'.format(self.s2w[new_state])
    self.w.itemconfig(self.sentence_item, text=self.sentence_base.format(self.sentence))

  def update_reward(self, current_state, new_state):
    new_reward = self.map_rewards[int(''.join([str(current_state), str(new_state)]))]
    self.w.itemconfig(self.rewards[self.current_reward], fill='')
    self.w.itemconfig(self.rewards[new_reward], fill='green')
    self.current_reward = new_reward

  def update_q_tables(self, current_state, new_state):
    q_idx = self.map_qtable[int(''.join([str(current_state), str(new_state)]))]
    self.update_q_tables_item(q_idx)
    x = q_idx // self.num_states
    y = q_idx % self.num_actions
    _, best_new_q = self.get_best_action_qvalue(new_state)
    new_q = self.Q[x, y] + self.current_reward + self.gamma * best_new_q
    if self.reg:
      max_Q = 10 if np.max(self.Q) == 0 else np.max(self.Q)
      new_q //= max_Q
    self.Q[x, y] = new_q
    self.old_qidx = q_idx

  def update_q_tables_item(self, q_idx, clean=False):
    self.w.itemconfig(self.old_q_pos[self.old_qidx], fill='')
    self.w.itemconfig(self.new_q_pos[self.old_qidx], fill='')
    self.w.itemconfig(self.old_q_pos[q_idx], fill='green')
    self.w.itemconfig(self.new_q_pos[q_idx], fill='green')

  def get_actions(self, state):
    if self.greedy:
      if np.random.rand() < self.epsilon:
        self.w.itemconfig(self.action_item, text='action take ({}): Random'.format(round(self.epsilon, 3)))
        action = np.random.choice(self.map_actions[state])
      else:
        self.w.itemconfig(self.action_item, text='action take ({}): Best'.format(round(self.epsilon, 3)))
        action, _ = self.get_best_action_qvalue(state)
      self.epsilon -= self.decrease_eps
    else:
      action = np.random.choice(self.map_actions[state])
    return action

  def get_best_action_qvalue(self, state):
    moves = [self.map_qtable[int(''.join([str(state), str(a)]))] for a in self.map_actions[state]]
    flat_old_q = self.Q.flatten()
    q = [(a, flat_old_q[m]) for a, m in zip(self.map_actions[state], moves)]
    q.sort(key=lambda x: x[1], reverse=True)
    return q[0]


if __name__ == '__main__':
  t = TwinQ(greedy=True, reg=True, decrease_eps=0.01, gamma=0.3)
  t.master.mainloop()
