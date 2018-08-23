import numpy as np
import tkinter as tk

# x0, y0, x1, y1
def draw_states(canvas):
  s0 = canvas.create_rectangle(50, 225, 100, 275)  # gauche milieu
  s1 = canvas.create_rectangle(225, 100, 275, 150)  # milieu haut
  s2 = canvas.create_rectangle(225, 225, 275, 275)  # milieu milieu
  s3 = canvas.create_rectangle(225, 350, 275, 400)  # milieu bas
  sf = canvas.create_rectangle(400, 225, 450, 275)  # droite milieu
  return [s0, s1, s2, s3, sf]


def draw_states_values(canvas):
  canvas.create_text(75, 250, text='sos')
  canvas.create_text(250, 125, text='je')
  canvas.create_text(250, 250, text='vais')
  canvas.create_text(250, 375, text='bien')
  canvas.create_text(425, 250, text='eos')


def draw_actions(canvas):
  # draw action relying S0 to S1 and reverse
  canvas.create_line(85, 225, 225, 130, arrow=tk.FIRST)
  canvas.create_line(65, 225, 225, 115, arrow=tk.LAST)

  # draw action relying S0 to S2 and reverse
  canvas.create_line(100, 240, 225, 240, arrow=tk.LAST)
  canvas.create_line(100, 260, 225, 260, arrow=tk.FIRST)

  # draw action relying S2 to SF and reverse
  canvas.create_line(275, 240, 400, 240, arrow=tk.LAST)
  canvas.create_line(275, 260, 400, 260, arrow=tk.FIRST)

  # draw action relying S1 to S2 and reverse
  canvas.create_line(240, 150, 240, 225, arrow=tk.FIRST)
  canvas.create_line(260, 150, 260, 225, arrow=tk.LAST)

  # draw action relying S2 to S3 and reverse
  canvas.create_line(240, 275, 240, 350, arrow=tk.FIRST)
  canvas.create_line(260, 275, 260, 350, arrow=tk.LAST)

  # draw action relying S0 to S3 and reverse
  canvas.create_line(85, 275, 225, 365, arrow=tk.FIRST)
  canvas.create_line(65, 275, 225, 380, arrow=tk.LAST)

  # draw action relying S1 to SF and reverse
  canvas.create_line(275, 115, 435, 225, arrow=tk.LAST)
  canvas.create_line(275, 130, 415, 225, arrow=tk.FIRST)

  # draw action relying S3 to SF and reverse
  canvas.create_line(275, 365, 415, 275, arrow=tk.FIRST)
  canvas.create_line(275, 380, 435, 275, arrow=tk.LAST)

  # draw action relying S0 to SF and reverse
  # canvas.create_line(40, 260, 50, 260)
  # canvas.create_line(40, 260, 40, 410)
  # canvas.create_line(40, 410, 460, 410)
  # canvas.create_line(460, 410, 460, 260)
  # canvas.create_line(450, 260, 460, 260, arrow=tk.FIRST)
  #
  # canvas.create_line(450, 240, 460, 240)
  # canvas.create_line(460, 240, 460, 90)
  # canvas.create_line(460, 90, 40, 90)
  # canvas.create_line(40, 90, 40, 240)
  # canvas.create_line(40, 240, 50, 240, arrow=tk.LAST)


def draw_reward_struct(canvas, center, best_reward):
  canvas.create_text(center, 65, text='Q(s, a) = r + g * max( Q(s\', a) )')
  canvas.create_text(center, 375, text='Rewards')
  gr = canvas.create_rectangle(center - 95, 400, center - 45, 450)  # good reward
  nr = canvas.create_rectangle(center - 25, 400, center + 25, 450)  # neutral reward
  br = canvas.create_rectangle(center + 45, 400, center + 95, 450)  # bad reward
  canvas.create_text(center - 70, 425, text=str(best_reward))
  canvas.create_text(center, 425, text='0')
  canvas.create_text(center + 70, 425, text='-1')
  return [gr, nr, br]


def fill_mat(canvas, texts_pos, data):
  fdata = data.flatten(order='F')
  for i, t in enumerate(texts_pos):
    canvas.itemconfig(t, text=int(fdata[i]))
  canvas.update()


def draw_mat_struct(canvas, center, num_x_case, num_y_case, name, ncenter, size_case=30):
  '''

  Outputs:
    -> list of tuple (x, y) corresponding to all cases center
  '''
  canvas.create_text(ncenter[0], ncenter[1], text=name)
  x, y = center

  nyc_2, nxc_2 = num_y_case // 2, num_x_case // 2
  sc2 = size_case // 2

  pair_x_case = True if num_x_case % 2 == 0 else False
  pair_y_case = True if num_y_case % 2 == 0 else False

  y_up = y - nyc_2 * size_case if pair_y_case else y - nyc_2 * size_case - sc2
  y_down = y + nyc_2 * size_case if pair_y_case else y + nyc_2 * size_case + sc2
  all_y = list(range(y_up, y_down + 1, size_case))

  x_left = x - nxc_2 * size_case if pair_x_case else x - nxc_2 * size_case - sc2
  x_right = x + nxc_2 * size_case if pair_x_case else x + nxc_2 * size_case + sc2
  all_x = list(range(x_left, x_right + 1, size_case))

  for yy in all_y:
    canvas.create_line(x_left, yy, x_right, yy)
  for xx in all_x:
    canvas.create_line(xx, y_up, xx, y_down)

  # centers = [(xx + sc2, yy + sc2) for yy in all_y[:-1] for xx in all_x[:-1]]  # tk_qleraning.py
  centers = [(xx + sc2, yy + sc2) for xx in all_x[:-1] for yy in all_y[:-1]]

  # draw rectangle in order to fill it if we want
  rects = [canvas.create_rectangle(x - sc2, y - sc2, x + sc2, y + sc2) for x, y in centers]

  init_mat = np.zeros((num_x_case, num_y_case), dtype=np.int)
  finit_mat = init_mat.flatten()
  # texts_mat = [canvas.create_text(x, y, text=finit_mat[i]) for i, (x, y) in enumerate(centers)]  # tk_qleraning.py
  texts_mat = [canvas.create_text(x, y, text=finit_mat[i]) for i, (x, y) in enumerate(sorted(centers, key=lambda x: x[1]))]

  return texts_mat, rects, centers


def draw_labels_mat(canvas, centers, texts_x, texts_y, size_case=30):
  xs = list(set(map(lambda x: x[0], centers)))
  ys = list(set(map(lambda y: y[1], centers)))
  x1 = centers[0][0] - size_case // 2 - 15
  y1 = centers[0][1] - size_case // 2 - 10
  for i, x in enumerate(sorted(xs)):
    canvas.create_text(x, y1, text=texts_x[i])
  for i, y in enumerate(sorted(ys)):
    canvas.create_text(x1, y, text=texts_y[i])


def draw_sentence(canvas):
  return canvas.create_text(250, 450, text='Sentence: ')
