import tqdm
import numpy as np


if __name__ == '__main__':
    '''
    states = ['sos', 'mon', 'nom', 'est', 'Brian', 'eos']
    start_state = 'sos'
    end_state = 'eos'
    actions = [0, 1, 2, 3, 4, 5]
                sos  mon  nom  est  Brian  eos
                 _    _    _    _     _     _
        sos   |  -1  100   -1   -1    -1    0
        mon   |  -1   -1   100  -1    -1    0
    R = nom   |  -1   -1   -1   100   -1    0
        est   |  -1   -1   -1   -1    100   0
        Brian |  0    -1   -1   -1    -1    100
        eos   |  -1   -1   -1   -1    -1    100
    '''
    num_states = 6
    num_actions = 6
    gamma = 0.9
    decreasing_step = 0.005
    R = np.asarray([[-1, 100, -1, -1, -1, 0], [-1, -1, 100, -1, -1, 0], [-1, -1, -1, 100, -1, 0],\
                    [-1, -1, -1, -1, 100, 0], [0, -1, -1, -1, -1, 100], [-1, -1, -1, -1, -1, 100]])
    s2idx = {'sos': 0, 'mon': 1, 'nom': 2, 'est': 3, 'Brian': 4, 'eos': 5}
    a2s = {v: k for k, v in s2idx.items()}
    max_try = 100

    num_episodes = []
    for _ in tqdm.tqdm(range(1000)):
        epsilon = 1
        Q = np.zeros((num_states, num_actions))
        memory = []
        all_path = []
        num_episode = 0
        finish = False
        while finish == False:
            s = 'sos'
            episode = []
            path = [s]
            loop_pos = 0
            while s != 'eos':
                if np.random.rand() < epsilon:
                    a = np.random.randint(0, num_actions)
                    epsilon -= decreasing_step
                else:
                    a = np.argmax(Q[s2idx[s]])
                s_next = a2s[a]
                q_next_best = np.max(Q[s2idx[s_next]])  # best Q value for the next step considering all possible actions
                reward = R[s2idx[s]][a]
                Q_to_update = Q[s2idx[s]][a] + reward + gamma * q_next_best  # Bellman update
                # normalize Q to avoid overflow
                max_Q = 10 if np.max(Q) == 0 else np.max(Q)
                Q_to_update = Q_to_update / max_Q
                Q[s2idx[s]][a] = Q_to_update
                episode.append([s, a, reward, s_next])
                s = s_next
                path.append(s)
                loop_pos += 1
                if loop_pos > max_try:
                    break
            num_episode += 1
            if ['sos', 'mon', 'nom', 'est', 'Brian', 'eos'] in all_path or num_episode > max_try:
                finish = True
            if num_episode < max_try:
                memory.append(episode)
                all_path.append(path)
        num_episodes.append(num_episode)
    print('It works in average of {} episodes'.format(np.mean(num_episodes)))
