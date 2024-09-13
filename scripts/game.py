import numpy as np
from .players import LearnerPlayer


def evaluate_state(state):
    state_mat = np.reshape(state, (3, 3))
    rsums = np.sum(state_mat, axis=1)
    csums = np.sum(state_mat, axis=0)
    dsums = np.array([np.sum(state[[0, 4, 8]]), np.sum(state[[2, 4, 6]])])
    if np.any(np.concatenate((rsums, csums, dsums)) == 3):
        return 1
    elif np.any(np.concatenate((rsums, csums, dsums)) == -3):
        return -1
    elif np.any(state == 0):
        return 42
    else:
        return 0


def play(pl1, pl2):
    state = np.zeros(9)
    while evaluate_state(state) == 42:
        state[pl1.move(state)] = 1
        if evaluate_state(state) == 42:
            state[pl2.move(state)] = -1
    ev = evaluate_state(state)
    if ev == 0:
        pl1.tie()
        pl2.tie()
    elif ev == 1:
        pl1.win()
        pl2.loose()
    else:
        pl1.loose()
        pl2.win()
    if isinstance(pl1, LearnerPlayer):
        pl1.update()
    if isinstance(pl2, LearnerPlayer):
        pl2.update()
    return ev
