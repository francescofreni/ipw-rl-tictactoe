import numpy as np
from abc import ABC, abstractmethod
import random


class Player(ABC):
    """
    Abstract base class for the players.

    Attributes:
            n (int): number of games
            results (array): array of results (1: win, -1: loss, 0: draw)
            games (list): list of games, where each game is a k × 11 matrix with k the number of actions,
                and the columns being the game state (1–9), the action taken (10) and the probability with which
                this action was taken (11)
    """
    def __init__(self):
        """
        Initialize the class instance.
        """
        self.n = 0
        self.results = np.array([])
        self.games = []

    @abstractmethod
    def move(self, state):
        """
        Makes a move
        """
        raise NotImplementedError  # pragma: no cover

    def loose(self):
        """
        Updates the number of games played and add a loss to the results
        """
        self.n += 1
        self.results = np.append(self.results, -1)

    def win(self):
        """
        Updates the number of games played and add a victory to the results
        """
        self.n += 1
        self.results = np.append(self.results, 1)

    def tie(self):
        """
        Updates the number of games played and add a draw to the results
        """
        self.n += 1
        self.results = np.append(self.results, 0)


class RandomPlayer(Player):
    """
    Class for the player adopting the random strategy
    """
    def __init__(self):
        """
        Initialize the class instance.
        """
        super().__init__()

    def move(self, state):
        """
        Makes a move

        Args:
            state (array): state of the game

        Returns:
            move (int): position in the grid
        """
        move = random.choice(np.where(state == 0)[0])
        if np.sum(state == 0) > 7:
            self.games.append(np.append(state, [move, 1/np.sum(state == 0)]))
        else:
            self.games[-1] = np.vstack((self.games[-1], np.append(state, [move, 1/np.sum(state == 0)])))
        return move


class LeftPlayer(Player):
    """
    Class for the player who chooses the first available position to the left
    """
    def __init__(self):
        """
        Initialize the class instance.
        """
        super().__init__()

    def move(self, state):
        """
        Makes a move

        Args:
            state (array): state of the game

        Returns:
            move (int): position in the grid
        """
        move = np.where(state == 0)[0][0]
        if np.sum(state == 0) > 7:
            self.games.append(np.append(state, [move, 1]))
        else:
            self.games[-1] = np.vstack((self.games[-1], np.append(state, [move, 1])))
        return move


class LearnerPlayer(Player):
    """
    Class for the player who learns how to play against the opponent
    """
    def __init__(self,
                 bsize: int = 500,
                 last: int = 500,
                 minplays: int = 500,
                 eta: int = 20,
                 bsize_clear: int = 500,
                 ):
        """
        Initialize the class instance.

        Args:
            bsize (int): the update is made every bsize steps
            last (int): the number of games that are kept in the history
            minplays (int): the minimum number of games to be played before updating
            eta (int): stepsize
            bsize_clear (int): the history is cleaned every bsize_clear steps
        """
        super().__init__()
        self.strategy = {}
        self.bsize = bsize
        self.last = last
        self.minplays = minplays
        self.eta = eta
        self.bsize_clear = bsize_clear

    def move(self, state):
        """
        Makes a move

        Args:
            state (array): state of the game

        Returns:
            move (int): position in the grid
        """
        state_str = ''.join(map(str, state))
        if state_str not in self.strategy:
            prob = np.zeros(9)
            prob[state == 0] = 1/np.sum(state == 0)
            self.strategy[state_str] = np.zeros_like(prob)
            self.strategy[state_str][prob == 0] = -np.inf
            self.strategy[state_str][prob != 0] = np.log(prob[prob != 0])
        else:
            prob = np.exp(self.strategy[state_str])/np.sum(np.exp(self.strategy[state_str]))
        move = random.choices(list(range(9)), prob)[0]
        if np.sum(state == 0) > 7:
            self.games.append(np.append(state, [move, prob[move]]))
        else:
            self.games[-1] = np.vstack((self.games[-1], np.append(state, [move, prob[move]])))
        return move

    def update(self):
        stepsize = self.eta / self.n
        # compute gradient
        if self.n >= self.minplays and self.n % self.bsize == 0:
            gradient = {key: np.zeros(9) for key in self.strategy}
            for i in range(self.n):
                curr_game = self.games[i]
                w_tilde = 1
                w = 1
                for j in range(curr_game.shape[0]):
                    state = curr_game[j, :9]
                    state_str = ''.join(map(str, state))
                    move = curr_game[j, 9]
                    w_tilde *= np.exp(self.strategy[state_str][int(move)])/np.sum(np.exp(self.strategy[state_str]))
                    w *= curr_game[j, 10]
                if w_tilde > 0:
                    for j in range(curr_game.shape[0]):
                        state = curr_game[j, :9]
                        state_str = ''.join(map(str, state))
                        move = curr_game[j, 9]
                        gradient[state_str][int(move)] += (
                                self.results[i] * w_tilde / w *
                                (1 - np.exp(self.strategy[state_str][int(move)])/np.sum(np.exp(self.strategy[state_str])))
                        )
                        idxs = np.where(self.strategy[state_str] > -np.inf)[0]
                        idxs = idxs[idxs != move]
                        gradient[state_str][idxs] += (
                                self.results[i] * w_tilde / w *
                                -(np.exp(self.strategy[state_str][idxs]) / np.sum(np.exp(self.strategy[state_str])))
                        )
            # gradient step
            for k in self.strategy:
                curr = self.strategy[k] + stepsize * gradient[k]
                curr -= np.mean(curr[curr > -np.inf])
                if np.max(curr) > 20:
                    curr *= 20/np.max(curr)
                self.strategy[k] = curr
        # keep last games
        if self.n >= self.last and self.n % self.bsize_clear == 0:
            self.n = self.last
            self.results = self.results[-self.last:]
            self.games = self.games[-self.last:]
