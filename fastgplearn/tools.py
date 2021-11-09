# -*- coding: utf-8 -*-

# @Time     : 2021/11/4 13:40
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import numpy as np


class Logs:
    """
    Log the message.

    Examples:

    >>> log = Logs()
    >>> log.record("score:0.9")
    >>> log.print(log)
    >>> "score:0.9"
    """

    def __init__(self, head_msg=''):
        self.header = """
    {} Results
----------------------------""".format(head_msg)

        self.temp = ""
        self.hold = []

    def prints(self,row=True):
        if row:
            for gen, i in enumerate(self.hold):
                print(f"gen {gen + 1}: {i}")
        else:
            for gen, i in enumerate(self.hold):
                print(i)

    def print(self, head=False,row=True):
        if head:
            print(self.header)
        elif self.temp != "":
            if row:
                print(f"gen {len(self.hold)}: {self.temp}")
            else:
                print(self.temp)
            self.temp = ""

    def record(self, msg):
        self.hold.append(str(msg))
        self.temp = msg

    def records(self, msg):
        [self.hold.extend(str(msgi)) for msgi in msg]
        self.temp = str(msg[-1])

    def record_and_print(self, msg, row=False):
        self.record(msg)
        self.print(row=row)


class Hall:
    """
    Hall of Frame.

    Examples:

    >>> hall = Hall(size=50)
    >>> hall.update(inds, gen_i, score, consts)
    >>> hall[i]

    """

    def __init__(self, size=10):
        self.size = size
        self.inds = None
        self.const_gen = None
        self.x_num = 0
        self.scores = None
        if self.size == 0 or self.size is None:
            self.update = self._update
        self.last_gen = 0

    def refresh_x_num(self, x_num):
        self.x_num = x_num

    def _update(self, *args):
        pass

    def update(self, inds, gen_i, score, consts):
        """Add individual."""
        sen = int(0.05 * score.shape[0])
        sen = 1000 if sen > 1000 else sen

        index = np.argsort(score)[::-1][:sen]
        inds = inds[index]
        score = score[index]

        if self.inds is None:
            self.inds = inds
            self.scores = score
            # self.res_gen_i = np.full_like(score,int(gen_i),dtype=np.int32)
            self.const_gen = np.repeat(consts.reshape(1, -1), score.shape[0], axis=0)
            self.last_gen += 1

        else:
            assert self.last_gen - gen_i == -1

            self.inds = np.concatenate((self.inds, inds), axis=0)
            self.scores = np.concatenate((self.scores, score), axis=0)
            # self.res_gen_i = np.concatenate((self.res_gen_i, np.full_like(score, int(gen_i),dtype=np.int32)),axis=0)
            self.const_gen = np.concatenate((self.const_gen, np.repeat(consts.reshape(1, -1), index.shape[0], axis=0)),
                                            axis=0)
            self.last_gen += 1

        if self.x_num > 0:
            self.change0()

        self.sort_and_hash()

    def change0(self):
        """Change the unused constants to 0."""
        x = np.logical_and(self.x_num <= self.inds[:, 1:], self.inds[:, 1:] < 100)
        mark = np.where(~np.any(x, axis=0))
        self.const_gen[mark] = 0.0

    def sort_and_hash(self):
        """Remove the repeat result,
        (Imperfect guarantee,due to the different individuals could be with same expression)."""
        # consts_num = self.const_gen.shape[1]
        inds_num = self.inds.shape[1]

        marks = np.concatenate((self.inds, self.const_gen, self.scores.reshape(-1, 1),), axis=1)
        marks = np.unique(marks, axis=0)

        inds, const_gen, scores = marks[:, :inds_num], marks[:, inds_num:-1], marks[:, -1]

        index = np.argsort(scores)[::-1][:self.size]
        self.inds = inds[index, :].astype(np.uint8)
        self.const_gen = const_gen[index, :].astype(np.float32)
        self.scores = scores[index]

    def __reversed__(self):
        index = np.argsort(self.scores)[::-1]
        self.inds = self.inds[index, :]
        self.res_gen_i = self.res_gen_i[index]
        self.scores = self.scores[index]

    def top_n(self, n):
        """Return the top n result."""
        return self.inds[:n, :], self.res_gen_i[:n], self.scores[:n], self.const_gen[:n]

    def __getitem__(self, n):
        """Return the n ed result."""
        return self.inds[n, :], self.const_gen[n], self.scores[n], self.const_gen[n]

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self)
