# pylint: disable=not-an-iterable
import json
from copy import deepcopy

import numpy as np

from behavioral import get_freepaths, behav_cost

arr = np.array


class NumpyEncoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class State:
    def __init__(self, width, depth, pos=None, vel=None,
                 static_obs=None, speed_lim=None,
                 same_speed_across=True, obstacle_pct=0.2, assure_open_path=False):
        self.width = width
        self.depth = depth
        self.obstacle_pct = obstacle_pct
        self.assure_open_path = assure_open_path
        self.same_speed_across = same_speed_across
        self.pos = pos if pos is not None \
            else np.random.randint(width, dtype=np.int8)
        self.vel = vel if vel is not None \
            else (np.random.randint(1, 4, dtype=np.int8))
        self.static_obs = static_obs if static_obs is not None \
            else self._gen_static(depth)
        self.speed_lim = speed_lim if speed_lim is not None \
            else self._gen_speed(depth)

    def _gen_static(self, dist):
        def mk_static(): return np.random.binomial(
            1, self.obstacle_pct, size=(dist, self.width))

        if not self.assure_open_path or self.pos >= self.width or self.pos < 0:
            return mk_static()
        while True:
            cp = deepcopy(self)
            proposal = mk_static()
            cp.static_obs = proposal
            if len(get_freepaths(cp)) > 0:
                return proposal

    def _gen_speed(self, dist):
        if self.same_speed_across:
            speeds = np.clip(np.random.normal(3, 2, size=dist), 1, 5)
            return np.tile(speeds, (self.width, 1)).T
        return np.clip(np.random.normal(3, 2, size=(dist, self.width)), 1, 5)

    def step(self, dist):
        """generate the next environment from random simulation after moving some distance/time"""
        self.static_obs = np.concatenate(
            [self.static_obs[dist:], self._gen_static(dist)])
        self.speed_lim = np.concatenate(
            [self.speed_lim[dist:], self._gen_speed(dist)])

    @property
    def obs(self):
        return np.concatenate([[self.pos, self.vel],
                               self.static_obs.flatten(),
                               self.speed_lim.flatten()])

    def truncated(self, depth):
        # return a version of self that has shorter depth
        copy = deepcopy(self)
        copy.depth = depth
        copy.static_obs = copy.static_obs[:depth]
        copy.speed_lim = copy.speed_lim[:depth]
        return copy

    def load_obs(self, obs):
        self.pos = obs[0]
        self.vel = obs[1]
        l = len(self.static_obs.flatten())
        self.static_obs = obs[2:l+2].reshape(self.static_obs.shape)
        self.speed_lim = obs[l+2:].reshape(self.speed_lim.shape)
        return self

    def as_dict(self):
        return {
            'pos': self.pos,
            'vel': self.vel,
            'static_obs': self.static_obs,
            'speed_lim': self.speed_lim,
        }

    def save(self, fname='state'):
        with open(fname, 'w') as f:
            f.write(json.dumps(self.as_dict(), cls=NumpyEncoder, indent=True))

    def load(self, fname='state'):
        with open(fname, 'r') as f:
            d = json.loads(f.read())
        self.pos = np.int16(d['pos'])
        self.vel = np.int16(d['vel'])
        self.static_obs = arr(d['static_obs'])
        self.speed_lim = arr(d['speed_lim'])

    def __str__(self):
        return str('\n'.join(['\n'.join([k, str(v.round(1) if k != 'dyna_obs' else v[:10])])
                              for k, v in self.as_dict().items()]))

    def __repr__(self):
        return self.__str__()
