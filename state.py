#pylint: disable=not-an-iterable
import time
import numpy as np
import json
from constants import TAU

arr = np.array

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class State:
    def __init__(self, width, depth, pos=None, vel=None, road=None, static_obs=None, dyna_obs=None, speed_lim=None):
        self.width = width
        self.depth = depth
        self.pos = pos if pos is not None \
            else np.random.randint(width, dtype=np.int8)
        self.vel = vel if pos is not None \
            else np.random.randint(1, 4, dtype=np.int8)
        self.road = road if road is not None \
            else self._gen_road(depth)
        self.static_obs = static_obs if static_obs is not None \
            else self._gen_static(depth)
        self.speed_lim = speed_lim if speed_lim is not None \
            else self._gen_speed(depth)
        # descritized "vertically" by TAU
        self.dyna_obs = dyna_obs if dyna_obs is not None else np.zeros(
            shape=(300, depth, width))  # self._gen_dyna()

    def _gen_static(self, dist):
        return np.random.binomial(1, 0.2, size=(dist, self.width))

    def _gen_speed(self, dist):
        return np.clip(np.random.normal(3, 2, size=(dist, self.width)), 1, 5)

    def _gen_road(self, dist):
        return np.zeros(shape=(dist, self.width))

    def _gen_dyna(self):
        # place a new dynamic obstacle (car) at some random column
        new_x = np.random.randint(self.width)
        return self._dyna_predict(new_x)

    def step(self, dist):
        """generate the next environment from random simulation after moving some distance and time"""
        self.static_obs = np.concatenate(
            [self.static_obs[dist:], self._gen_static(dist)])
        self.speed_lim = np.concatenate(
            [self.speed_lim[dist:], self._gen_speed(dist)])
        self.road = np.concatenate(
            [self.road[dist:], self._gen_road(dist)])

        # pre_x = np.argwhere(self.dyna_obs)  # only works with single cell dynas
        # new = self._gen_dyna(new_x)
        # pre = np.any([self._dyna_predict(pre_x) for x in pre_x])
        # np.dyna_obs = np.any([dyna[time:, dist:, :], new_dyna])
        # self.dyna_obs = np.concatenate([gen_road, self.static_obs])[:3]

    def _dyna_predict(self, x):
        # extrapolate obstacle over time (using speed_lim) (later use behavioral model or even RL policy to predict)
        p = arr([np.arange(3), [x, x, x]]).T  # determine path through grid
        v = arr([self.speed_lim[i, j] for i, j in p])
        dyna = self._dyna_expand(p[:, 1], v)
        return np.vstack([dyna, np.zeros(shape=(10, 3, 3))])

    @staticmethod
    def _dyna_expand(p, v):
        p = arr([np.arange(len(p)), p]).T
        p_c = p * [LAYER_DIST, 1]  # convert to cartesian coords
        dists = np.linalg.norm(np.diff(p_c, axis=0), axis=1)
        avg_v = (v[1:] + v[:-1]) / 2
        dt = dists / avg_v  # time to reach each node
        # we don't have sensor data to know how fast so assume constant speed from last known
        dt = arr([*dt, dt[-1]])
        seed = np.zeros(shape=(len(p), 3, 3))
        for i, (j, k) in enumerate(p):
            seed[i, j, k] = 1
        # rounding error problem?
        return np.repeat(seed, np.rint(TAU * dt).astype(int), axis=0)

    @property
    def obs(self):
        return np.concatenate([[self.pos, self.vel],
                               self.static_obs.flatten(),
                               self.speed_lim.flatten()])

    def load_obs(self, obs):
        self.pos = obs[0]
        self.vel = obs[1]
        l = len(self.static_obs.flatten())
        self.static_obs = obs[2:l+2].reshape(self.static_obs.shape)
        self.static_obs = obs[l+2:].reshape(self.speed_lim.shape)

    def as_dict(self):
        return {
            'pos': self.pos,
            'vel': self.vel,
            # 'road': self.road,
            'static_obs': self.static_obs,
            # 'dyna_obs': self.dyna_obs,
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
        return str('\n'.join(['\n'.join([k, str(v.round(1) if k != 'dyna_obs' else v[:10])]) for k, v in self.as_dict().items()]))
