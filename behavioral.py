#pylint: disable=not-an-iterable
import time
import random
from copy import deepcopy
import numpy as np
from itertools import product as product_, chain
import json

arr = np.array
# TODO: make velocity tracking an optimization instead of constraint


def product(xs, repeat): return list(product_(xs, repeat=repeat))


MAX_CA = 2  # 0.2g
tau = 10  # time descritization used for dynamic obstacle (in hertz)
layer_dist = 1  # function of v0?


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
    def __init__(self, width, depth, pos=None, vel=None, road=None, static_obs=None, dyna_obs=None, speed_lim=None, rseed=None):
        self.width = width
        self.depth = depth
        if rseed is not None:
            np.random.seed(rseed)
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
        # descritized "vertically" by tau
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
        p_c = p * [layer_dist, 1]  # convert to cartesian coords
        dists = np.linalg.norm(np.diff(p_c, axis=0), axis=1)
        avg_v = (v[1:] + v[:-1]) / 2
        dt = dists / avg_v  # time to reach each node
        # we don't have sensor data to know how fast so assume constant speed from last known
        dt = arr([*dt, dt[-1]])
        seed = np.zeros(shape=(len(p), 3, 3))
        for i, (j, k) in enumerate(p):
            seed[i, j, k] = 1
        # rounding error problem?
        return np.repeat(seed, np.rint(tau * dt).astype(int), axis=0)

    def as_space(self):
        return np.concatenate([[self.pos, self.vel],
                               self.road.flatten(), self.static_obs.flatten(),
                               self.dyna_obs.flatten(), self.speed_lim.flatten()])

    def as_dict(self):
        return {
            'pos': self.pos,
            'vel': self.vel,
            'road': self.road,
            'static_obs': self.static_obs,
            'dyna_obs': self.dyna_obs,
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
        self.road = arr(d['road'])
        self.static_obs = arr(d['static_obs'])
        self.dyna_obs = arr(d['dyna_obs'])
        self.speed_lim = arr(d['speed_lim'])

    def __str__(self):
        return str('\n'.join(['\n'.join([k, str(v.round(1) if k != 'dyna_obs' else v[:10])]) for k, v in self.as_dict().items()]))


def idx_exists(idx, array):
    return all((arr(idx) >= 0) & (arr(idx) < array.shape))


def rel_to_abs(init, deltas):
    # (1, [1, 0, -1, -1]) -> [2,  2,  1,  0]
    return np.cumsum(deltas) + init


def open_path(state, path):
    """
    Considers static obstacles and road geom
    """
    for i, p in enumerate(path):
        if not idx_exists([i, p], state.static_obs) \
                or state.static_obs[i, p]:
            return False
    return True


def open_vel(state, path, vel):
    """
    given path assumed to be staticly open, considers dynamic obstacles
        and regulator elements (encoded as speed limit occupancy grids)
        and centripedal acceleration
    """
    def cent_acc(path):
        path = arr([state.pos, *path])
        dpath = np.diff(path)
        curv = np.diff(dpath) / layer_dist
        return curv * vel[:-1]**2
    cacc = cent_acc(path)

    t = 0
    vel = [state.vel, *vel]
    for i, p in enumerate(path):
        dt = (2 * layer_dist) / (vel[i] + vel[i+1])
        t += dt
        t_int = int(round(t, np.log10(tau).astype(int))*tau)
        if not idx_exists([t_int, i, p], state.dyna_obs) \
                or state.dyna_obs[t_int, i, p] \
                or vel[i+1] > state.speed_lim[i, p] \
                or (idx_exists(i, cacc) and cacc[i] > MAX_CA):
            return False
    return True


def open_paths(state, paths):
    return [path for path in paths if open_path(state, path)]


def open_vels(state, path, vels):
    return [vel for vel in vels if open_vel(state, path, vel)]


class NoPathError(Exception):
    def __init__(self, state):
        print('pos:', state.pos)
        print(state.static_obs)


def get_freespace(state, weights=None):
    """
    choose action (i.e. behavior trajectory) that minimizes cost
    """
    # convert relative action space to absolute space
    lat_moves = product([-1, 0, 1], state.depth)
    vel_moves = product([-1, 0, 1], state.depth)
    paths = np.cumsum(lat_moves, axis=1) + state.pos
    vels = np.cumsum(vel_moves, axis=1) + state.vel

    # hack to deal with negative vels (and succeeding 0's)
    vels = np.maximum(np.ones(vels.shape, dtype=int)*.1, vels)
    vels = np.unique([tuple(v) for v in vels], axis=0)

    opaths = open_paths(state, paths)
    return [[path, vel] for path in opaths
             for vel in open_vels(state, path, vels)]


def get_behav(state, weights=None):
    freespace = get_freespace(state)
    if len(freespace) == 0:
        raise NoPathError(state)

    def cost(action): return behav_cost(state, action, weights)
    p, v = min(freespace, key=cost)
    return [p, v]


def safe(state, action):
    path, vel = action
    return open_path(state, path) and open_vel(state, path, vel)


def behav_cost(prv_state, action, weights=None, verbose=False):
    """
    path cost
        fd: distance
        fk: curvature cost
        fe: centripdal acc cost (bending energy?)
        fl: lane crossings
    speed cost
        fr: reference cost
        fa: acceleration cost
        fj: jerk cost
        fc: centripedal acceleration cost
        ft: time cost (?) (good to keep from setting v=0 unneccessarily)
    """
    weights = weights if weights is not None else np.ones(shape=7)
    # TODO: should we penalize centripedal acceleration in add. to constraining?
    # TODO: use prev_state speed limit or curr_state?
    path_, vel_ = action

    if any(path_ < 0) or any(path_ >= prv_state.width):  # planned outside sensor range
        return 100

    vel = arr([prv_state.vel, *vel_])
    path = arr([prv_state.pos, *path_])
    dpath = np.diff(path)
    dists = np.sqrt(dpath**2 + layer_dist**2)
    ref_vel = arr([prv_state.speed_lim[i, p] for i, p in enumerate(path_)])

    vel_err = vel_ - ref_vel
    # TODO: right way to handle accel negative?
    accel = np.where(np.diff(vel) < 0, -1, 1) * \
        np.diff(vel)**2/(2*dists)
    jerk = accel - accel[1]  # TODO... must divide by t? (instantenous)
    curv = np.diff(dpath) / layer_dist
    cacc = curv * vel_[:-1]**2

    fr = sum(np.abs(vel_err))
    fa = sum(np.abs(accel))
    fj = sum(np.abs(jerk))

    fd = sum(dists) - prv_state.depth
    fk = sum(np.abs(curv))
    fl = sum(dpath != 0)
    fc = sum(np.abs(cacc))

    if verbose:
        print(action)
        print(dict(zip(['fr', 'fa', 'fj'], arr([fr, fa, fj]).round(1))))
        print(dict(zip(['fd', 'fk', 'fl', 'fc'],
                       arr([fd, fk, fl, fc]).round(1))))

    return np.dot([fr, fa, fj, fd, fk, fl, fc], weights)


if __name__ == '__main__':
    state = State(3, 3)
    state.load()
    p, v, c = get_behav(state)
    print(p, v)
    open_vel(state, p[1:], v[1:])
    # print(State._dyna_expand(p, v))

# time = np.cumsum((2 * layer_dist) / (vel_[:-1] + vel_[1:]))
