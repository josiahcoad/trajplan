#pylint: disable=not-an-iterable
from copy import deepcopy
import numpy as np
from itertools import product
from constants import MAX_CA, TAU, LAYER_DIST
from state import State
arr = np.array


def combinations(xs, repeat): return list(product(xs, repeat=repeat))


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
        curv = np.diff(dpath) / LAYER_DIST
        return curv * vel[:-1]**2
    cacc = cent_acc(path)

    t = 0
    vel = [state.vel, *vel]
    for i, p in enumerate(path):
        dt = (2 * LAYER_DIST) / (vel[i] + vel[i+1])
        t += dt
        t_int = int(round(t, np.log10(TAU).astype(int))*TAU)
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
        pass


def get_freespace(state):
    """
    choose action (i.e. behavior trajectory) that minimizes cost
    """
    # convert relative action space to absolute space
    lat_moves = combinations([-1, 0, 1], state.depth)
    vel_moves = combinations([-1, 0, 1], state.depth)
    paths = np.cumsum(lat_moves, axis=1) + state.pos.round().astype(int)
    vels = np.cumsum(vel_moves, axis=1) + state.vel.round().astype(int)

    # hack to deal with negative vels (and succeeding 0's)
    vels = np.maximum(np.ones(vels.shape, dtype=int)*.1, vels)
    vels = np.unique([tuple(v) for v in vels], axis=0)

    opaths = open_paths(state, paths)
    return list(product(opaths, vels))
    # return [[path, vel] for path in opaths
    #         for vel in open_vels(state, path, vels)]


def get_behav(state, weights=None, absolute=False):
    freespace = get_freespace(state)
    if len(freespace) == 0:
        raise NoPathError(state)

    def cost(action):
        return behav_cost(state, action, weights)

    p, v = min(freespace, key=cost)
    if absolute:
        return [p, v]
    # convert back to relative movements
    return np.concatenate([
        np.diff(np.insert(p, 0, state.pos)),
        np.diff(np.insert(v, 0, state.vel))])


def safe(state, action):
    path, vel = action
    return open_path(state, path) and open_vel(state, path, vel)


def behav_cost(state, action, weights=None, return_parts=False):
    """
    path cost
        fd: distance
        fk: curvature cost
        fl: lane crossings
    speed cost
        fr: reference cost
        fa: acceleration cost
        fj: jerk cost
        fc: centripedal acceleration cost
        # ft: time cost (?) (good to keep from setting v=0 unneccessarily)
    expecting weights in order: ['fr', 'fa', 'fj', 'fd', 'fk', 'fl', 'fc', 'ft]
    action should be absolute action
    """
    keys = ['fr', 'fa', 'fj', 'fd', 'fk', 'fl', 'fc', 'ft']
    if isinstance(weights, dict):
        weights = [weights.get(k, 1) for k in keys]
    weights = weights if weights is not None else np.ones(shape=7)
    # TODO: should we penalize centripedal acceleration in add. to constraining?
    path_, vel_ = action

    # TODO: make not hard coded (e.g. assumes lanes are 1 thick)
    if any(path_.round() < 0) or any(path_.round() > state.width - 1):
        return 100

    vel = arr([state.vel, *vel_])
    path = arr([state.pos, *path_])
    dpath = np.diff(path)
    lchange = np.abs(np.diff(path.round())) # TODO: make not hard coded (e.g. assumes lanes are 1 thick)
    dists = np.sqrt(dpath**2 + LAYER_DIST**2)
    ref_vel = arr([state.speed_lim[i, int(round(p))] for i, p in enumerate(path_)])

    vel_err = vel_ - ref_vel
    # TODO: right way to handle accel negative?
    accel = np.where(np.diff(vel) < 0, -1, 1) * \
            np.diff(vel)**2/(2*dists)
    jerk = accel - accel[1]  # TODO: must divide by t? (but instantenous...)
    curv = np.diff(dpath) / LAYER_DIST
    cacc = curv * vel_[:-1]**2

    # fr = sum(np.abs(np.where(vel_err < 0, vel_err, vel_err*2)))
    fr = sum(np.abs(vel_err**2))
    fa = sum(np.abs(accel))
    fj = sum(np.abs(jerk))

    fd = sum(dists) - state.depth
    fk = sum(np.abs(curv))
    fl = sum(lchange)
    fc = sum(np.abs(cacc))
    ft = sum((path_ - path_.round())**2)

    measures = [fr, fa, fj, fd, fk, fl, fc, ft]
    cost = np.dot(measures, weights)

    if return_parts:
        return cost, dict(zip(keys, measures))
    return cost


def test():
    state = State(3, 3)
    state.load()
    print(state)
    print(behav_cost(state, arr([[0,1,2], [3,4,4]]), return_parts=True))

if __name__ == '__main__':
    test()