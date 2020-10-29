import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from motion import get_spline

arr = np.array

def plot_eps(env):
    states = [h[0] for h in env.history]
    actions = [h[1] for h in env.history]
    mdist = env.move_dist
    statics = [s.static_obs for s in states]
    speeds = [s.speed_lim for s in states]
    epspeed = np.vstack([speeds[0], *[speed[-mdist:] for speed in speeds[1:]]])
    epstatic = np.vstack([statics[0], *[static[-mdist:] for static in statics[1:]]])
    _, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2twin = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=epspeed[i, j]/10, color='red', zorder=-1))
            # ax1.text((i+.5), (j-.5), epspeed[i, j].round().astype(int))
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1))

    xseed = np.arange(states[0].depth + 1)
    xs = [xseed + i * mdist for i in range(len(actions))]
    p_bc = 0 # path (starting) boundary condition (first derivative)
    v_bc = 0 # velocity ^^
    path_len = 0
    for i, (x, state, (path, vel)) in enumerate(zip(xs, states, actions)):
        # plot path
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        ax1.scatter(x[:mdist], path[:mdist], color='purple')
        spline = get_spline(x, path, p_bc, True)
        p_bc = spline([x[mdist]], 1)[0] # eval the first deriv at the 'mdist' point
        xs = np.linspace(x[0], x[mdist], num=20)
        ys = spline(xs)
        ax1.plot(xs, ys, color='green')

        # plot heading and steer
        dy = np.diff(ys)
        dx = np.diff(xs)
        head = np.rad2deg(np.arctan2(dy, dx)) + [0]
        deltas = [0] + np.sqrt(dx**2 + dy**2)
        dists = np.cumsum(deltas) + path_len
        ax2.plot(dists, head, color='blue')
        steers = np.diff(head) / 30
        ax2twin.plot(dists[1:], steers, color='green')
        path_len = dists[-1]
        # TODO: curve = np.diff(steers)

        # plot velocity profile
        ax3.scatter(x[:mdist], vel[:mdist], color='purple')
        # idx = list(zip(range(i, mdist+i),
        #                arr(path[:mdist]).round().astype(int)))
        # ax3.scatter(x[:mdist]+1, epspeed[idx],
        #             color='orange', label='refvel' if i == 0 else None)
        vspline = get_spline(x, vel, v_bc, True)
        v_bc = vspline([x[mdist]], 1)[0]
        xs = np.linspace(x[0], x[mdist], num=20)
        ax3.plot(xs, vspline(xs), color='blue',
                label='vel' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 1), color='green',
                label='acc' if i == 0 else None)
        ax3.plot(xs, vspline(xs, 2), color='red',
                label='jrk' if i == 0 else None)
    
    # plot last point
    ax1.scatter(x[mdist], path[mdist], color='purple')
    
    # set plotting options 
    color = 'tab:blue'
    ax2.set_xlabel('path dist (s)')
    ax2.set_ylabel('heading (deg)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-75, 75)
    color = 'tab:green'
    ax2twin.set_ylabel('turn', color=color) 
    ax2twin.tick_params(axis='y', labelcolor=color)
    ax2twin.set_ylim(-1, 1)

    ax3.legend()
    ax3.set_xlim(*ax1.get_xlim())

    plt.show()


def save_episode(env):
    history = env.history
    states = arr([s[0] for s in history])
    np.save('history.npy', states)
