import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from motion import get_spline

arr = np.array


def plot_eps(env):
    # last state appears behind a wall... don't need to show
    states = [h[0] for h in env.history[:-1]]
    # last state doesn't has an associated action of None
    actions = [h[1] for h in env.history[:-1]]
    mdist = env.move_dist
    statics = [s.static_obs for s in states]
    speeds = [s.speed_lim for s in states]
    epspeed = np.vstack([speeds[0], *[speed[-mdist:] for speed in speeds[1:]]])
    epstatic = np.vstack([statics[0], *[static[-mdist:]
                                        for static in statics[1:]]])
    _, (ax1, ax2) = plt.subplots(3, 1, sharex=True)
    ax2twin = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    for i in range(epspeed.shape[0]):
        for j in range(epspeed.shape[1]):
            ax1.add_patch(Rectangle(((i+.5), (j-.5)), 1, 1,
                                    alpha=(5-epspeed[i, j])/10, color='red', zorder=-1))
            # ax1.text((i+.5), (j-.5), epspeed[i, j].round().astype(int))
            if epstatic[i, j]:
                ax1.add_patch(Rectangle(((i+.5), (j-.5)),
                                        1, 1, color='cornflowerblue'))

    xseed = np.arange(states[0].depth + 1)
    xs = [xseed + i * mdist for i in range(len(actions))]
    p_bc = 0  # path (starting) boundary condition (first derivative)
    v_bc = 0  # velocity ^^
    for i, (x, state, (path, vel)) in enumerate(zip(xs, states, actions)):
        path = [state.pos] + list(path)
        vel = [state.vel] + list(vel)
        # plot path
        ax1.scatter(x[:mdist], path[:mdist], color='purple')
        spline = get_spline(x, path, p_bc, True)
        # eval the first deriv at the 'mdist' point
        p_bc = spline([x[mdist]], 1)[0]
        xs = np.linspace(x[0], x[mdist], num=20)
        ax1.plot(xs, spline(xs), '-', color='green', linewidth=0.7)

        # plot velocity profile
        ax2.scatter(x[:mdist], vel[:mdist], color='purple')
        vspline = get_spline(x, vel, v_bc, True)
        v_bc = vspline([x[mdist]], 1)[0]
        xs = np.linspace(x[0], x[mdist], num=20)
        ax2.plot(xs, vspline(xs), color='green', linewidth=0.7)

    # plot last point
    # pylint: disable=undefined-loop-variable
    ax1.scatter(x[mdist], path[mdist], color='purple')
    ax2.scatter(x[mdist], vel[mdist], color='purple')

    # set plotting options
    ax1.set_title('Path')
    ax2.set_title('Velocity Profile')
    ax2.set_ylim(0, 3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def save_episode(env):
    history = env.history
    states = arr([s[0] for s in history])
    np.save('history.npy', states)
