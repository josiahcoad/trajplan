import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.bench.monitor import load_results
from matplotlib import animation


def plot_train(folder):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    def animate(_):
        df = load_results(folder)
        df = df.iloc[100:]
        df = df[df.r != 0]
        s = (df.r / df.l)
        s = s.rolling(1000).mean()
        ax1.clear()
        ax1.scatter(np.cumsum(df.l), df.r / df.l, s=.1)
        ax1.plot(np.cumsum(df.l)[-len(s):], s, c='red')
        ax1.set_ylim(-1, 2)
    _ = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


plot_train(folder='logs/training')
