import pandas as pd
import numpy as np

df = pd.read_csv('logs/training/monitor.csv')

import matplotlib.pyplot as plt

s = (df.r / df.l)
s = s[s != 0].rolling(1000).mean()
plt.plot(np.cumsum(df.l)[-len(s):], s)
plt.show()

# results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
# plt.show()