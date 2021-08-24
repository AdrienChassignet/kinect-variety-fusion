import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


df = pd.read_pickle("scenes_configs.pkl")

# palette = sns.color_palette(n_colors=nb_res_thresh)
# fig, ax = plt.subplots()
# p2 = sns.lineplot(x="Lowe's ratio", y="Nb points", hue='Residual Threshold', ci=None, data=df_lm, palette=palette)
p1 = sns.catplot(x="Camera configuration", y="Pixel error (mean)", hue='Baseline', col='Scene', data=df, kind='bar')
p2 = sns.catplot(x="Camera configuration", y="Nb points", hue='Baseline', col='Scene', data=df, kind='bar')

# for item, color in zip(df_lm.groupby('Residual Threshold'),palette):
#     #item[1] is a grouped data frame
#     for x,y,m in item[1][["Lowe's ratio",'Nb points','Nb points']].values:
#         ax.text(x,y,f'{m:.0f}',color=color)

for ax in p1.axes.ravel():
    for c in ax.containers:
        labels = [f'{(v.get_height()):.1f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)
for ax in p2.axes.ravel():
    for c in ax.containers:
        labels = [f'{(v.get_height()):.0f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)

plt.show()