import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#import main

# data = {
#     "Scene": [],
#     "Pixel error (mean)": [],
#     "Depth error (mean)": [],
#     "Nb points": [],
#     "Baseline": [],
#     "Camera configuration":[]
# }

# cam_configs = [[0,1,2,3,4,5,6,7,8], [0,2,4,6,8], [1,3,4,5,7], [3,4,5], [1,4,7]]
# config_names = ['Full', 'Corners', 'Cross', 'Horizontal', 'Vertical']
# files = ["data/scene1_small/", "data/scene1_wide/", "data/scene2_small/", "data/scene2_wide/"]
# for i, cams in enumerate(cam_configs):
#     for j, filename in enumerate(files):
#         FOLDER_NAME = filename
#         print("File: ", filename, " // Config: ", config_names[i])
#         try:
#             n, res = main(cameras=cams)
#             data["Scene"].append(j//2 + 1)
#             data["Pixel error (mean)"].append(np.mean(res[0]))
#             data["Depth error (mean)"].append(np.mean(res[1]))
#             data["Nb points"].append(n)
#             if j%2 == 0:
#                 data["Baseline"].append('Small')
#             else:
#                 data["Baseline"].append('Wide')
#             data["Camera configuration"].append(config_names[i])
#         except:
#             print('Error')

# df = pd.DataFrame(data)
# df.to_pickle("scenes_configs.pkl")


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