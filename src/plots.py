import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Scene 1 small baseline cameras [1,3,4,5,7]

data = {
    "Lowe's ratio": [0.6, 0.62, 0.64, 0.66, 0.68,
                     0.6, 0.62, 0.64, 0.66, 0.68,
                     0.6, 0.62, 0.64, 0.66, 0.68, 
                     0.6, 0.62, 0.64, 0.66, 0.68,
                     0.6, 0.62, 0.64, 0.66,
                     0.6, 0.64, 0.68, 0.7,
                     0.6, 0.64, 0.68],
    "Pixel error (mean)": [1.69, 1.38, 1.43, 1.53, 1.65,
                    1.62, 1.30, 1.36, 1.47, 1.61, 
                    1.58, 1.25, 1.36, 1.47, 1.51, 
                    1.34, 1.02, 1.55, 1.69, 1.69, 
                    1.28, 1.12, 2.01, 2.47,
                    5.48, 4.35, 4.08, 4.19,
                    5.56, 4.49, 4.16],
    "Px error std": [1.79, 1.66, 2.59, 2.63, 3.53,
                     1.17, 1.02, 2.26, 2.23, 3.37, 
                     1.16, 0.98, 2.30, 2.30, 2.22, 
                     1.19, 0.94, 2.78, 2.89, 2.70, 
                     1.49, 1.22, 3.97, 4.32,
                     11.34, 10.65, 10.06, 9.39,
                     11.50, 10.88, 10.07],
    "Depth error": [1.84, 1.83, 2.16, 2.84, 2.25,
                    1.64, 1.66, 1.99, 2.73, 2.12, 
                    1.61, 1.62, 1.96, 2.71, 2.08, 
                    1.55, 1.41, 2.12, 2.75, 2.22, 
                    1.68, 1.35, 2.46, 2.90,
                    4.08, 3.81, 3.58, 3.88,
                    4.12, 3.87, 3.65],
    "d error std": [4.83, 4.56, 4.32, 4.02, 4.11,
                    1.44, 1.47, 1.53, 2.09, 2.05, 
                    1.39, 1.43, 1.48, 2.00, 1.61, 
                    1.52, 1.36, 1.40, 1.68, 1.45, 
                    1.65, 1.60, 1.52, 1.53,
                    7.58, 6.63, 6.17, 8.45,
                    7.68, 6.77, 6.24],
    "Nb points": [554, 626, 717, 788, 884, 
                  548, 616, 703, 774, 867, 
                  531, 595, 670, 719, 808, 
                  389, 424, 438, 433, 522, 
                  206, 202, 206, 179,
                  548, 703, 867, 944,
                  531, 670, 808],
    "Residual Threshold": [1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                           1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                           1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 
                           1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 
                           1e-9, 1e-9, 1e-9, 1e-9,
                           1e-6, 1e-6, 1e-6, 1e-6,
                           1e-7, 1e-7, 1e-7],
    "Method":['Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 
              'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 
              'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 
              'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 
              'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt', 'Levenberg-Marquardt',
              'Nelder-Mead', 'Nelder-Mead', 'Nelder-Mead', 'Nelder-Mead', 
              'Nelder-Mead', 'Nelder-Mead', 'Nelder-Mead']
}

df = pd.DataFrame(data)
df = pd.read_pickle("scene1_small_13457.pkl")
df_lm = df[df['Method'] == 'Nelder-Mead']

nb_res_thresh = len(df.groupby('Residual Threshold'))
nb_method = len(df.groupby('Method'))
nb_ratios = len(df.groupby("Lowe's ratio"))


palette = sns.color_palette(n_colors=nb_res_thresh)
fig, ax = plt.subplots()
p2 = sns.lineplot(x="Lowe's ratio", y="Nb points", hue='Residual Threshold', ci=None, data=df_lm, palette=palette)
p1 = sns.catplot(x="Lowe's ratio", y="Pixel error (mean)", hue='Residual Threshold', col='Method', data=df, kind='bar', palette=palette)

for item, color in zip(df_lm.groupby('Residual Threshold'),palette):
    #item[1] is a grouped data frame
    for x,y,m in item[1][["Lowe's ratio",'Nb points','Nb points']].values:
        ax.text(x,y,f'{m:.0f}',color=color)
# for i, (key, item) in enumerate(df.groupby(['Residual Threshold', 'Method', "Lowe's ratio"])):
#     #item[1] is a grouped data frame
#     l = item[['Pixel error']].values
#     mean_val = min(l, key=lambda x:abs(x-np.mean(l)))
#     for x,y,m,rt,met in item[["Lowe's ratio",'Pixel error','Nb points', 'Residual Threshold', 'Method']].values:
#         if y == mean_val and met=='lm':
#             ax.text(x,y,f'{m:.0f}',color=palette[i//(nb_method*nb_ratios)])

for ax in p1.axes.ravel():
    for c in ax.containers:
        labels = [f'{(v.get_height()):.1f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)

plt.show()