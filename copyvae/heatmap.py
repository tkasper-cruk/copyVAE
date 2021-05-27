#!/usr/bin/env python3
import matplotlib.pylab as plt
import seaborn as sns

def draw_heatmap(data_arr, file_name):

    plt.figure(figsize=(17,100),dpi=120)
    fig = sns.heatmap(data_arr, cbar_kws={"shrink": 0.2},
                            cmap=sns.blend_palette(['#7CAACD','white', '#CB6527'],
                            as_cmap=True)).get_figure()
    fig.savefig('figure/{}.png'.format(file_name))

    return None
