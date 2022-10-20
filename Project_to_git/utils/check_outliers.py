import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np
from collections import Counter

run_title = sys.argv[1]

csv_path = f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/v3_code/final_exp/{run_title}/results_{run_title}_outliers.csv'
save_path = f'final_exp/{run_title}/patient_scatterplot_outliers.png'
df = pd.read_csv(csv_path)


def scatter_plot_per_patient(df1, save_path='bubble_plot.png'):
    #https://datascience.stackexchange.com/questions/89692/plot-two-categorical-variables
    df = df1.copy()
    df["file_stem"] = [f'ID{x[1]}_Class{x[0]}_file{x[2]}' for x in df['file_stem'].str.split('_')]
    counts = df.groupby(['predictions', 'targets', 'file_stem']).size().reset_index(name='Count')
    xnoise, ynoise = np.random.random(len(counts)) / 2, np.random.random(len(counts)) / 2
    counts["targets"] = counts['targets'] + xnoise
    counts["predictions"] = counts['predictions'] + ynoise

    # Plot the scatterplot
    ax = sns.scatterplot(data=counts,
                  x='predictions', y='targets', hue='file_stem', size='Count', edgecolors="black", sizes=(70, 900), alpha=0.9,palette="Paired")
    ax.invert_yaxis()
    plt.xticks([0.25, 1.25, 2.25, 3.25, 4.25], [0,1,2,3,4])  # The reason the xticks start at 0.25
    # and go up in increments of 1 is because the center of the noise will be around 0.25 and ordinal
    # encoded labels go up in increments of 1.
    plt.yticks([0.25], [4])  # This has the same reason explained for xticks
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Classification per patient")

    # Format figure size, spines and grid
    ax.grid(axis='y', which='minor', color='black', alpha=0.2)
    ax.grid(axis='x', which='minor', color='black', alpha=0.2)
    sns.despine(left=True)
    # Format ticks
    ax.tick_params(axis='both', length=0, pad=10, labelsize=12)
    ax.tick_params(axis='x', which='minor', length=25, width=0.8, color=[0, 0, 0, 0.2])
    minor_xticks = [tick + 0.5 for tick in ax.get_xticks() if tick != ax.get_xticks()[-1]]
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_xticks, minor=True)

    ax.legend(bbox_to_anchor=(1.3, 1), borderaxespad=0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()

# scatter_plot_per_patient(df, save_path)



df["file_stem"] = [f'ID{x[1]}_Class{x[0]}_{x[2]}' for x in df['file_stem'].str.split('_')]
counts = df.groupby(['predictions', 'targets', 'file_stem']).size().reset_index(name='Count')
print(counts)
file_list = list(df["file_stem"])
frame_counts_per_file = Counter(file_list)
# print(counts.loc[(counts.file_stem == 'ID12_Class4_210210') &
#                  (counts.predictions == 0)]['Count'])
print(frame_counts_per_file)
proportions_0 = {}
for (k, v) in frame_counts_per_file.items():
    proportions_0[k] = counts.loc[(counts.file_stem == k) &
                                  (counts.predictions == 0)]['Count']/v

print(proportions_0)