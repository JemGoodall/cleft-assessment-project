import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from helper_funcs import scatter_plot_per_patient

run_title = sys.argv[1]

csv_path = f'/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/thesis/v3_code/final_exp/{run_title}/results_{run_title}.csv'
save_path = f'final_exp/{run_title}/patient_scatterplot_new.png'
df = pd.read_csv(csv_path, delimiter='|')
df = df.drop(df.columns[0], axis=1)

scatter_plot_per_patient(df, save_path)