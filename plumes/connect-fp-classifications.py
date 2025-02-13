"""

connect-fp-classifications.py

This script connects fp to manual classifications completed after running get-fp-hotspots

Author: Karlee Zammal the Party Mammal
Contact: karlee.zammit@nrcan-rncan.gc.ca
Date: 2025-01-25

"""

import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns

def plot_var(TP, FP, FP_smoke, FP_not_smoke, var):


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(5,10))
    sns.histplot(ax=ax1, data=TP, x=var, hue='daynight', stat='density')
    ax1.set_title('TP')
    sns.histplot(ax=ax2, data=FP, x=var, hue='daynight', stat='density')
    ax2.set_title('FP-all')
    sns.histplot(ax=ax3, data=FP_smoke, x=var, hue='daynight', stat='density')
    ax3.set_title('FP-smoke')
    sns.histplot(ax=ax4, data=FP_not_smoke, x=var, hue='daynight', stat='density')
    ax4.set_title('FP-not-smoke')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    if var == 'frp':
        ax1.set_xlim(0, 25)

    fig.suptitle(var)
    plt.tight_layout()
    plt.savefig(r'C:\Users\kzammit\Documents\plumes\plots\class-script' + '\\' + var + '.png')


def calc_stats(TP, FP, FP_smoke, FP_not_smoke, var):

    min_fire = TP[var].min()
    min_fp_all = FP[var].min()
    min_fp_smoke = FP_smoke[var].min()
    min_fp_not_smoke = FP_not_smoke[var].min()

    max_fire = TP[var].max()
    max_fp_all = FP[var].max()
    max_fp_smoke = FP_smoke[var].max()
    max_fp_not_smoke = FP_not_smoke[var].max()

    mean_smoke = FP_smoke[var].mean()
    mean_not_smoke = FP_not_smoke[var].mean()
    mean_all_fp = FP[var].mean()
    mean_fire = TP[var].mean()

    std_smoke = FP_smoke[var].std()
    std_not_smoke = FP_not_smoke[var].std()
    std_all_fp = FP[var].std()
    std_fire = TP[var].std()

    output = {'var': var,
              'min-TP': min_fire, 'max-TP': max_fire, 'mean-TP': mean_fire, 'std-TP': std_fire,
              'min-FP': min_fp_all, 'max-FP': max_fp_all, 'mean-FP': mean_all_fp, 'std-FP': std_all_fp,
              'min-FP-smoke': min_fp_smoke, 'max-FP-smoke': max_fp_smoke, 'mean-FP-smoke': mean_smoke,
              'std-smoke': std_smoke,
              'min-FP-not-smoke': min_fp_not_smoke, 'max-FP-not-smoke': max_fp_not_smoke,
              'mean-FP-not-smoke': mean_not_smoke, 'std-not-smoke': std_not_smoke}

    return output

if __name__ == "__main__":

    df_folder = r'C:\Users\kzammit\Documents\plumes\dfs'

    fp_file = pd.read_csv(df_folder + '\\' + 'fp-2023-09-23.csv')
    fp_file = fp_file.loc[:, : 'sat']

    tp_file = pd.read_csv(df_folder + '\\' + 'tp-2023-09-23.csv')
    tp_file = tp_file.loc[:, : 'sat']

    class_file = pd.read_excel(df_folder + '\\' + 'all-false-positives-manual-2023-09-23.xlsx')

    # assign the ba class to the false positive dataframe
    class_file['orig_index'] = class_file['orig_index'].apply(ast.literal_eval)

    # Loop over each row in class_file to assign 'In_BA' values to the corresponding indices in fp_file
    for idx, row in class_file.iterrows():
        indices = row['orig_index']  # List of indices
        in_ba_value = row['In BA?']  # Value from 'In_BA' column
        img_value = row['VIIRS Smoke?']

        # Assign the value from 'In_BA' to the corresponding indices in fp_file
        fp_file.loc[indices, 'BA'] = in_ba_value
        fp_file.loc[indices, 'Img'] = img_value


    TP = fp_file[fp_file['BA']=='Y']

    # Add back in all of the other TP's
    TP = pd.concat([TP, tp_file])

    FP = fp_file[fp_file['BA']=='N']
    FP_smoke = FP[FP['Img'].isin(['ST1', 'ST2'])]
    FP_not_smoke = FP[~FP['Img'].isin(['ST1', 'ST2'])]

    plot_var(TP, FP, FP_smoke, FP_not_smoke, 'frp')
    plot_var(TP, FP, FP_smoke, FP_not_smoke, 'bright_ti4')
    plot_var(TP, FP, FP_smoke, FP_not_smoke, 'scan')
    plot_var(TP, FP, FP_smoke, FP_not_smoke, 'track')

    stats_df = pd.DataFrame(columns=['var',
                                     'min-TP', 'max-TP', 'mean-TP', 'std-TP',
                                     'min-FP', 'max-FP', 'mean-FP', 'std-FP',
                                     'min-FP-smoke', 'max-FP-smoke', 'mean-FP-smoke', 'std-smoke',
                                     'min-FP-not-smoke', 'max-FP-not-smoke', 'mean-FP-not-smoke', 'std-not-smoke'])

    stats_df.loc[len(stats_df)] = calc_stats(TP, FP, FP_smoke, FP_not_smoke, 'frp')
    stats_df.loc[len(stats_df)] = calc_stats(TP, FP, FP_smoke, FP_not_smoke, 'bright_ti4')
    stats_df.loc[len(stats_df)] = calc_stats(TP, FP, FP_smoke, FP_not_smoke, 'scan')
    stats_df.loc[len(stats_df)] = calc_stats(TP, FP, FP_smoke, FP_not_smoke, 'track')
    stats_df.to_excel(r'C:\Users\kzammit\Documents\plumes\plots\class-script\stats-df.xlsx', index=False)



