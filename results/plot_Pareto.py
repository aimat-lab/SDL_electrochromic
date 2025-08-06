import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from matplotlib.patches import Ellipse
from botorch.utils.multi_objective.pareto import is_non_dominated



def plot_error_ellipse(x, y, std_x, std_y, ax, color, n_std=1, **kwargs):
    """
    Plots an error ellipse representing n_std standard deviations with a specific color.

    Parameters:
    x, y: Coordinates of the center
    std_x, std_y: Standard deviations along the x and y axis
    ax: Matplotlib Axes to draw on
    color: Color of the ellipse edge
    n_std: Number of standard deviations to draw
    kwargs: Additional arguments passed to Ellipse
    """
    std_devs = [1, 0.5]
    alphas = [0.15, 0.3]

    # Draw from the largest to the smallest ellipse
    for n_std, alpha in zip(std_devs, alphas):
        ellipse = Ellipse(
            (x, y),
            width=n_std * std_x,
            height=n_std * std_y,
            facecolor=color,
            edgecolor=color,
            linestyle=kwargs.get('linestyle', '--'),
            linewidth=kwargs.get('linewidth', 1),
            alpha=alpha
        )
        ax.add_patch(ellipse)


def get_sample_id(idx):
    return int(idx[-3:])


def plot_pareto_fronts(df):
    s_per_batch = 20
    n_tot_batches = 10
    
    y_min, y_max = 0, 1.6
    x_min, x_max = 0, .5
    sobol_color = 'b'#'darkred'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use a colormap for the optimized points
    norm = plt.Normalize(vmin=0, vmax=s_per_batch*n_tot_batches)
    
    for batch in range(10):
        idx_batch = [i for i in range((batch+1)*s_per_batch) if i in df.index]
        df_batch = df.loc[idx_batch]
        
        color = 'b' if batch==0 else plt.cm.turbo(norm((batch+1)*s_per_batch))
        alpha = 1. #.1 * (batch+1)
        linewidth = 1.7
        label = 'Random init' if batch==0 else f'Batch {batch}'
        
        # extract Pareto front mask
        train_obj = torch.from_numpy(df_batch[['defects_avg', 'optical_density_avg']].to_numpy()*np.array([-1, 1]))
        mask = is_non_dominated(train_obj).cpu().numpy()
        idx_pareto = np.array([i for i in range(len(mask)) if mask[i]])
        df_pareto = df.iloc[idx_pareto]
        
        # Add connection between Pareto points
        df_pareto_sorted = df_pareto.sort_values(by='defects_avg')
        print(df_pareto_sorted)
        pareto_x = df_pareto_sorted['defects_avg'].values
        pareto_y = df_pareto_sorted['optical_density_avg'].values
        for i in range(1, len(pareto_x)):
            # Draw a vertical line from previous point to current point
            #
            ax.plot(
                [pareto_x[i-1], pareto_x[i]], [pareto_y[i-1], pareto_y[i-1]], 
                c=color,
                linestyle='--', 
                linewidth=linewidth,
                alpha=alpha,
            )
            # Draw a horizontal line from current point to the same y level as the next point
            ax.plot(
                [pareto_x[i], pareto_x[i]], [pareto_y[i-1], pareto_y[i]], 
                c=color,
                linestyle='--', 
                linewidth=linewidth,
                alpha=alpha,
            )
        ax.plot(
            [pareto_x[0], pareto_x[0]], [y_min, pareto_y[0]], 
            c=color,
            linestyle='--', 
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.plot(
            [pareto_x[-1], x_max], [pareto_y[-1], pareto_y[-1]], 
            c=color,
            linestyle='--', 
            linewidth=linewidth, 
            alpha=alpha,
            label=label,
        )

    # Rest of datapoints
    idx_sobol = np.array([i for i in range(10)])
    idx_optim = np.array([i for i in range(10, len(df))])
    color_optim = np.array([df.index[ii] for ii in idx_optim])
    sobol = ax.scatter(
        df.iloc[idx_sobol]['defects_avg'], 
        df.iloc[idx_sobol]['optical_density_avg'], 
        s=23, c=sobol_color, alpha=alpha, #label='Sobol',
    )
    # Add error ellipses for initial samples with the same color
    for i, (x, y) in enumerate(zip(df.iloc[idx_sobol]['defects_avg'], df.iloc[idx_sobol]['optical_density_avg'])):
        std_x = df.iloc[idx_sobol[i]]['defects_std']
        std_y = df.iloc[idx_sobol[i]]['optical_density_std']
        color = sobol_color  # Fixed color matching Sobol scatter plot
        plot_error_ellipse(x, y, std_x, std_y, ax, color)
    
    qnehvi = ax.scatter(
        df.iloc[idx_optim]['defects_avg'], 
        df.iloc[idx_optim]['optical_density_avg'], 
        s=23, alpha=alpha, #label='qHNEVI',
        c=color_optim, cmap='turbo', norm=norm,
    )
    # Add error ellipses for optimized samples with matching colors
    for i, (x, y) in enumerate(zip(df.iloc[idx_optim]['defects_avg'], df.iloc[idx_optim]['optical_density_avg'])):
        std_x = df.iloc[idx_optim[i]]['defects_std']
        std_y = df.iloc[idx_optim[i]]['optical_density_std']
        color = plt.cm.turbo(norm(color_optim[i]))
        plot_error_ellipse(x, y, std_x, std_y, ax, color)

    # Adding legend and color bar
    plt.legend(fontsize=12, ncols=4, loc='upper left')
    cbar = plt.colorbar(qnehvi)
    cbar.set_label('Sample ID', rotation=90, fontsize=13)
    cbar.set_ticks(np.arange(0, s_per_batch*n_tot_batches + 1, step=s_per_batch))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Defect area (%)', fontsize=16)
    plt.ylabel('Absorbance at $\\lambda_{max}$', fontsize=16)
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    # plt.title('Pareto front evolution', fontsize=15)
    plt.grid(True)
    plt.savefig(f'./Pareto front evolution.png', dpi=400)
    plt.close()


## PRINT PARETO FRONTS EVOLUTION IN ONE FIGURE
df = pd.read_excel(f'./dataset_2obj.xlsx', index_col=0)[[
    'ink_concentration', 
    'spin_speed', 
    'spin_time', 
    'spin_acceleration', 
    'air_temp',
    'air_humidity',
    'defects', 
    'optical_density',
]]
df = df.iloc[:200]
grouped_df = (
    df.reset_index()  # Reset index to include sample_id in the dataframe
    .groupby(
        ["ink_concentration", "spin_speed", "spin_time", "spin_acceleration", 'air_temp', 'air_humidity'],
        as_index=False,
    )
    .agg(
        sample_id=("sample_id", "min"),  # Find the lowest sample_id
        defects_avg=("defects", "mean"),  # Mean of defects
        defects_std=("defects", "std"),  # Std of defects
        optical_density_avg=("optical_density", "mean"),  # Mean of optical_density
        optical_density_std=("optical_density", "std"),  # Std of optical_density
    )
)
grouped_df = grouped_df.set_index("sample_id").sort_index()

df = grouped_df[[
    'ink_concentration', 
    'spin_speed', 
    'spin_time', 
    'spin_acceleration', 
    'air_temp',
    'air_humidity',
    'defects_avg', 
    'defects_std',
    'optical_density_avg',
    'optical_density_std',
]]
new_indices = [int(get_sample_id(idx)) for idx in df.index]
df.index = new_indices
plot_pareto_fronts(df)
