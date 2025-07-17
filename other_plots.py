import numpy as np
import matplotlib.pyplot as plt


def plot_setup(ax):
    ax.set_xlabel('thymol (wt. %)', fontsize=12)
    ax.set_ylabel(r'r ($\mathrm{\mu m}$)', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    


def plot_r_vs_wt(path):
    r = np.loadtxt(path, usecols=0) * 1e6
    r_dev = np.loadtxt(path, usecols=1) * 1e6
    wt_percent = np.loadtxt(path, usecols=2)
    alphas = [0.66, 0.72, 0.78, 0.92]

    fig, ax = plt.subplots(figsize=(6,4))
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\alpha$', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.spines['top'].set_visible(False)
    plot_setup(ax)

    colors = ['#88CCEE', '#44AA99', '#117733', '#DDCC77']

    handles_all = []
    labels_all = []

    for i, col in enumerate(colors):
        label_r = 'particle radii' if i == 0 else None
        label_a = r'$\alpha$ values' if i == 0 else None

        # Plot r
        l1, = ax.plot(
            wt_percent[i], r[i],
            'o', markersize=4, markerfacecolor='none', markeredgecolor=col, label=label_r
        )

        # Plot alpha
        l2, = ax2.plot(
            wt_percent[i], alphas[i],
            'x', markersize=4, markerfacecolor='none', markeredgecolor=col, label=label_a
        )

        # Collect handles/labels only on first iteration
        if i == 0:
            handles_all.extend([l1, l2])
            labels_all.extend([label_r, label_a])

        # Annotate r
        ax.text(
            wt_percent[i], r[i] + 0.05,
            f"{r[i]:.2f} μm",
            ha='center', va='bottom', fontsize=6
        )

    # Combined legend (labels in order: r, then alpha)
    ax.legend(handles_all, labels_all, loc='upper left')
    ax2.yaxis.set_tick_params(which='both', length=0)

    plt.savefig('/home/elias/proj/_photon_correlation/bsplot.png', dpi=300)
    plt.show()

plot_r_vs_wt('/home/elias/proj/_photon_correlation/_r.txt')