from matplotlib.axes import Axes


def format_plot(ax: Axes, color):
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="both", colors=color)
    ax.spines["top"].set_color(color)
    ax.spines["bottom"].set_color(color)
    ax.spines["left"].set_color(color)
    ax.spines["right"].set_color(color)
