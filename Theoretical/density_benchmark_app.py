"""
Density Estimation Benchmark — Native Matplotlib Desktop Application
Compares Histogram, fixed-BW KDE, and k-NN density estimation on two datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from scipy.stats import gaussian_kde, norm
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")
RNG = np.random.default_rng(seed=42)
# Dataset A: Synthetic Gaussian Mixture 
def make_synthetic() -> np.ndarray:
    a = RNG.normal(0.0,  0.2, 300)
    b = RNG.normal(10.0, 3.0, 700)
    return np.concatenate([a, b])

# Dataset B: Old Faithful (eruption duration)
def make_faithful() -> np.ndarray:
    df = sns.load_dataset("geyser")
    return df["duration"].dropna().values

def scale(raw: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(raw.reshape(-1, 1)).ravel()

GRID_PTS = 800

def eval_grid(data: np.ndarray, pad: float = 0.5) -> np.ndarray:
    return np.linspace(data.min() - pad, data.max() + pad, GRID_PTS)

# Histogram (Freedman-Diaconis)
def fd_histogram(data: np.ndarray):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    h   = max(2.0 * iqr * len(data) ** (-1 / 3), 1e-3)
    counts, edges = np.histogram(data, bins=np.arange(data.min(), data.max() + h, h), density=True)
    xs = np.repeat(edges, 2)
    ys = np.concatenate([[0], np.repeat(counts, 2), [0]])
    return xs, ys

# Fixed-BW KDE (Scott's rule via gaussian_kde)
def fixed_kde(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(data)(grid)

#k-NN Density Estimator 
def knn_density(data_sorted: np.ndarray, grid: np.ndarray, k: int) -> np.ndarray:
    """
    f̂(x) = (k - 1) / (2 · n · R_k(x))
    R_k(x): distance to the k-th nearest neighbour (self excluded).
    """
    n   = len(data_sorted)
    k   = int(np.clip(k, 2, n - 1))
    out = np.empty(len(grid))

    for i, x in enumerate(grid):
        dists = np.abs(data_sorted - x)
        dists_sorted = np.sort(dists)
        R_k = dists_sorted[k]
        out[i] = (k - 1) / (2.0 * n * R_k) if R_k > 1e-12 else 0.0

    return out

def true_density_A(grid: np.ndarray, raw: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(raw.reshape(-1, 1))
    mu_a = scaler.transform([[0.0]])[0, 0]
    mu_b = scaler.transform([[10.0]])[0, 0]
    s    = scaler.scale_[0]
    w_a, w_b = 300 / 1000, 700 / 1000
    return w_a * norm.pdf(grid, mu_a, 0.2 / s) + w_b * norm.pdf(grid, mu_b, 3.0 / s)

raw_syn = make_synthetic()
raw_ff  = make_faithful()

datasets = {
    "Synthetic Mixture": scale(raw_syn),
    "Old Faithful":      scale(raw_ff),
}

fig = plt.figure(figsize=(13, 7.5))
fig.patch.set_facecolor("#FAFAFA")

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    left=0.30, right=0.97,
    top=0.92,  bottom=0.06,
    hspace=0.08, wspace=0.35,
    height_ratios=[5, 0.55, 0.55],
)

ax_main  = fig.add_subplot(gs[0, :])
ax_slide = fig.add_subplot(gs[1, 1])
ax_radio = fig.add_subplot(gs[0, 0])   # repurposed for widgets below

ax_main.set_facecolor("#FFFFFF")
ax_main.spines[["top", "right"]].set_visible(False)

ax_radio_widget  = fig.add_axes([0.02, 0.55, 0.22, 0.28])
ax_slider_widget = fig.add_axes([0.02, 0.38, 0.22, 0.07])

ax_radio.remove()

C_HIST  = "#90A4AE"
C_KDE   = "#E53935"
C_KNN   = "#F57C00"
C_TRUE  = "#1A237E"
C_RUG   = "#78909C"

K_INIT = 20

active_label = "Synthetic Mixture"
data = datasets[active_label]
data_sorted = np.sort(data)
grid = eval_grid(data)

hx, hy = fd_histogram(data)
line_hist, = ax_main.plot(hx, hy, color=C_HIST, lw=1.2, label="Histogram (FD)", alpha=0.85)
fill_hist  = ax_main.fill_between(hx, hy, alpha=0.22, color=C_HIST)

kde_y = fixed_kde(data, grid)
line_kde, = ax_main.plot(grid, kde_y, color=C_KDE, lw=2.0,
                         linestyle="--", label=f"KDE (Scott)")

knn_y = knn_density(data_sorted, grid, K_INIT)
line_knn, = ax_main.plot(grid, knn_y, color=C_KNN, lw=2.2,
                          label=f"$k$-NN  ($k$={K_INIT})")

true_y = true_density_A(grid, raw_syn)
line_true, = ax_main.plot(grid, true_y, color=C_TRUE, lw=2.5,
                           linestyle=":", label="True density", alpha=0.9)

rug_y = np.full(len(data), ax_main.get_ylim()[0])
rug = ax_main.scatter(data, rug_y - 0.005, marker="|", s=18,
                       color=C_RUG, alpha=0.35, label="Observations", zorder=2)

ax_main.set_xlabel("Scaled $x$", fontsize=11)
ax_main.set_ylabel("Density  $\\hat{f}(x)$", fontsize=11)
ax_main.set_title("Density Estimation Benchmark — Synthetic Gaussian Mixture",
                  fontsize=13, fontweight="bold", pad=10)
leg = ax_main.legend(loc="upper right", framealpha=0.9, fontsize=10,
                     edgecolor="#CCCCCC")

slider = Slider(
    ax=ax_slider_widget,
    label="$k$",
    valmin=5, valmax=200, valinit=K_INIT, valstep=1,
    color=C_KNN,
)
slider.label.set_fontsize(12)
slider.valtext.set_fontsize(11)

radio = RadioButtons(
    ax_radio_widget,
    labels=("Synthetic Mixture", "Old Faithful"),
    activecolor=C_KNN,
)
for label in radio.labels:
    label.set_fontsize(10)

fig.text(0.12, 0.86, "Controls", ha="center", fontsize=12,
         fontweight="bold", color="#37474F")
fig.text(0.12, 0.82, "Dataset", ha="center", fontsize=10, color="#546E7A")
fig.text(0.12, 0.46, "$k$-NN  bandwidth", ha="center", fontsize=10, color="#546E7A")

caption = (
    "k-NN adapts its window locally: it contracts in dense regions and expands in sparse ones, "
    "avoiding the global-bandwidth failure of fixed KDE."
)
fig.text(0.30, 0.005, caption, fontsize=8.5, color="#546E7A", style="italic")

_fill_ref = [fill_hist]   # mutable container so closures can update it
_rug_ref  = [rug]

def _redraw(k: int):
    data       = datasets[active_label]
    data_sorted = np.sort(data)
    grid        = eval_grid(data)

    hx, hy = fd_histogram(data)
    line_hist.set_xdata(hx)
    line_hist.set_ydata(hy)
    _fill_ref[0].remove()
    _fill_ref[0] = ax_main.fill_between(hx, hy, alpha=0.22, color=C_HIST)

    line_kde.set_xdata(grid)
    line_kde.set_ydata(fixed_kde(data, grid))

    line_knn.set_xdata(grid)
    line_knn.set_ydata(knn_density(data_sorted, grid, k))
    line_knn.set_label(f"$k$-NN  ($k$={k})")

    if active_label == "Synthetic Mixture":
        line_true.set_xdata(grid)
        line_true.set_ydata(true_density_A(grid, raw_syn))
        line_true.set_visible(True)
    else:
        line_true.set_visible(False)

    _rug_ref[0].remove()
    _rug_ref[0] = ax_main.scatter(
        data, np.full(len(data), ax_main.get_ylim()[0] - 0.005),
        marker="|", s=18, color=C_RUG, alpha=0.35, zorder=2,
    )

    ax_main.relim()
    ax_main.autoscale_view()
    ax_main.set_title(
        f"Density Estimation Benchmark — {active_label}",
        fontsize=13, fontweight="bold", pad=10,
    )
    leg.remove()
    ax_main.legend(loc="upper right", framealpha=0.9, fontsize=10, edgecolor="#CCCCCC")
    fig.canvas.draw_idle()


def on_slider(val):
    _redraw(int(slider.val))


def on_radio(label):
    global active_label
    active_label = label
    _redraw(int(slider.val))


slider.on_changed(on_slider)
radio.on_clicked(on_radio)

plt.show()
