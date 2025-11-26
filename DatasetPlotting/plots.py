import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---- Publication settings ----
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "grid.alpha": 0.25,
    "figure.constrained_layout.use": True,
})

# -------- Approximate values digitized from your figure --------
# Epochs 0..13  (14 points)
epochs = np.arange(0, 14)

# Loss (unitless)
train_loss = np.array([1.24, 0.97, 0.93, 0.92, 0.90, 0.89, 0.89, 0.90,
                       0.89, 0.88, 0.89, 0.88, 0.87, 0.86])
val_loss   = np.array([1.06, 0.98, 1.00, 0.94, 0.94, 0.91, 0.90, 0.92,
                       0.90, 0.91, 0.90, 0.93, 0.96, 0.94])

# Accuracy (%)  — screenshot values, read off the y-axis
train_acc = np.array([33.0, 46.2, 48.5, 49.8, 51.2, 51.0, 51.5, 52.0,
                      52.6, 52.0, 51.2, 51.4, 52.2, 51.6])
val_acc   = np.array([41.2, 44.0, 42.4, 50.0, 45.3, 50.0, 44.0, 50.2,
                      48.7, 50.3, 43.2, 41.0, 41.8, 42.2])

# ---------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))  # two panels: loss & accuracy

# (a) Loss
ax = axs[0]
ax.plot(epochs, train_loss, label="Training Loss")
ax.plot(epochs, val_loss,   label="Validation Loss")
ax.set_title("Training & Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend(frameon=False, loc="upper right")
ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

# (b) Accuracy
ax = axs[1]
ax.plot(epochs, train_acc, label="Training Accuracy")
ax.plot(epochs, val_acc,   label="Validation Accuracy")
ax.set_title("Training & Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.grid(True)
ax.legend(frameon=False, loc="lower right")
ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

plt.savefig("training_curves_pub_nolr.pdf")   # vector for thesis
plt.savefig("training_curves_pub_nolr.png")   # high-DPI raster if needed
plt.show()