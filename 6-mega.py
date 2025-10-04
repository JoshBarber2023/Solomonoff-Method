import matplotlib.pyplot as plt
import numpy as np

# ---------------- Predicted per-cell probabilities ----------------
pred_data = [
    [(0,1.00,1,0.00), (0,1.00,1,0.00), (0,0.90,6,0.10), (0,1.00,1,0.00), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (0,0.61,6,0.39), (6,0.88,0,0.12), (0,0.57,6,0.43), (0,0.96,6,0.04)],
    [(0,0.90,6,0.10), (6,0.87,0,0.13), (6,0.65,0,0.35), (6,0.92,0,0.08), (0,0.77,6,0.23)],
    [(0,1.00,1,0.00), (0,0.61,6,0.39), (6,0.61,0,0.39), (0,0.65,6,0.35), (0,0.87,6,0.13)],
    [(0,1.00,1,0.00), (0,1.00,1,0.00), (0,0.85,6,0.15), (0,0.97,6,0.03), (0,0.95,6,0.05)]
]

bma_data = [
    [(0,1.000,1,0.000), (0,1.000,1,0.000), (0,0.945,6,0.055), (0,1.000,1,0.000), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (0,0.548,6,0.452), (6,0.899,0,0.101), (0,0.576,6,0.424), (0,0.979,6,0.021)],
    [(0,0.945,6,0.055), (6,0.948,0,0.052), (6,0.618,0,0.382), (6,0.958,0,0.042), (0,0.893,6,0.107)],
    [(0,1.000,1,0.000), (0,0.548,6,0.452), (6,0.548,0,0.452), (0,0.618,6,0.382), (0,0.948,6,0.052)],
    [(0,1.000,1,0.000), (0,1.000,1,0.000), (0,0.935,6,0.065), (0,0.979,6,0.021), (0,0.991,6,0.009)]
]

# ---------------- Actual output grid ----------------
actual_grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 6, 6, 6, 0],
    [0, 6, 0, 6, 0],
    [0, 6, 0, 6, 0],
    [0, 0, 0, 0, 0]
])
n_rows, n_cols = actual_grid.shape

# ---------------- Color map ----------------
color_map = {
    0: 'white', 1: 'red', 2: 'green', 3: 'blue', 4: 'yellow',
    5: 'cyan', 6: 'orange', 7: 'purple', 8: 'gray', 9: 'brown'
}

# ---------------- Helper function ----------------
def plot_prob_grid(ax, data, title):
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    correct = 0
    for i in range(n_rows):
        for j in range(n_cols):
            top1_color, top1_prob, _, _ = data[i][j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_map[top1_color], alpha=top1_prob))
            ax.text(j+0.5, i+0.5, f"{top1_prob:.2f}", ha='center', va='center', fontsize=12)
            if top1_color == actual_grid[i,j]:
                correct += 1

    for i in range(n_rows+1):
        ax.axhline(i, color='black', lw=1)
    for j in range(n_cols+1):
        ax.axvline(j, color='black', lw=1)

    return correct / (n_rows * n_cols) * 100

# ---------------- Plot Figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(20,6))

# Left: Solomonoff Predictions
solomonoff_acc = plot_prob_grid(axes[0], pred_data, "Solomonoff Predictions")

# Middle: BMA Predictions
bma_acc = plot_prob_grid(axes[1], bma_data, "BMA Predictions")

# Right: Ground Truth
ax = axes[2]
ax.set_title("Ground Truth", fontsize=18, fontweight='bold')
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.invert_yaxis()
ax.set_aspect('equal')
for i in range(n_rows):
    for j in range(n_cols):
        color = color_map[actual_grid[i,j]]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        ax.text(j+0.5, i+0.5, str(actual_grid[i,j]), ha='center', va='center', fontsize=12, fontweight='bold')
for i in range(n_rows+1):
    ax.axhline(i, color='black', lw=1)
for j in range(n_cols+1):
    ax.axvline(j, color='black', lw=1)

# ---------------- Display accuracies ----------------
fig.text(0.17, 0.02, f"Top-1 Accuracy: {solomonoff_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')
fig.text(0.50, 0.02, f"Top-1 Accuracy: {bma_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0,0.05,1,0.95])
plt.show()
