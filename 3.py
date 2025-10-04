import matplotlib.pyplot as plt
import numpy as np

# ---------------- Predicted per-cell probabilities ----------------
pred_data = [
    [(0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (0,0.73,7,0.27), (0,0.73,3,0.27), (0,0.73,7,0.27), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (7,0.73,3,0.27), (3,1.00,0,0.00), (7,0.73,3,0.27), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (7,0.86,3,0.14), (3,0.86,7,0.14), (7,0.86,3,0.14), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00), (0,1.00,1,0.00)]
]

# ---------------- BMA per-cell probabilities ----------------
bma_data = [
    [(0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (0,0.728,7,0.272), (0,0.728,3,0.272), (0,0.728,7,0.272), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (7,0.728,3,0.272), (3,1.000,0,0.000), (7,0.728,3,0.272), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (7,0.940,3,0.060), (3,0.940,7,0.060), (7,0.940,3,0.060), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000), (0,1.000,1,0.000)]
]

# ---------------- Actual output grid ----------------
actual_grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 7, 3, 7, 0],
    [0, 7, 3, 7, 0],
    [0, 0, 0, 0, 0]
])

n_rows, n_cols = actual_grid.shape

# ---------------- Color map ----------------
color_map = {
    0: 'white', 1: 'red', 2: 'green', 3: 'blue', 4: 'yellow',
    5: 'cyan', 6: 'orange', 7: 'purple', 8: 'gray', 9: 'brown'
}

# ---------------- Helper functions ----------------
def plot_prob_grid(ax, data, title):
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols)+0.5)
    ax.set_yticks(np.arange(n_rows)+0.5)
    ax.set_xticklabels(range(n_cols))
    ax.set_yticklabels(range(n_rows))
    ax.set_aspect('equal')

    correct = 0
    for i in range(n_rows):
        for j in range(n_cols):
            top_color, top_prob, _, _ = data[i][j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_map[top_color], alpha=top_prob))
            ax.text(j+0.5, i+0.5, f"{top_prob:.2f}", ha='center', va='center', fontsize=12)
            if top_color == actual_grid[i,j]:
                correct += 1

    # Draw grid lines
    for i in range(n_rows+1):
        ax.axhline(i, color='black', lw=1)
    for j in range(n_cols+1):
        ax.axvline(j, color='black', lw=1)

    return correct / (n_rows * n_cols) * 100

# ---------------- Plot Figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Left: Predicted
pred_acc = plot_prob_grid(axes[0], pred_data, "Solomonoff (Method 1)")

# Middle: BMA
bma_acc = plot_prob_grid(axes[1], bma_data, "BMA (Method 2)")

# Right: Actual
ax = axes[2]
ax.set_title("Ground Truth", fontsize=18, fontweight='bold')
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.invert_yaxis()
ax.set_xticks(np.arange(n_cols)+0.5)
ax.set_yticks(np.arange(n_rows)+0.5)
ax.set_xticklabels(range(n_cols))
ax.set_yticklabels(range(n_rows))
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

# ---------------- Display accuracies under panels ----------------
fig.text(0.17, 0.02, f"Top-1 Accuracy: {pred_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')
fig.text(0.50, 0.02, f"Top-1 Accuracy: {bma_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0,0.05,1,0.95])
plt.show()
