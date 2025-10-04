import matplotlib.pyplot as plt
import numpy as np

# ---------------- Predicted per-cell probabilities ----------------
pred_data = [
    [(5,0.69,6,0.31), (5,0.61,6,0.27), (5,0.60,0,0.40), (0,1.00,1,0.00), (0,1.00,1,0.00)],
    [(5,0.61,6,0.27), (6,0.73,5,0.27), (5,0.61,6,0.27), (0,0.73,5,0.27), (0,1.00,1,0.00)],
    [(5,0.69,0,0.31), (5,0.61,6,0.27), (5,0.69,0,0.31), (0,1.00,1,0.00), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (0,0.63,5,0.37), (0,0.90,5,0.10), (0,0.90,5,0.10), (0,1.00,1,0.00)],
    [(0,1.00,1,0.00), (0,1.00,1,0.00), (0,0.90,5,0.10), (0,0.90,5,0.10), (0,0.90,5,0.10)]
]


# ---------------- BMA predicted probabilities ----------------
bma_data = [
    [(6,0.576,5,0.424), (5,0.657,6,0.246), (0,0.703,5,0.297), (0,1.000,1,0.000), (0,1.000,1,0.000)],
    [(5,0.657,6,0.246), (6,0.754,5,0.246), (5,0.657,6,0.246), (0,0.754,5,0.246), (0,1.000,1,0.000)],
    [(0,0.576,5,0.424), (5,0.657,6,0.246), (0,0.576,5,0.424), (0,1.000,1,0.000), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (0,0.627,5,0.373), (0,0.874,5,0.126), (0,0.874,5,0.126), (0,1.000,1,0.000)],
    [(0,1.000,1,0.000), (0,1.000,1,0.000), (0,0.874,5,0.126), (0,0.874,5,0.126), (0,0.874,5,0.126)]
]

# ---------------- Actual output grid ----------------
actual_grid = np.array([
    [6, 5, 0, 0, 0],
    [5, 6, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
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
fig, axes = plt.subplots(1, 3, figsize=(20,6))

# Left: Solomonoff Predictions
solomonoff_acc = plot_prob_grid(axes[0], pred_data, "Solomonoff (Method 1)")

# Middle: BMA Predictions
bma_acc = plot_prob_grid(axes[1], bma_data, "BMA (Method 2)")

# Right: Ground Truth
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

# ---------------- Display accuracies ----------------
fig.text(0.17, 0.02, f"Top-1 Accuracy: {solomonoff_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')
fig.text(0.50, 0.02, f"Top-1 Accuracy: {bma_acc:.1f}%", ha='center', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0,0.05,1,0.95])
plt.show()
