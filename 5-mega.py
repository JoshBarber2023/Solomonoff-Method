import matplotlib.pyplot as plt
import numpy as np

# ---------------- Predicted per-cell probabilities ----------------
pred_data = [
    [(0,0.82,8,0.18), (0,0.52,8,0.48), (0,1.00,1,0.00), (0,0.95,3,0.05), (0,0.95,6,0.05)],
    [(8,0.80,0,0.20), (8,1.00,0,0.00), (0,0.97,8,0.03), (0,0.91,3,0.05), (0,0.91,6,0.05)],
    [(8,0.55,0,0.42), (8,0.85,0,0.11), (3,0.79,0,0.21), (6,0.79,0,0.16), (6,0.84,0,0.16)],
    [(0,0.84,8,0.16), (0,0.54,8,0.46), (0,0.51,3,0.49), (6,0.49,0,0.45), (6,0.55,0,0.45)],
    [(0,0.84,8,0.16), (0,0.57,8,0.43), (0,0.60,3,0.40), (0,0.55,6,0.40), (0,0.55,6,0.45)]
]

# ---------------- BMA predicted probabilities ----------------
bma_data = [
    [(0,0.907,8,0.093), (0,0.605,8,0.395), (0,1.000,1,0.000), (0,0.946,3,0.054), (0,0.946,6,0.054)],
    [(8,0.800,0,0.200), (8,1.000,0,0.000), (0,0.993,8,0.007), (0,0.939,3,0.054), (0,0.939,6,0.054)],
    [(8,0.642,0,0.351), (8,0.943,0,0.049), (3,0.766,0,0.234), (6,0.766,0,0.180), (6,0.820,0,0.180)],
    [(0,0.785,8,0.215), (8,0.517,0,0.483), (3,0.620,0,0.380), (6,0.620,0,0.326), (6,0.674,0,0.326)],
    [(0,0.785,8,0.215), (0,0.519,8,0.481), (3,0.563,0,0.437), (6,0.563,0,0.383), (6,0.617,0,0.383)]
]

# ---------------- Actual output grid ----------------
actual_grid = np.array([
    [0, 8, 0, 6, 0],
    [8, 8, 0, 6, 0],
    [8, 0, 3, 6, 6],
    [8, 0, 3, 0, 6],
    [8, 0, 3, 0, 6]
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
