import matplotlib.pyplot as plt
import numpy as np

# ---------------- Predicted per-cell probabilities (Top-1 and Top-2) ----------------
pred_data = [
    [(8,0.65,0,0.35), (8,0.93,0,0.07), (0,0.51,3,0.32), (0,0.68,6,0.32), (0,0.68,6,0.32)],
    [(8,0.65,0,0.35), (8,0.93,0,0.07), (0,0.51,3,0.32), (0,0.68,6,0.32), (0,0.68,6,0.32)],
    [(0,0.53,8,0.47), (8,0.57,0,0.43), (3,1.00,0,0.00), (6,1.00,0,0.00), (6,1.00,0,0.00)],
    [(0,0.68,8,0.32), (0,0.58,8,0.42), (3,1.00,0,0.00), (6,1.00,0,0.00), (6,1.00,0,0.00)],
    [(0,0.68,8,0.32), (0,0.58,8,0.42), (3,0.67,0,0.33), (6,0.67,0,0.33), (6,0.67,0,0.33)]
]

bma_data = [
    [ (8,0.536, 0,0.464), (8,0.939, 0,0.061), (3,0.360, 0,0.340), (0,0.640, 6,0.360), (0,0.640, 6,0.360) ],
    [ (8,0.536, 0,0.464), (8,0.939, 0,0.061), (3,0.360, 0,0.340), (0,0.640, 6,0.360), (0,0.640, 6,0.360) ],
    [ (0,0.599, 8,0.401), (8,0.640, 0,0.360), (3,1.000, 0,0.000), (6,1.000, 0,0.000), (6,1.000, 0,0.000) ],
    [ (0,0.640, 8,0.360), (8,0.600, 0,0.400), (3,1.000, 0,0.000), (6,1.000, 0,0.000), (6,1.000, 0,0.000) ],
    [ (0,0.640, 8,0.360), (8,0.600, 0,0.400), (3,0.825, 0,0.175), (6,0.825, 0,0.175), (6,0.825, 0,0.175) ]
]

# ---------------- Actual grid ----------------
actual_grid = np.array([
    [0, 8, 0, 6, 0],
    [8, 8, 0, 6, 0],
    [8, 0, 3, 6, 6],
    [8, 0, 3, 0, 6],
    [8, 0, 3, 0, 6]
])

# ---------------- Color map ----------------
color_map = {0:'white',1:'red',2:'green',3:'blue',4:'yellow',
             5:'cyan',6:'orange',7:'purple',8:'gray'}

n_rows, n_cols = actual_grid.shape

# ---------------- Helper functions ----------------
def plot_prob_grid(ax, data, title):
    ax.set_title(title, fontsize=20, fontweight='bold')  # larger title
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
            top1_color, top1_prob, _, _ = data[i][j]
            # Fill with top-1 color using alpha = probability
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_map[top1_color], alpha=top1_prob))
            # show probability
            ax.text(j+0.5, i+0.5, f"{top1_prob:.2f}", ha='center', va='center', fontsize=12)
    
    # Grid lines
    for i in range(n_rows+1):
        ax.axhline(i, color='black', lw=1)
    for j in range(n_cols+1):
        ax.axvline(j, color='black', lw=1)

def compute_accuracy(data, actual_grid):
    correct = sum(1 for i in range(n_rows) for j in range(n_cols) if data[i][j][0] == actual_grid[i,j])
    return correct / (n_rows * n_cols) * 100

# ---------------- Plotting ----------------
fig, axes = plt.subplots(1, 3, figsize=(20,7))  # slightly bigger figure

plot_prob_grid(axes[0], pred_data, "Solomonoff (Method 1)")
pred_acc = compute_accuracy(pred_data, actual_grid)

plot_prob_grid(axes[1], bma_data, "BMA (Method 2)")
bma_acc = compute_accuracy(bma_data, actual_grid)

# Ground truth
ax = axes[2]
ax.set_title("Ground Truth", fontsize=20, fontweight='bold')
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
        ax.add_patch(plt.Rectangle((j,i),1,1,color=color))
        ax.text(j+0.5,i+0.5,str(actual_grid[i,j]), ha='center', va='center', fontsize=14, fontweight='bold')
for i in range(n_rows+1):
    ax.axhline(i,color='black',lw=1)
for j in range(n_cols+1):
    ax.axvline(j,color='black',lw=1)

# Accuracy labels
fig.text(0.17, 0.05, f"Top-1 Accuracy: {pred_acc:.1f}%", ha='center', fontsize=16, fontweight='bold')
fig.text(0.50, 0.05, f"Top-1 Accuracy: {bma_acc:.1f}%", ha='center', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0,0.05,1,0.95])
plt.show()
