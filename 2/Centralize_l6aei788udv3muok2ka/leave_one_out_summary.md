# 🧮 Leave-One-Out Solomonoff Analysis for `Centralize_l6aei788udv3muok2ka`

**Held-out example index (1-based):** 3


## 📊 Ranked Hypotheses

| Rank | Hypothesis | Simplicity | Train Acc. | Solomonoff Score | Solomonoff Weight |
|------|------------|------------|------------|-----------------|-----------------|
| 1 | Objects are moved from their original location to the center of the grid maintaining their relational pattern | 0.573 | 0.880 | 0.504 | 0.265 |
| 2 | The input objects which are cross-shaped clusters of cells are repositioned to the centre of the grid. They maintain their color and form. | 0.487 | 0.840 | 0.409 | 0.215 |
| 3 | The pattern structure for an unique shape rotates 90 degrees counterclockwise locally between the input grid and output grid without affecting other objects or the entire input grid. | 0.497 | 0.700 | 0.348 | 0.183 |
| 4 | The objects in the input grid move towards the center of the grid, maintaining their original shape and color. | 0.382 | 0.900 | 0.344 | 0.181 |
| 5 | The object in the input grid moves diagonally downward from the top left corner towards the center of the grid, maintaining its original shape and color. | 0.291 | 0.880 | 0.256 | 0.135 |
| 6 | The objects in the grid move from their initial positions to form a plus '+' shape centered on the grid. The cell at the center of the object becomes the center of the '+' shape in the output, and the rest of the cells position themselves around this center cell. | 0.045 | 0.880 | 0.040 | 0.021 |

## 🎯 Held-out Example

**Input Grid:**
```
0 0 0 0 0
0 0 0 0 0
6 6 6 0 0
6 0 6 0 0
6 0 6 0 0
```
**Expected Output Grid:**
```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```

## 🤖 Predictions on Held-out Example

| Hypothesis # | Weight | Matches Expected? | Predicted Grid |
|--------------|--------|-------------------|----------------|
| 1 | 0.040 | ❌ |<br>```
0 0 0 0 0
0 6 0 0 0
0 6 6 6 0
0 6 0 0 0
0 0 0 0 0
```|
| 2 | 0.256 | ✅ |<br>```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```|
| 3 | 0.344 | ✅ |<br>```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```|
| 4 | 0.409 | ✅ |<br>```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```|
| 5 | 0.504 | ✅ |<br>```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```|
| 6 | 0.348 | ❌ |<br>```
0 0 0 0 0
0 0 6 0 0
0 0 6 6 6
0 0 0 6 0
0 0 0 6 0
```|

## 🏗️ Aggregated (Solomonoff-weighted) Output

**Argmax Grid:**
```
0 0 0 0 0
0 6 6 6 0
0 6 0 6 0
0 6 0 6 0
0 0 0 0 0
```

## 🔎 Per-cell Probabilities (Top 2 Colors per Cell)

| Row | Col | Top-1 | Top-2 |
|-----|-----|-------|-------|
| 0 | 0 | 0:1.00 | 1:0.00 |
| 0 | 1 | 0:1.00 | 1:0.00 |
| 0 | 2 | 0:1.00 | 1:0.00 |
| 0 | 3 | 0:1.00 | 1:0.00 |
| 0 | 4 | 0:1.00 | 1:0.00 |
| 1 | 0 | 0:1.00 | 1:0.00 |
| 1 | 1 | 6:0.82 | 0:0.18 |
| 1 | 2 | 6:0.98 | 0:0.02 |
| 1 | 3 | 6:0.80 | 0:0.20 |
| 1 | 4 | 0:1.00 | 1:0.00 |
| 2 | 0 | 0:1.00 | 1:0.00 |
| 2 | 1 | 6:0.82 | 0:0.18 |
| 2 | 2 | 0:0.80 | 6:0.20 |
| 2 | 3 | 6:1.00 | 0:0.00 |
| 2 | 4 | 0:0.82 | 6:0.18 |
| 3 | 0 | 0:1.00 | 1:0.00 |
| 3 | 1 | 6:0.82 | 0:0.18 |
| 3 | 2 | 0:1.00 | 1:0.00 |
| 3 | 3 | 6:0.98 | 0:0.02 |
| 3 | 4 | 0:1.00 | 1:0.00 |
| 4 | 0 | 0:1.00 | 1:0.00 |
| 4 | 1 | 0:1.00 | 1:0.00 |
| 4 | 2 | 0:1.00 | 1:0.00 |
| 4 | 3 | 0:0.82 | 6:0.18 |
| 4 | 4 | 0:1.00 | 1:0.00 |