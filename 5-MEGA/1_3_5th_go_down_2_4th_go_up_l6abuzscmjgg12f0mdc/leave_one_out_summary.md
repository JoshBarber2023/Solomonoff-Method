# 🧮 Leave-One-Out Solomonoff Analysis for `1_3_5th_go_down_2_4th_go_up_l6abuzscmjgg12f0mdc`

**Held-out example index (1-based):** 3


## 📊 Ranked Hypotheses

| Rank | Hypothesis | Simplicity | Train Acc. | Solomonoff Score | Solomonoff Weight |
|------|------------|------------|------------|-----------------|-----------------|
| 1 | Individual objects in the grid are duplicated in a downward direction until they reach the bottom row or another object. | 0.543 | 0.700 | 0.380 | 0.081 |
| 2 | The objects are replicated in the output grid by adding copies in the vertical direction. | 0.673 | 0.560 | 0.377 | 0.081 |
| 3 | Objects are replicated downwards in the grid based on the color of the object. The number of replications is determined by the number of the different colors present in the grid. | 0.523 | 0.680 | 0.355 | 0.076 |
| 4 | The objects on the grid are duplicated in a vertical manner, filling each column from top to bottom. | 0.558 | 0.620 | 0.346 | 0.074 |
| 5 | Objects in the grid are duplicated to fill their column from top to bottom, maintaining their color and position. | 0.508 | 0.660 | 0.335 | 0.072 |
| 6 | Each object in the input grid is duplicated in the output grid in a vertical line, maintaining their horizontal position. The number of duplicates is equal to the height of the grid. | 0.477 | 0.660 | 0.315 | 0.067 |
| 7 | Each unique colored object in the input grid is expanded vertically to fill any empty space above it until it hits another object or the grid boundary, while preserving its original shape. | 0.482 | 0.600 | 0.289 | 0.062 |
| 8 | Each object in the grid is duplicated in the vertical direction, with copies placed above and below the original object until they fill the grid or reach another object. | 0.573 | 0.500 | 0.286 | 0.061 |
| 9 | Each object in the input grid is replicated in the output grid, appearing in the same column but spread across all rows. | 0.432 | 0.640 | 0.277 | 0.059 |
| 10 | Every object in the grid extends vertically to fill the entire column it is located in, from top to bottom, maintaining the original color of each object. | 0.437 | 0.620 | 0.271 | 0.058 |
| 11 | Objects in the input grid expand both vertically upwards and downwards from their original position, maintaining their original column. Some objects may expand to occupy a total of 2, 3, or 4 cells in the output grid depending on their input relationships. | 0.342 | 0.680 | 0.232 | 0.050 |
| 12 | Each object in the input grid is expanded vertically downwards to fill the empty space below it, until it reaches another object or the bottom of the grid. If an object is already at the bottom of the grid, it expands upwards instead. | 0.327 | 0.680 | 0.222 | 0.047 |
| 13 | The objects in the grid are shifted, duplicated, or expanded in the vertical direction according to their relations to other objects. | 0.352 | 0.620 | 0.218 | 0.047 |
| 14 | Objects in the input grid are replicated in the output grid, extending vertically downwards from their original positions. The number of replications depends on the position of the object in the input grid. | 0.312 | 0.680 | 0.212 | 0.045 |
| 15 | Objects in the grid are replicated vertically and positioned in a way to form a column of each object type. | 0.422 | 0.480 | 0.203 | 0.043 |
| 16 | Objects in the grid are duplicated vertically based on their initial position from the top to the bottom of the grid. Each object is replicated based on the number of its initial vertical position plus one. | 0.367 | 0.540 | 0.198 | 0.042 |
| 17 | Each object in the grid is replicated in its row until it fills the row from its original position to the leftmost or rightmost side, based on its initial position in the input grid. | 0.402 | 0.440 | 0.177 | 0.038 |
| 18 | Each colored cell in the input grid expands vertically, creating a column that includes the original cell's position and extends both upwards and downwards until it hits another colored cell or the grid border. | 0.281 | 0.600 | 0.169 | 0.036 |
| 19 | All the distinct colored cells in the input grid are duplicated and arranged in a vertical column in their respective columns in the output grid | 0.261 | 0.600 | 0.157 | 0.034 |
| 20 | Each object in the input grid is expanded vertically, stretching downwards until it reaches the bottom of the grid or another object. The expanded parts take on the same color (value) as the original object. If a space is encountered which is already occupied by another object's expansion, the latter overwrites the former. | 0.055 | 0.680 | 0.038 | 0.008 |


## 🎯 Held-out Example

**Input Grid:**
```
0 0 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 0 0 0
0 0 0 0 0
```
**Expected Output Grid:**
```
0 8 0 6 0
8 8 0 6 0
8 0 3 6 6
8 0 3 0 6
8 0 3 0 6
```

## 🤖 Predictions on Held-out Example

| Hypothesis # | Weight | Matches Expected? | Predicted Grid |
|--------------|--------|-------------------|----------------|
| 1 | 0.212 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 3 6 6
0 0 3 6 6
```|
| 2 | 0.277 | ❌ |<br>```
0 8 0 3 6
0 8 0 3 6
0 8 0 3 6
0 8 0 3 6
0 8 0 3 6
```|
| 3 | 0.315 | ❌ |<br>```
0 8 0 0 0
0 8 0 0 0
0 8 3 6 6
0 8 0 0 0
0 8 0 0 0
```|
| 4 | 0.346 | ❌ |<br>```
8 8 0 0 0
8 8 0 0 0
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
```|
| 5 | 0.271 | ❌ |<br>```
0 8 0 0 0
0 8 0 0 0
0 8 3 6 6
0 8 0 0 0
0 8 0 0 0
```|
| 6 | 0.380 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 3 6 6
0 0 3 6 6
```|
| 7 | 0.222 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
```|
| 8 | 0.169 | ❌ |<br>```
0 8 0 0 0
8 8 0 0 0
0 8 3 6 6
0 8 0 0 0
0 8 0 0 0
```|
| 9 | 0.203 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
8 8 0 0 0
8 8 0 0 0
```|
| 10 | 0.377 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 0 0 0
0 0 0 0 0
```|
| 11 | 0.177 | ❌ |<br>```
0 0 0 0 0
8 8 8 8 8
3 3 3 6 6
0 0 0 0 0
0 0 0 0 0
```|
| 12 | 0.286 | ❌ |<br>```
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 3 6 6
0 0 0 0 0
```|
| 13 | 0.218 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 3 6 6
```|
| 14 | 0.335 | ❌ |<br>```
0 8 0 0 0
8 8 0 0 0
0 8 3 6 6
0 8 0 0 0
0 8 0 0 0
```|
| 15 | 0.198 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 3 6 6
```|
| 16 | 0.289 | ❌ |<br>```
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 0 0 0
0 0 0 0 0
```|
| 17 | 0.355 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 3 6 6
0 0 0 0 0
```|
| 18 | 0.157 | ❌ |<br>```
0 8 0 0 0
0 8 0 0 0
0 8 0 0 0
0 8 0 0 0
0 0 3 6 6
```|
| 19 | 0.232 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 3 6 6
```|
| 20 | 0.038 | ❌ |<br>```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
```|

## 🏗️ Aggregated (Solomonoff-weighted) Output

**Argmax Grid:**
```
0 0 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 0 6 6
0 0 0 0 0
```

## 🔎 Per-cell Probabilities (Top 2 Colors per Cell)

| Row | Col | Top-1 | Top-2 |
|-----|-----|-------|-------|
| 0 | 0 | 0:0.82 | 8:0.18 |
| 0 | 1 | 0:0.52 | 8:0.48 |
| 0 | 2 | 0:1.00 | 1:0.00 |
| 0 | 3 | 0:0.95 | 3:0.05 |
| 0 | 4 | 0:0.95 | 6:0.05 |
| 1 | 0 | 8:0.80 | 0:0.20 |
| 1 | 1 | 8:1.00 | 0:0.00 |
| 1 | 2 | 0:0.97 | 8:0.03 |
| 1 | 3 | 0:0.91 | 3:0.05 |
| 1 | 4 | 0:0.91 | 6:0.05 |
| 2 | 0 | 8:0.55 | 0:0.42 |
| 2 | 1 | 8:0.85 | 0:0.11 |
| 2 | 2 | 3:0.79 | 0:0.21 |
| 2 | 3 | 6:0.79 | 0:0.16 |
| 2 | 4 | 6:0.84 | 0:0.16 |
| 3 | 0 | 0:0.84 | 8:0.16 |
| 3 | 1 | 0:0.54 | 8:0.46 |
| 3 | 2 | 0:0.51 | 3:0.49 |
| 3 | 3 | 6:0.49 | 0:0.45 |
| 3 | 4 | 6:0.55 | 0:0.45 |
| 4 | 0 | 0:0.84 | 8:0.16 |
| 4 | 1 | 0:0.57 | 8:0.43 |
| 4 | 2 | 0:0.60 | 3:0.40 |
| 4 | 3 | 0:0.55 | 6:0.40 |
| 4 | 4 | 0:0.55 | 6:0.45 |