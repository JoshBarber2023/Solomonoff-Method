# 🧮 Leave-One-Out Solomonoff Analysis for `1_3_5th_go_down_2_4th_go_up_l6abuzscmjgg12f0mdc`

**Held-out example index (1-based):** 3


## 📊 Ranked Hypotheses

| Rank | Hypothesis                                                                                                                                                                                                             | Simplicity | Train Acc. | Solomonoff Score | Solomonoff Weight |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | ---------------- | ----------------- |
| 1    | Different cells in the input grid seem to replicate in the vertical direction across various rows in the output grid while maintaining their column locations.                                                         | 0.513      | 0.620      | 0.318            | 0.245             |
| 2    | Objects in the input grid are propagated in the vertical direction, extending from their original positions both upwards and downwards till they hit another object or the boundary of the grid.                       | 0.432      | 0.540      | 0.233            | 0.179             |
| 3    | Objects are duplicated vertically, creating a column with the same object from top to bottom of the grid. The number of duplications of the object is equal to the height of the grid.                                 | 0.372      | 0.620      | 0.231            | 0.178             |
| 4    | Objects in the grid vertically expand or stretch until reaching either the top or bottom of the grid while maintaining their original positions and other properties.                                                  | 0.402      | 0.560      | 0.225            | 0.173             |
| 5    | The transformation consists on duplicating each object typically from top to bottom to multiple positions around the grid.                                                                                             | 0.472      | 0.420      | 0.198            | 0.152             |
| 6    | Objects are translated from their original position to fill the grid vertically, resulting in each unique object being repeated in a straight vertical line. Each object keeps its original color during this process. | 0.206      | 0.460      | 0.095            | 0.073             |

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
| 1 | 0.095 | ❌ |<br>```
8 0 3 6 6
8 0 3 6 6
8 0 3 6 6
8 0 3 6 6
8 0 3 6 6
```|
| 2 | 0.231 | ❌ |<br>```
0 8 8 0 0
0 8 8 0 0
0 8 3 6 6
0 8 3 6 6
0 8 3 6 6
```|
| 3 | 0.318 | ❌ |<br>```
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
8 8 3 6 6
```|
| 4 | 0.233 | ❌ |<br>```
8 8 0 0 0
8 8 0 0 0
0 0 3 6 6
0 0 3 6 6
0 0 0 0 0
```|
| 5 | 0.198 | ❌ |<br>```
8 8 0 0 0
8 8 0 0 0
8 8 3 6 6
0 0 3 6 6
0 0 0 0 0
```|
| 6 | 0.225 | ❌ |<br>```
0 8 0 0 0
0 8 0 0 0
0 0 3 6 6
0 0 3 6 6
0 0 3 6 6
```|

## 🏗️ Aggregated (Solomonoff-weighted) Output

**Argmax Grid:**
```
8 8 0 0 0
8 8 0 0 0
0 8 3 6 6
0 0 3 6 6
0 0 3 6 6
```

## 🔎 Per-cell Probabilities (Top 2 Colors per Cell)

| Row | Col | Top-1 | Top-2 |
|-----|-----|-------|-------|
| 0 | 0 | 8:0.65 | 0:0.35 |
| 0 | 1 | 8:0.93 | 0:0.07 |
| 0 | 2 | 0:0.51 | 3:0.32 |
| 0 | 3 | 0:0.68 | 6:0.32 |
| 0 | 4 | 0:0.68 | 6:0.32 |
| 1 | 0 | 8:0.65 | 0:0.35 |
| 1 | 1 | 8:0.93 | 0:0.07 |
| 1 | 2 | 0:0.51 | 3:0.32 |
| 1 | 3 | 0:0.68 | 6:0.32 |
| 1 | 4 | 0:0.68 | 6:0.32 |
| 2 | 0 | 0:0.53 | 8:0.47 |
| 2 | 1 | 8:0.57 | 0:0.43 |
| 2 | 2 | 3:1.00 | 0:0.00 |
| 2 | 3 | 6:1.00 | 0:0.00 |
| 2 | 4 | 6:1.00 | 0:0.00 |
| 3 | 0 | 0:0.68 | 8:0.32 |
| 3 | 1 | 0:0.58 | 8:0.42 |
| 3 | 2 | 3:1.00 | 0:0.00 |
| 3 | 3 | 6:1.00 | 0:0.00 |
| 3 | 4 | 6:1.00 | 0:0.00 |
| 4 | 0 | 0:0.68 | 8:0.32 |
| 4 | 1 | 0:0.58 | 8:0.42 |
| 4 | 2 | 3:0.67 | 0:0.33 |
| 4 | 3 | 6:0.67 | 0:0.33 |
| 4 | 4 | 6:0.67 | 0:0.33 |