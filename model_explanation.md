# Model Formula and Monotonicity Constraints

This document explains the mathematical formulation of the nonlinear fitting model and how monotonicity constraints are enforced for 1D and 2D parameters.

## 1. Model Overview

The model predicts a response variable $Y$ based on a set of risk factors (parameters). The relationship is modeled as:

$$ Y = \exp\left( \sum\_{k} f_k(X_k) \right) \cdot \epsilon $$

Taking the logarithm, we get a linear additive model:

$$ \ln(Y) = \sum\_{k} f_k(X_k) + \ln(\epsilon) $$

where $f_k(X_k)$ is the contribution of the $k$-th risk factor. The function $f_k$ can be:

- **DIM_0**: A linear term $f(x) = \beta \cdot x$.
- **DIM_1**: A piecewise linear function (spline) of one variable.
- **DIM_2**: A piecewise bilinear surface of two variables.

## 2. One-Dimensional Monotonicity (DIM_1)

For a 1D parameter with knots $k_0, k_1, \dots, k_n$, the function values at the knots are $v_0, v_1, \dots, v_n$. The function is interpolated linearly between knots.

### Monotonic Increasing

To enforce that the function is monotonically increasing ($v_{i+1} \ge v_i$), we parameterize the values using non-negative increments (deltas):

$$ v*0 = P_0 $$
$$ v_i = v*{i-1} + \delta_i \quad \text{for } i=1 \dots n $$

where $\delta_i \ge 0$.

In the optimization, we solve for $P_0$ (unbounded) and $\delta_1, \dots, \delta_n$ (lower bound 0).

### Monotonic Decreasing

Similarly, for monotonically decreasing functions:

$$ v*i = v*{i-1} - \delta_i \quad \text{for } i=1 \dots n $$

where $\delta_i \ge 0$.

## 3. Two-Dimensional Monotonicity (DIM_2)

For a 2D parameter with knots $u_0, \dots, u_R$ (rows) and $v_0, \dots, v_C$ (columns), we define a grid of values $Z_{i,j}$.

We want to enforce monotonicity in both dimensions. For example, "Increasing/Increasing" means:

- $Z_{i+1, j} \ge Z_{i, j}$ (Increasing in U)
- $Z_{i, j+1} \ge Z_{i, j}$ (Increasing in V)

This is achieved using a parameterization similar to **Isotonic Regression** on a grid.

### Parameterization

We define the grid values $Z_{i,j}$ using a set of non-negative parameters:

- $z_{00}$: Base value at origin (unbounded).
- $d^u_i$: Increments along the first column ($i=0 \dots R-2$).
- $d^v_j$: Increments along the first row ($j=0 \dots C-2$).
- $d^{int}_k$: Interaction increments for internal points.

The reconstruction logic is:

1.  **Origin**:
    $$ Z*{0,0} = z*{00} $$

2.  **First Column (U-edge)**:
    $$ Z*{i+1, 0} = Z*{i, 0} + d^u_i, \quad d^u_i \ge 0 $$

3.  **First Row (V-edge)**:
    $$ Z*{0, j+1} = Z*{0, j} + d^v_j, \quad d^v_j \ge 0 $$

4.  **Internal Points**:
    For $i > 0, j > 0$, the value $Z_{i,j}$ must be greater than or equal to both its left neighbor $Z_{i, j-1}$ and its upper neighbor $Z_{i-1, j}$.

    $$ Z*{i, j} = \max(Z*{i-1, j}, Z*{i, j-1}) + d^{int}*{k}, \quad d^{int}\_{k} \ge 0 $$

This construction guarantees that every point is greater than or equal to its predecessors in both directions, ensuring global monotonicity on the grid.

### Other Directions

For other monotonicity combinations (e.g., Increasing/Decreasing), the grid is flipped internally before applying the constraints, and then flipped back.

- **Inc/Dec**: Flip V-axis, apply Inc/Inc, flip back.
- **Dec/Inc**: Flip U-axis, apply Inc/Inc, flip back.
- **Dec/Dec**: Flip both axes, apply Inc/Inc, flip back.
