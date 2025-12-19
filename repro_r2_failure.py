import numpy as np
import pandas as pd

# Mock data
y_true = np.array([0.01, 0.02, 0.03])
X = np.array([1.0, 1.0, 1.0]) # Linear predictor
beta = 0.5 # DIM_0_LN coefficient (exp(alpha))

# y_pred_true = beta * X = 0.5
# y_pred_old = exp(beta * X) = exp(0.5) = 1.648

y_pred_old = np.exp(beta * X)
y_pred_new = beta * X

def calc_r2(y, y_p):
    ss_res = np.sum((y - y_p)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

print(f"y_true: {y_true}")
print(f"y_pred_old: {y_pred_old}")
print(f"y_pred_new: {y_pred_new}")

r2_old = calc_r2(y_true, y_pred_old)
r2_new = calc_r2(y_true, y_pred_new)

print(f"R2 (Old): {r2_old:.4f}")
print(f"R2 (New): {r2_new:.4f}")

# What if beta is larger?
beta = 5.0
y_pred_old = np.exp(beta * X) # exp(5) = 148
y_pred_new = beta * X # 5
r2_old = calc_r2(y_true, y_pred_old)
print(f"Unconstrained scale - R2 (Old) with beta=5: {r2_old:.4f}")
