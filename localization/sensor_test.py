import numpy as np

def build_sensor_table(W=6, sigma_hit=1.0,
                       alpha_hit=0.74, alpha_short=0.07,
                       alpha_max=0.07, alpha_rand=0.12):
    z_max = W - 1

    z = np.arange(W, dtype=np.float64)[:, None]   # (W,1)
    d = np.arange(W, dtype=np.float64)[None, :]   # (1,W)

    z_grid = np.broadcast_to(z, (W, W))
    d_grid = np.broadcast_to(d, (W, W))

    # p_hit
    p_hit = np.exp(-((z_grid - d_grid) ** 2) / (2.0 * sigma_hit**2))
    p_hit = p_hit / np.sum(p_hit, axis=0, keepdims=True)

    # p_short
    p_short = np.zeros((W, W), dtype=np.float64)
    valid_short = (z_grid <= d_grid) & (d_grid > 0)
    p_short[valid_short] = (2.0 / d_grid[valid_short]) * (
        1.0 - z_grid[valid_short] / d_grid[valid_short]
    )

    # p_max
    p_max = np.zeros((W, W), dtype=np.float64)
    p_max[z_grid == z_max] = 1.0

    # p_rand
    p_rand = np.zeros((W, W), dtype=np.float64)
    p_rand[z_grid < z_max] = 1.0 / z_max

    table = (
        alpha_hit * p_hit +
        alpha_short * p_short +
        alpha_max * p_max +
        alpha_rand * p_rand
    )

    return table, p_hit, p_short, p_max, p_rand

table, p_hit, p_short, p_max, p_rand = build_sensor_table()

np.set_printoptions(precision=4, suppress=True)
print("p_hit:\n", p_hit)
print("p_short:\n", p_short)
print("p_max:\n", p_max)
print("p_rand:\n", p_rand)
print("final table:\n", table)
print("column sums of final table:", np.sum(table, axis=0))
