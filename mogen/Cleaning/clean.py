

def update_cast_joint_errors(q, limits, eps=1e-6):
    below_lower = q < limits[:, 0]
    above_upper = q > limits[:, 1]

    q[below_lower] += eps
    q[above_upper] -= eps
    return q