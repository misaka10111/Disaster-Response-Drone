import numpy as np


def move_robot_toward_target(start, target, step_size=0.5):
    x, y = start
    tx, ty = target

    path = [(x, y)]
    for _ in range(600):
        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)
        if dist < 0.3:
            break

        x += step_size * dx / dist
        y += step_size * dy / dist
        path.append((x, y))

    return path


def move_robot_toward_multiple_targets(start, targets, step_size=0.5):
    """
    Visit all targets in order:
    start → person1 → person2 → ...
    """
    path = []
    current = start

    for tgt in targets:
        segment = move_robot_toward_target(current, tgt, step_size)
        path.extend(segment)
        current = tgt

    return path
