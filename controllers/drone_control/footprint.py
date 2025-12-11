import matplotlib.animation as animation


def animate_uav_path(ax, path, interval=80):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    (line,) = ax.plot([], [], color="cyan", linewidth=2)
    uav_icon, = ax.plot([], [], marker="o", markersize=10, color="blue")

    def init():
        line.set_data([], [])
        uav_icon.set_data([], [])
        return line, uav_icon

    def update(i):
        line.set_data(xs[:i], ys[:i])
        if i > 0:
            uav_icon.set_data([xs[i]], [ys[i]])
        return line, uav_icon

    ani = animation.FuncAnimation(
        ax.figure,
        update,
        frames=len(path),
        init_func=init,
        interval=interval,
        blit=False
    )

    return ani
