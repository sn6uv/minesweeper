"""Utility for plotting win rate from logs"""
import matplotlib.pyplot as plt
import dateutil.parser


def read_logs(filename):
    ts, ys, ls = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if 'Win rate:' in line:
                parts = line.split()
                t, y = parts[0:2], parts[-1]
                t = dateutil.parser.parse(' '.join(t))
                y = float(y.rstrip('%'))
                ts.append(t)
                ys.append(y)
                seen_l = False
            if 'Testing: ' in line and not seen_l:
                l = float(line.split()[2])
                ls.append(l)
                seen_l = True

    # Remove outlier
    ls[0] = None

    # Convert times to duration in hours
    t0 = ts[0]
    for i, t in enumerate(ts):
        ts[i] = (t - t0).total_seconds() / 3600
    return ts, ys, ls
                

def plot_win_rate(t, y, l):
    fig = plt.figure()

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('win rate (%)', color=color)
    ax1.plot(t, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(t, l, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training progress')
    fig.tight_layout()
    plt.show()


def main():
    t, y, l = read_logs('log.txt')
    plot_win_rate(t, y, l)

if __name__ == '__main__':
    main()
