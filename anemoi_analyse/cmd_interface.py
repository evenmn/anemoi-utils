import argparse
import matplotlib.pyplot as plt

from loss_plotter import read_data, get_total_iteration


def get_parser():
    parser = argparse.ArgumentParser(description="Plot loss")
    parser.add_argument('logs', type=str, help='Logs to analyse', nargs='*')
    parser.add_argument('-x', type=str, default='train_mse_step', help='x variable')
    parser.add_argument('-y', type=str, default='total_iteration', help='y variable')
    parser.add_argument('-a', '--average', type=int, default=None, help='Running average using second order Savitzky-Golay filter')
    parser.add_argument('--list-variables', action='store_true', default=False, help='List available variables')
    parser.add_argument('--logx', action='store_true', default=False, help='Logarithmic x-axis')
    parser.add_argument('--logy', action='store_true', default=False, help='Logarithmic y-axis')
    return parser


def run():
    args = get_parser().parse_args()

    xs = []
    ys = []
    all_keys = []
    for log in args.logs:
        data = read_data(log)
        
        keys = set(data.keys())
        if args.list_variables:
            if len(args.logs) == 1:
                print(keys)
                exit()
            all_keys.append(keys)
        else:
            assert args.x in keys, "x variable not found"
            assert args.y in keys, "y variable not found"

            data['total_iteration'] = get_total_iteration(data)
            xs.append(data[args.x])
            ys.append(data[args.y])

    if args.list_variables:
        common_set = set(all_keys[0])
        for s in all_keys[1:]:
            common_set.intersection_update(s)
        print(common_set)
        exit()

    labels = args.logs

    for x, y, label in zip(xs, ys, labels):
        if args.average is not None:
            x = savgol_filter(x, args.average, 2)
        plt.plot(x, y, label=label)
    plt.legend(loc='best')
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    if args.logx:
        plt.xscale('log')
    if args.logy:
        plt.yscale('log')
    plt.show()
