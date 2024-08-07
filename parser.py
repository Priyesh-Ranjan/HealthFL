import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of models")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=11)
    parser.add_argument("-n", "--num_clients", type=int, default=10)
    parser.add_argument("--output_folder", type=str, default="experiments",
                        help="path to output folder, e.g. \"experiment\"")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar"], default="mnist")
    parser.add_argument("--loader_type", type=str, choices=["iid", "byLabel", "dirichlet"], default="iid")
    parser.add_argument("--loader_path", type=str, default="./data/loader.pk", help="where to save the data partitions")
    parser.add_argument("--AR", type=str, )
    #parser.add_argument("--n_attacker_backdoor", type=int, default=0)
    parser.add_argument("--attacks", type=str, help="if contains \"backdoor\", activate the corresponding tests")
    #parser.add_argument("--save_model_weights", action="store_true")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cuda')
    parser.add_argument("--inner_epochs", type=int, default=1)

    args = parser.parse_args()

    n = args.num_clients

    #m = args.n_attacker_backdoor
    #args.attacker_list_backdoor = np.random.permutation(list(range(n)))[:m]

    if args.experiment_name == None:
        args.experiment_name = f"{args.loader_type}/{args.attacks}/{args.AR}"

    return args


if __name__ == "__main__":

    import _main

    args = parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)
