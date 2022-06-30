from __future__ import print_function
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
import bagua.torch_api as bagua
from sampler import DistributedHeterogeneousSampler

# python3 -m bagua.distributed.launch --nproc_per_node=8 benchmark.py --algorithm relay 2>&1 | tee ./logs/test.log

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.fuse_optimizer:
            optimizer.fuse_step()
        else:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Bagua CIFAR10 Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cifar10-vgg11",
        help="cifar10-vgg11, cifar100-resnet20",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="MOM",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        metavar="M",
        help="Learning rate step gamma (default: 0.9)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gradient_allreduce",
        help="gradient_allreduce, bytegrad, decentralized, low_precision_decentralized, qadam, async",
    )
    parser.add_argument(
        "--async-sync-interval",
        default=500,
        type=int,
        help="Model synchronization interval(ms) for async algorithm",
    )
    parser.add_argument(
        "--set-deterministic",
        action="store_true",
        default=False,
        help="set deterministic or not",
    )
    parser.add_argument(
        "--fuse-optimizer",
        action="store_true",
        default=False,
        help="fuse optimizer or not",
    )
    parser.add_argument(
        "--alpha",
        default=0.1,
        type=float,
        help="Alpha parameter for the non iid dirichlet distribution of the data",
    )
    parser.add_argument(
        "--topology",
        default="binary_tree",
        type=str,
        help="chain, binary_tree, random_binary_tree, double_binary_trees",
    )

    args = parser.parse_args()
    if args.set_deterministic:
        print("set_deterministic: True")
        np.random.seed(666)
        random.seed(666)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(666)
        torch.cuda.manual_seed_all(666 + int(bagua.get_rank()))
        torch.set_printoptions(precision=10)

    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
    if bagua.get_rank() == 0:
        logging.getLogger().setLevel(logging.INFO)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset_args = {"root":"./data", "train": True, "download": True, "transform": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])}

    test_dataset_args = {"root": "./data", "train": False, "transform": transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])}

    if bagua.get_local_rank() == 0:
        if args.experiment == "cifar10-vgg11":
            dataset1 = datasets.CIFAR10(**train_dataset_args)
        elif args.experiment == "cifar100-resnet20":
            dataset1 = datasets.CIFAR100(**train_dataset_args)
        else:
            raise NotImplementedError 
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        if args.experiment == "cifar10-vgg11":
            dataset1 = datasets.CIFAR10(**train_dataset_args)
        elif args.experiment == "cifar100-resnet20":
            dataset1 = datasets.CIFAR100(**train_dataset_args)
        else:
            raise NotImplementedError 

    
    if args.experiment == "cifar10-vgg11":
        dataset2 = datasets.CIFAR10(**test_dataset_args)
    elif args.experiment == "cifar100-resnet20":
        dataset2 = datasets.CIFAR100(**test_dataset_args)
    else:
        raise NotImplementedError 

    train_sampler = DistributedHeterogeneousSampler(
        dataset=dataset1, num_workers=bagua.get_world_size(), rank=bagua.get_local_rank(), alpha=args.alpha
    )

    train_kwargs.update(
        {
            "sampler": train_sampler,
            "batch_size": args.batch_size // bagua.get_world_size(),
            "shuffle": False,
        }
    )
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.experiment == "cifar10-vgg11":
        from models.vgg import vgg11
        model = vgg11().cuda()
    elif args.experiment == "cifar100-resnet20":
        from models.resnet import ResNet20
        model = ResNet20().cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    if args.algorithm == "gradient_allreduce":
        from bagua.torch_api.algorithms import gradient_allreduce

        algorithm = gradient_allreduce.GradientAllReduceAlgorithm()
    elif args.algorithm == "decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.DecentralizedAlgorithm()
    elif args.algorithm == "low_precision_decentralized":
        from bagua.torch_api.algorithms import decentralized

        algorithm = decentralized.LowPrecisionDecentralizedAlgorithm()
    elif args.algorithm == "bytegrad":
        from bagua.torch_api.algorithms import bytegrad

        algorithm = bytegrad.ByteGradAlgorithm()
    elif args.algorithm == "qadam":
        from bagua.torch_api.algorithms import q_adam

        optimizer = q_adam.QAdamOptimizer(
            model.parameters(), lr=args.lr, warmup_steps=100
        )
        algorithm = q_adam.QAdamAlgorithm(optimizer)
    elif args.algorithm == "async":
        from bagua.torch_api.algorithms import async_model_average

        algorithm = async_model_average.AsyncModelAverageAlgorithm(
            sync_interval_ms=args.async_sync_interval,
        )
    elif args.algorithm == "relay":
        from relay import RelayAlgorithm

        algorithm = RelayAlgorithm(optimizer=optimizer, topology=args.topology)
    elif args.algorithm == "allreduce":
        from allreduce import AllreduceAlgorithm
        
        algorithm = AllreduceAlgorithm(optimizer=optimizer)
    else:
        raise NotImplementedError

    
    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=not args.fuse_optimizer,
    )

    if args.fuse_optimizer:
        optimizer = bagua.contrib.fuse_optimizer(optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    import time
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if args.algorithm == "async":
            model.bagua_algorithm.resume(model)

        train(args, model, train_loader, optimizer, epoch)

        if args.algorithm == "async":
            model.bagua_algorithm.abort(model)

        if args.algorithm == "relay" and "random" in args.topology and epoch > 70 and epoch % 10 == 1:
            if bagua.get_local_rank() == 0: logging.info('REBUILDING TREE')
            model.bagua_algorithm.rebuild_tree()

        test(model, test_loader)
        scheduler.step()
        end = time.time()
        if bagua.get_local_rank() == 0: logging.info('Epoch time: {}'.format(end-start))
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    


if __name__ == "__main__":
    main()
