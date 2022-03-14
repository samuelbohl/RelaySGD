from cProfile import label
from math import log10
from turtle import title
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import minimum_spanning_tree
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from RelaySum.Sampler import AvgSampler
from RelaySum.Worker import Worker

def run_relaysum_experiment(num_workers, timesteps):
    # first we create a list of all workers
    workers = [Worker(i) for i in range(num_workers)]

    # generate a random network graph where each node represents a Worker
    p = 0.5
    g = erdos_renyi_graph(num_workers, p)

    # calculate the MST
    g_mst = minimum_spanning_tree(g)

    # add the neighbors to each Worker, so they know whom to communicate with
    for w_id in range(num_workers):
        for nb in g_mst.neighbors(w_id):
            workers[w_id].add_neighbor(workers[nb])

    # create the Sampler which keeps track of the true mean
    sampler = AvgSampler()
    errors = []
    cur_mse = 0
    for t in range(timesteps):
        # simulate each worker sequentially
        for worker in workers:
            worker.step(sampler)

        # calculate MSE
        cur_mse = mean_squared_error([worker.get_mean() for worker in workers], [sampler.get_mean() for worker in workers])
        errors.append(cur_mse)

    print('Estimted means:')
    print([worker.get_mean() for worker in workers])
    print('True mean:')
    print(sampler.get_mean())

    return errors

T = 400
errors = run_relaysum_experiment(10, T)
sns.lineplot(range(T), errors, label='10 Workers')

errors = run_relaysum_experiment(20, T)
sns.lineplot(range(T), errors, label='20 Workers')

errors = run_relaysum_experiment(40, T)
sns.lineplot(range(T), errors, label='40 Workers')

errors = run_relaysum_experiment(160, T)
sns.lineplot(range(T), errors, label='160 Workers')
plt.yscale('log')
plt.title('RelaySum for Distributed Mean Estimation')
plt.ylabel('MSE to true mean')
plt.xlabel('Steps')
plt.savefig('relaysum_dist_mean_est.png')