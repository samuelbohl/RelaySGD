import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import minimum_spanning_tree
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from RelaySGD.Worker import Worker

def run_relaysgd_experiment(num_workers, batch_size, timesteps):
    # first we generate some random data for the function y = 3x + 4
    X = 2 * np.random.rand(batch_size * num_workers, 1)
    y = 4 + 3 * X + np.random.randn(batch_size * num_workers, 1)

    # define the learning rate
    lr = 0.001

    # then we create a list of all the workers
    workers = []
    for i in range(num_workers):
        X_temp = np.c_[np.ones((batch_size,1)), X[i * batch_size : ((i + 1) * batch_size)]]
        y_temp = y[i * batch_size : ((i + 1) * batch_size)]
        theta_temp = np.random.randn(2,1)
        workers.append(Worker(i, lr, X_temp, y_temp, theta_temp))


    # generate a random network graph where each node represents a Worker
    p = 0.5
    g = erdos_renyi_graph(num_workers, p)

    # calculate the MST
    g_mst = minimum_spanning_tree(g)

    # add the neighbors to each Worker, so they know whom to communicate with
    for w_id in range(num_workers):
        for nb in g_mst.neighbors(w_id):
            workers[w_id].add_neighbor(workers[nb])

    # run the experiment timeloop
    errors = []
    cur_mse = 0
    for t in range(timesteps):
        # simulate each worker sequentially
        for worker in workers:
            worker.step()

        # calculate MSE
        error = 0
        X_b = np.c_[np.ones((len(X),1)),X]
        for worker in workers:
            predictions = X_b.dot(worker.get_theta())
            error += (1/len(X)) * np.sum(np.square(predictions-y))
        cur_mse = error / len(workers)
        errors.append(cur_mse)

    print(workers[0].get_theta())

    return errors

T = 1000 # timesteps
errors = run_relaysgd_experiment(10, 100, T)
sns.lineplot(range(T), errors, label='10 Workers')

errors = run_relaysgd_experiment(20, 100, T)
sns.lineplot(range(T), errors, label='20 Workers')

errors = run_relaysgd_experiment(40, 100, T)
sns.lineplot(range(T), errors, label='40 Workers')


plt.yscale('log')
plt.title('RelaySGD')
plt.ylabel('MSE')
plt.xlabel('Steps')
plt.savefig('relaysgd.png')