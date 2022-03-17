import numpy as np

class Worker:
    def __init__(self, id: int, lr: float, X, y, theta):
        self.id = id                    # worker id
        self.neighbors = []             # neighbors in network
        self.inc_m = []                 # incomming message
        self.inc_c = []                 # incomming counts
        self.out_m = np.array([])       # outgoing message
        self.out_c = 0                  # outgoing count
        self.recv_m = []                # receiver buffer messages
        self.recv_c = []                # reviever buffer counts
        self.X = X                      # worker sample data
        self.y = y                      # worker sample data
        self.lr = lr                    # learning rate
        self.theta = theta              # function to learn {x^{t}}

    # adds a new neighbor to this worker
    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    # returns the outgoing message of the source
    def recieve_m(self, source):
        return source.get_out_m()

    # returns the outgoing count of the source
    def recieve_c(self, source):
        return source.get_out_c()

    # returns current outgoing message (m^{t-1})
    def get_out_m(self):
        return self.out_m

    # returns current outgoing count (c^{t-1})
    def get_out_c(self):
        return self.out_c

    # all neighbors can receive this message with recv_m
    def send_m(self, value):
        self.out_m = value

    # all neighbors can receive this message with recv_m
    def send_c(self, value):
        self.out_c = value

    # computes one timestep of RelaySGD
    def step(self):
        # select random samples for SGD
        ind = [np.random.randint(0,len(self.X)) for i in range(len(self.y))]
        X_r = np.array([self.X[i] for i in ind])
        y_r = np.array([self.y[i] for i in ind])

        # calculate current prediction
        pred = np.dot(X_r, self.theta)

        # update gradient (x^{t + 1/2})
        self.theta = self.theta - self.lr * X_r.T.dot((pred - y_r))

        # send m^{t} => current theta (x^{t + 1/2}) plus recieved messages (m^{t-1}) to all neighbors
        recv_sum = sum(self.recv_m)
        self.send_m(self.theta + recv_sum)
        # send c^{t} => 1 + recieved count (c^{t-1}) to all neighbors
        recv_count = sum(self.recv_c)
        self.send_c(1 + recv_count)

        # reset receiver buffer
        self.inc_m = []
        self.inc_c = []

        # recieve m^{t} and c^{t} from all neighbors
        for nj in self.neighbors:
            self.inc_m.append(self.recieve_m(nj))
            self.inc_c.append(self.recieve_c(nj))

        # update sum of counts (n^{t+1})
        sum_counts = 1 + sum(self.inc_c)
        # update theta (x^{t+1})
        inc_sum = sum(self.inc_m)
        if inc_sum.shape == (2,1):
            self.theta = (1 / sum_counts) * (self.theta + inc_sum)

    # returns current theta estimation
    def get_theta(self):
        return self.theta
