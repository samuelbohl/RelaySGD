class Worker:
    def __init__(self, id: int):
        self.id = id
        self.neighbors = []
        self.inc_m = []                 # incomming means
        self.inc_c = []                 # incomming counts
        self.out_m = 0
        self.out_c = 0
        self.recv_m = []
        self.recv_c = []
        self.x = 0
        self.cur_sample = 0
        self.sum_samples = 0
        self.sum_counts = 0

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def recieve_m(self, source):
        return source.get_out_m()

    def recieve_c(self, source):
        return source.get_out_c()

    # returns outgoing mean (m^{t-1})
    def get_out_m(self):
        return self.out_m

    # returns outgoing count (c^{t-1})
    def get_out_c(self):
        return self.out_c

    # all neighbors can receive this message with recv_m
    def send_m(self, value):
        self.out_m = value

    # all neighbors can receive this message with recv_m
    def send_c(self, value):
        self.out_c = value

    def step(self, sampler):
        # get sample
        self.cur_sample = sampler.sample()

        # send m^{t} => sum of current sample (d^{t}) plus recieved messages (m^{t-1}) to all neighbors
        recv_sum = sum(self.recv_m)
        self.send_m(self.cur_sample + recv_sum)
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

        # update sum of samples (y^{t+1})
        self.sum_samples += self.cur_sample + sum(self.inc_m)
        # update sum of counts (s^{t+1})
        self.sum_counts += 1 + sum(self.inc_c)

        # avg estimate (x^t)
        self.x = self.sum_samples / self.sum_counts
    
    def get_mean(self):
        return self.x
