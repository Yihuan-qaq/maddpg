import torch


class args_parser():
    def __init__(self, load_dict=None):
        if load_dict != None:
            self.algorithm = load_dict["algorithm"]
            self.is_layered = load_dict["is_layered"]
            self.is_random_edge = load_dict["is_random_edge"]
            self.is_random_weight = load_dict["is_random_weight"]
            self.max_episodes = load_dict["max_episodes"]
        else:
            self.algorithm = "W_avg"
            self.is_layered = 1
            self.is_random_edge = 0
            self.is_random_weight = 0

        self.model = "lenet"
        self.batch_size = 10
        self.max_ep_step = 100
        self.num_iteration = 120
        self.num_edge_aggregation = 1
        self.num_communication = 1
        self.data_distribution = 0

        self.num_clients = 24
        self.num_edges = 6

        self.lr = 0.01
        self.lr_decay = 0.995
        self.lr_decay_epoch = 1
        self.momentum = 0
        self.weight_decay = 0

        self.gamma = 0.9
        self.rl_batch_size = 32
        self.memory_capacity = 10000
        self.TAU = 0.01

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
