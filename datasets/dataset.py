import random
import numpy as np
from torch.utils.data.dataset import Dataset
from config.config import cfg

class MultipleDatasets(Dataset):
    def __init__(self, 
                 dbs,
                 partition,
                 make_same_len=True, 
                 total_len=None, 
                 verbose=False):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len
        # self.partition = partition
        self.partition = {k: v for k, v in sorted(partition.items(), key=lambda item: item[1])}
        self.dataset = {}
        for db in dbs:
            self.dataset.update({db.__class__.__name__: db})

        if verbose:
            print('datasets:', [len(self.dbs[i]) for i in range(self.db_num)])
            print(
                f'Sample Ratio: {self.partition}')

    def __len__(self):
        return self.max_db_data_num

    def __getitem__(self, index):
        p = np.random.rand()
        v = list(self.partition.values())
        k = list(self.partition.keys())
        for i,v_i in enumerate(v):
            if p<=v_i:
                return self.dataset[k[i]][index % len(self.dataset[k[i]])]
