from torch.utils.data import Dataset
import torch
from pathlib import Path
import h5py
from torch.nn.utils.rnn import pad_sequence


class DyckNDataset(Dataset):
    def __init__(self, dataset, path, transform=None, device='cpu', timescales=False):
        super(Dataset, self).__init__()
        self.transform = transform
        self.device = device
        self.dataset = dataset
        self.path = path
        self.parenthesis = [('(', ')'), ('[', ']'), ('{', '}'), ('|', '!'),
                            ('A', 'a'), ('B', 'b'), ('C', 'c'), ('D', 'd')]
        self.vocabulary = {pair[0]: 2*i for i, pair in enumerate(self.parenthesis)}
        self.vocabulary.update({pair[1]: 2*i+1 for i, pair in enumerate(self.parenthesis)})
        self.vocabulary['.'] = 2*len(self.parenthesis)
        self.timescales = timescales

        if dataset == 'train':
            file = path + '/train.h5'
        elif dataset == 'validation':
            file = path + '/validation.h5'
        elif dataset == 'test':
            file = path + '/test.h5'
        else:
            raise ValueError(f'Wrong dataset type {dataset}')

        p = Path(file)
        if not p.exists():
            raise ValueError(f'Invalid file {file}')

        with h5py.File(file, 'r') as f:
            self.N = f['type'][()]
            self.p = f['p'][()]
            self.q = f['q'][()]
            data = f['data']
            labels = f['labels']
            depths = f['depths']
            ts = f['timescales']
            self.binary_fmt = '{:0' + str(self.N) + 'b}'
            print('Parameters for parenthesis dataset:', self.N, self.p, self.q)
            self.data = []
            self.labels = []
            if timescales:
                self.timescales = []
            all_inputs = set()
            for i in data:
                all_inputs = all_inputs | set(data[str(i)][()])
                self.data.append(self._convert_parenthesis(data[str(i)][()]))
                self.labels.append(self._convert_to_binary(labels[str(i)][()]))
                if timescales:
                    self.timescales.append(torch.tensor(ts[str(i)][()]))
            print(all_inputs)
        self.vocabulary = {e: self.vocabulary[e] for e in all_inputs}
        print(self.vocabulary)
        self.samples = len(self.data)

    def _convert_parenthesis(self, sequence):
        new_sequence = [self.vocabulary[char] for char in sequence]
        return torch.tensor(new_sequence, device=self.device)

    def _convert_to_binary(self, sequence):
        new_sequence = [[int(i) for i in list(self.binary_fmt.format(elem))] for elem in sequence]
        return torch.tensor(new_sequence, device=self.device, dtype=torch.float)

    def __len__(self):
        return self.samples

    def __getitem__(self, item):
        sample = {
            'input': self.data[item],
            'output': self.labels[item],
        }
        if self.timescales:
            sample['timescales'] = self.timescales[item]

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample


class SequenceCollator:
    def __init__(self, dim=0, batch_first=False, device='cpu'):
        self.dim = dim
        self.batch_first = batch_first
        self.device = device

    def seq_collate_fn(self, batch):
        batch_size = len(batch)
        fields = list(batch[0].keys())
        lengths = torch.tensor([len(elem[fields[0]]) for elem in batch], device=self.device)
        order = lengths.argsort(descending=True)
        new_batch = {
            'lengths': lengths[order],
            'order': order,
        }
        for field in fields:
            new_tensor = pad_sequence([batch[order[i]][field] for i in range(len(batch))], batch_first=self.batch_first)
            new_batch[field] = new_tensor
        return new_batch
