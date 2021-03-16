from datasets import SequenceCollator
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

# from thalamus.experiments.lm import LanguageModelingExperiment
def padded_all_same_metric(scores, targets, lengths, threshold=0.5):
    batch_size = len(lengths)
    total_metric = 0
    for i in range(batch_size):
        elem_metric = torch.all((scores[i][:lengths[i]] > threshold) == targets[i][:lengths[i]])
        total_metric += elem_metric
    # print('Total Metric for this Batch:', total_metric.item())
    return total_metric.item()


class DyckNExperiment:
    """Experiment on Learning Dyck-n formal language"""

    def __init__(self, dataset, model, optimizer, seq_length=200, metric=padded_all_same_metric,
                 name='DyckNExperiment', dataset_parameters=None,
                 batch_size=32, lr=1e-3, max_epochs=20, clip_gradients=None,
                 checkpoint_directory=None, device='cpu', print_steps=500,
                 load_checkpoint_epoch=None, save_every=None):
        self.name = name
        self.device = device
        self.model = model
        self.optimizer_type = optimizer
        self.dataset = dataset
        self.dataset_parameters = dataset_parameters
        self.batch_first = True
        self.seq_length = seq_length
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.clip_gradients = clip_gradients
        self.metric = metric

        self.checkpoint_directory = checkpoint_directory
        self.load_checkpoint_epoch = load_checkpoint_epoch
        self.save_every = save_every
        if self.checkpoint_directory is not None:
            if self.save_every is None:
                self.save_every = 1 if max_epochs == 1 else max_epochs - 1
            else:
                self.save_every = save_every
        else:
            # Disable saving by setting save_every beyond number of epochs
            self.save_every = max_epochs + 1

        if self.optimizer_type is None:
            raise ValueError('Optimizer needs to be defined')
        if self.model is None:
            raise ValueError('Model needs to be defined')
        if self.dataset is None:
            raise ValueError('Dataset needs to be defined')

        self.print_steps = print_steps
        self.loss_function = None
        self.optimizer = None
        self.epoch = 0
        self.total_params = 0

    def load_data(self, subset):
        data_instance = self.dataset(subset, device=self.device, **self.dataset_parameters)
        return data_instance

    def model_setup(self):
        self.model.to(self.device)
        self.loss_function = nn.MSELoss(reduction='sum').to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr)

    def save_checkpoint(self, state):
        if self.checkpoint_directory is None:
            return
        torch.save(state, self.checkpoint_directory + '/' + self.name +
                   '_checkpoint_epoch_' + str(self.epoch) + '.tar')

    def load_checkpoint(self, file):
        state = torch.load(file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])
        self.epoch = state['epoch'] + 1  # start at the next epoch
        self.clip_gradients = state['clip_val']
        self.metric = state['metric']
        return state['results']

    def save_model(self, directory):
        torch.save({
            'model': self.model.state_dict(),
            'loss': self.loss_function,
        }, directory + '/' + self.name + '_model_weights.pt')

    def load_model(self, file):
        state = torch.load(file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state['model'])
        self.loss_function = state['loss']

    def _compute_varlen_loss(self, scores, targets, lengths):
        batch_size = len(lengths)
        N = targets.size(2)
        total_loss = torch.tensor(0.0, device=self.device)
        total_elems = torch.sum(lengths)
        for i in range(batch_size):
            elem_loss = self.loss_function(scores[i][:lengths[i]], targets[i][:lengths[i]])
            total_loss += elem_loss / N
        total_loss = total_loss / total_elems
        return total_loss.item()

    def eval_model(self, dataloader):
        total_metric = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                lengths = sample_batched['lengths']
                batch_size = len(lengths)
                N = targets.size(2)
                # Predict for this batch
                scores, _, lengths_new = self.model(source, lengths)
                # NOTE: Softmax missing at the output of the network
                metric_value = self.metric(scores, targets, lengths)
                val_loss = self._compute_varlen_loss(scores, targets, lengths)
                total_loss += val_loss
                total_metric += metric_value
                total_items += batch_size
                total_batches += 1
        return total_metric / total_items, total_items, total_loss / total_batches

    def detach_hidden_state(self, state):
        if isinstance(state, torch.Tensor):
            state = state.detach()
        elif isinstance(state, tuple):
            state = tuple(self.detach_hidden_state(s) for s in state)
        elif isinstance(state, list):
            state = [self.detach_hidden_state(s) for s in state]
        else:
            raise TypeError(f'unknown type {state}')
        return state

    def train_model(self, test_last=0):
        dataset_train = self.load_data("train")
        collator = SequenceCollator(batch_first=self.batch_first, device=self.device)
        self.data_train = dataset_train
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                      collate_fn=collator.seq_collate_fn)

        dataset_val = self.load_data("validation")
        self.data_val = dataset_val
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                    collate_fn=collator.seq_collate_fn)

        self.model_setup()
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.load_checkpoint_epoch:
            results = self.load_checkpoint(self.checkpoint_directory + '/' +
                                           self.name + '_checkpoint_epoch_' +
                                           str(self.load_checkpoint_epoch) + '.tar')
            max_val_metric = max(results['val_metric'])
        else:
            results = {
                'val_metric': [],
                'val_loss': [],
                'train_loss': [],
                'test_metric': [],
            }
            max_val_metric = 0.0
        print("Seq Length: ", self.seq_length)
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            self.model.train()
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader_train):
                source, targets = sample_batched['input'], sample_batched['output']
                lengths = sample_batched['lengths']
                batch_size = len(lengths)
                N = targets.size(2)
                max_length = lengths.max()
                seq_length = self.seq_length
                loss_batch = torch.tensor(0.0, device=self.device)
                hidden_state = None
                # NOTE: We don't need to reset the initial hidden state because the default is to use zero for c0 and h0
                for start_block in range(0, max_length, seq_length):
                    local_lengths = torch.clamp(lengths - start_block, 1, seq_length)
                    end_block = start_block + local_lengths.max()
                    # last_elem = torch.sum(local_lengths > 0)
                    self.model.zero_grad()
                    scores, hidden_state, lengths_new = self.model(source[:, start_block:end_block], local_lengths, hidden_state)
                    # make sure that we skip the empty ones
                    local_lengths = torch.clamp(lengths - start_block, 0, seq_length)
                    total_loss = torch.tensor(0.0, device=self.device)
                    total_elems = torch.sum(local_lengths)
                    for i in range(batch_size):
                        if local_lengths[i] == 0:
                            continue
                        # print(i, local_lengths)
                        elem_loss = self.loss_function(scores[i, :local_lengths[i]],
                                                       targets[i, start_block:start_block+local_lengths[i]])
                        total_loss += elem_loss / N
                    total_loss = total_loss / total_elems
                    total_loss.backward()
                    if self.clip_gradients is not None:
                        _ = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradients)
                    self.optimizer.step()
                    loss_batch += total_loss
                    if hidden_state is not None:
                        hidden_state = self.detach_hidden_state(hidden_state)
                train_loss.append(loss_batch.item())
                if i_batch % self.print_steps == 0:
                    print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch + 1))
            print('Epoch', epoch, 'Train Loss:', torch.mean(torch.tensor(train_loss)).numpy())
            # Validation
            self.model.eval()
            total_metric, total_items, total_val_loss = self.eval_model(dataloader_val)
            results['val_metric'].append(total_metric)
            results['val_loss'].append(total_val_loss)
            results['train_loss'].append(train_loss)
            print('Epoch', epoch, 'Validation metric:', total_metric, ' (out of total_items:', total_items, ')')
            print('Epoch', epoch, 'Validation loss:', total_val_loss)

            if total_val_loss < 1e-4:
                print('Convergence achieved at epoch ', epoch)
                break

            self.epoch = epoch

            if epoch % self.save_every == 0 or \
                    (results['val_metric'][-1] > max_val_metric and epoch > self.max_epochs*0.1):
                # Save a checkpoint for reference
                state = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'results': results,
                    'clip_val': self.clip_gradients,
                    'learning_rate': self.lr,
                    'metric': self.metric,
                }
                self.save_checkpoint(state)
                print("Saved checkpoint for epoch {}".format(epoch))
                if results['val_metric'][-1] > max_val_metric:
                    max_val_metric = results['val_metric'][-1]

            if epoch >= self.max_epochs - test_last:
                res = self.test_model()
                results['test_metric'].append(res['test_metric'])

        return results

    def test_model(self):
        # Test the model
        results = {'test_metric': 0.0}
        self.model.eval()
        collator = SequenceCollator(batch_first=self.batch_first, device=self.device)
        dataset_test = self.load_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collator.seq_collate_fn)
        total_metric, total_items, _ = self.eval_model(dataloader_test)
        print('Test metric: ', total_metric, ' (out of total_items:', total_items, ')')
        results['test_metric'] = total_metric
        return results

    def test_summary(self, threshold=0.5):
        summary = {
            'lengths': [],
            'timescales': [],  # This stores the max_timescale
            'metric': [],
        }
        self.model.eval()
        collator = SequenceCollator(batch_first=self.batch_first, device=self.device)
        dataset_test = self.dataset("test", device=self.device, timescales=True, **self.dataset_parameters)
        dataloader = DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0,
                                collate_fn=collator.seq_collate_fn)
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                lengths = sample_batched['lengths']
                summary['lengths'].extend(list(lengths.cpu().numpy()))
                timescales = sample_batched['timescales']
                scores, _, lengths_new = self.model(source, lengths)
                batch_size = lengths.size(0)
                for i in range(batch_size):
                    summary['timescales'].append(timescales[i][:lengths[i]].cpu().numpy())
                    elem_metric = torch.all((scores[i][:lengths[i]] > threshold) == targets[i][:lengths[i]]).cpu()
                    summary['metric'].append(elem_metric.item())
        return summary
