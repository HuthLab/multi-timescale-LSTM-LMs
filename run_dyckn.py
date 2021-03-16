from datasets import DyckNDataset
from experiments import DyckNExperiment
from model_dyckn import BaselineLSTMModel, MTSLSTMModel, LSTMBiasInvGammaInitializer
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path


def main(input_path, output_path, model_name, units, n_layers, lr, batch_size, max_epochs, load_epoch, alpha, scale,
         seq_length, test_last, rand_seed):
    od = Path(output_path)
    od.mkdir(parents=True, exist_ok=True)

    experiment_name = 'DyckN'
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(device)
    
    # Set torch's random seed
    torch.manual_seed(rand_seed)

    dataset_parameters = {
        'path': input_path,
    }

    # Pre-read the dataset information:
    d = DyckNDataset("train", input_path)
    N = d.N
    V = len(d.vocabulary)

    if model_name == 'Baseline':
        model = BaselineLSTMModel(units, input_size=V, output_size=N, num_layers=n_layers)
    elif model_name == 'MTS':
        bias_initializer = LSTMBiasInvGammaInitializer(alpha, scale)
        model = MTSLSTMModel(units, input_size=V, output_size=N, fixed_bias=True, init_bias=bias_initializer,
                             num_layers=n_layers)
    else:
        raise ValueError(f'Unknown model name {model_name}')

    if seq_length == 0:
        seq_length = None

    optimizer = optim.Adam
    clip_gradients = 1.0
    # Count and print number of parameters
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters in model: ', total_params)

    experiment = DyckNExperiment(
        DyckNDataset,
        model,
        optimizer,
        seq_length=seq_length,
        lr=lr,
        dataset_parameters=dataset_parameters,
        batch_size=batch_size,
        max_epochs=max_epochs,
        clip_gradients=clip_gradients,
        load_checkpoint_epoch=load_epoch,
        name=experiment_name,
        device=device,
        print_steps=1000,
        checkpoint_directory=output_path,
        save_every=50,
    )

    results_train = experiment.train_model(test_last)

    # Save the trained model weights
    experiment.save_model(output_path)

    # Test the model
    results_test = experiment.test_model()
    np.savez_compressed(output_path + '/results.npz', results_train=results_train, results_test=results_test,
                        total_params=total_params, allow_pickle=True)

    # Test the model
    summary = experiment.test_summary()
    print('Save summary')
    np.savez_compressed(output_path + '/summary.npz', summary=summary, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', default='./results/', type=str, help='Output directory')
    parser.add_argument('-i', default='./data/dyckn/', type=str, help='Input data directory')
    parser.add_argument('-m', "--model", type=str, default="Baseline", choices=['Baseline', 'MTS'], help="Model (from choices)")
    parser.add_argument("-u", "--units", type=int, default=256, help="Number of cells in the RNN per layer")
    parser.add_argument("-l", "--layers", type=int, default=1, help="Number of layers")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('-le', "--load_epoch", type=int, help="Epoch number to load from", default=None)
    parser.add_argument("--alpha", type=float, default=1.5, help="Alpha for Inv Gamma initialization")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for Inv Gamma initialization")
    parser.add_argument("--test_last", type=int, default=1, help="Store test results for the last # of epochs")
    parser.add_argument('-s', "--seq_length", type=int, help="Sequence length for stateful training", default=200)
    parser.add_argument('-r', "--seed", type=int, help="Random seed", default=0)

    args = parser.parse_args()
    print(args)
    lr = args.lr
    batch_size = args.batch_size
    max_epochs = args.epochs
    units = args.units
    n_layers = args.layers
    model = args.model
    load_epoch = args.load_epoch
    alpha = args.alpha
    scale = args.scale
    seq_length = args.seq_length
    test_last = args.test_last
    rand_seed = args.seed

    main(args.i, args.o, model, units, n_layers, lr, batch_size, max_epochs, load_epoch, alpha, scale, seq_length,
         test_last, rand_seed)
