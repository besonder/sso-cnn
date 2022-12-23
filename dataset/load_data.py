from .core import *
from .torch_backend import *
from utils.option import Config

DATA_DIR = './data'

def get_dataset(args: Config):
    args.logger(f'Using batch size: {args.batch_size}')
    dataset = cifar10(DATA_DIR)
    t = Timer()
    args.logger('Preprocessing training data')

    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)) / 255.0, dataset['train']['labels']))
    args.logger(f'Finished in {t():.2} seconds')
    args.logger('Preprocessing test data')
    test_set = list(zip(transpose(dataset['test']['data']) / 255.0, dataset['test']['labels']))
    args.logger(f'Finished in {t():.2} seconds')

    train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])

    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=args.num_workers)
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    for batch in train_batches:
        break
    args.logger(batch.keys())

    return train_batches, test_batches
