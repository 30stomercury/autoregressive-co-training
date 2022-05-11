import yaml
import os
import random
import torch
import logging
from dataloader import ls_data, collate_fn
from parser import get_runner_args
from cotraining import Cotraining

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2022
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)
    return logging.getLogger(logger_name)


def update_config(path, config_filename, config):
    with open(os.path.join(path, config_filename), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

def Dataloader(config, split):
    # ls data
    split_set = ls_data(config['data'], part=split)
    split_config= 'dev_dataloader'
    # Write train statistics
    if split == 'train':
        split_config = 'tr_dataloader'

    # Train dataloader
    data_loader = torch.utils.data.DataLoader(
        split_set, 
        collate_fn=collate_fn(),
        **config[split_config])

    print(f'total {split} data: {len(split_set)}')

    return data_loader

if __name__ == '__main__':

    config, args, path = get_runner_args()
    logger_main = setup_logger('main', os.path.join(path, 'log'))
    logger_main.info('Log file to {}'.format(os.path.join(path, 'log')))

    # Init dataloader
    tr_loader = Dataloader(config, 'train')
    dev_loader = None

    # Load Model from specific epoch
    prev_epoch = 1
    ckpt_path = None
    optim_path = None
    if args.ckpt:
        ckpt_path = os.path.join(path, 'ckpt', 'model_{}.ckpt'.format(args.ckpt))
        prev_epoch = int(args.ckpt) + 1

    # Init model
    if 'steps' not in config:
        config['steps'] = 0
    model_solver = Cotraining(tr_loader, dev_loader, config, device)
    model_solver.load(ckpt_path, device=device)
    print('Models:\n', model_solver.model)
    print('Training params:\n', config['training'])

    # Save epoch 0
    if not args.ckpt:
        model_solver.save(0, path)

    if not args.dev:
        # Start training if not args.dev
        logger_main.info('Start training model from epoch {} of {}'.format(
            prev_epoch, config['training']['epoch']))
        total_params = sum(p.numel() for p in model_solver.model.parameters() if p.requires_grad)
        config['model']['n_params'] = total_params
        logger_main.info('Model params: {}'.format(total_params))
        for e in range(prev_epoch, config['training']['epoch'] + 1):
            # Train 
            model_solver.model.train()
            summary = model_solver.run_epoch(phase='train')
            msg = f'Epoch {e} - train '
            for k, v in summary.items():
                msg += f'{k}: {v:.3f}, '
            logger_main.info(msg[:-2])

            # Save train
            if e % config['training']['save_every'] == 0:
                model_solver.save(e, path)
                config['steps'] = model_solver.steps
                update_config(path, args.config.split('/')[-1], config)
