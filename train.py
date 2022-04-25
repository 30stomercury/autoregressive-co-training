import yaml
import os
import random

import torch
from utils import setup_loggers
from dataloader import ls_data, collate_fn
from parser import get_runner_args

from cotraining import Cotraining

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 2022
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)

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
    logger_main = setup_loggers('main', os.path.join(path, 'log'), args)
    logger_main.info('Log file to {}'.format(os.path.join(path, 'log')))

    # Init dataloader
    tr_loader = Dataloader(config, 'train')
    dev_loader = None

    # Load Model from specific epoch
    prev_epoch = 1
    ckpt_path = None
    optim_path = None
    if args.ckpt:
        ckpt_path = os.path.join(path, 'ckpt', 'cotraining_model_{}.ckpt'.format(args.ckpt))
        prev_epoch = int(args.ckpt) + 1

    # Init model
    if 'steps' not in config:
        config['steps'] = 0
    cotraining_solver = Cotraining(tr_loader, dev_loader, config, device)
    cotraining_solver.load(ckpt_path, device=device)
    print('Models:\n', cotraining_solver.model)
    print('Training params:\n', config['training'])
    # Save epoch 0
    if not args.ckpt:
        cotraining_solver.save(0, path)

    if not args.dev:
        # Start training if not args.dev
        logger_main.info('Start training model from epoch {} of {}'.format(
            prev_epoch, config['training']['epoch']))
        total_params = sum(p.numel() for p in cotraining_solver.model.parameters() if p.requires_grad)
        config['model']['n_params'] = total_params
        logger_main.info('Model params: {}'.format(total_params))
        if 'best_dev_err' not in config:
            config['best_dev_err'] = 1000
            config['best_epoch'] = 0
        update_config(path, args.config.split('/')[-1], config)
        for e in range(prev_epoch, config['training']['epoch']+1):
            # Train 
            cotraining_solver.model.train()
            loss, error, ent, ce_loss, rec_loss = cotraining_solver.run_epoch(phase='train')

            # Save train
            logger_main.info(
                'Epoch {} - train err: {:.3f}, train loss: {:.3f}, ent: {:.3f}, ce_loss: {:.3f}, kl loss: {:.3f}, rec loss: {:.3f}'.format(
                    e, error, loss, ent, ce_loss, ce_loss-ent, rec_loss
                )
            )
            if e % config['training']['save_every'] == 0:
                cotraining_solver.save(e, path)
            # Eval
            loss, error = cotraining_solver.run_epoch(phase='eval')
            logger_main.info('Epoch {} - ave dev err: {:.3f}, ave dev loss: {:.3f}'.format(e, error, loss))
            if error < config['best_dev_err']:
                config['best_dev_err'] = error
                config['best_epoch'] = e
                update_config(path, args.config.split('/')[-1], config)
        logger_main.info('Best epoch {} - ave dev err: {:.3f}'.format(
            config['best_epoch'], config['best_dev_err'])) 
    # Test
    ckpt_path = os.path.join(path, 'ckpt', 'cotraining_model_{}.ckpt'.format(config['best_epoch']))
    cotraining_solver.load(ckpt_path, device=device)
    for split in []:
        eval_loader = Dataloader(config, split)
        cotraining_solver.eval_loader = eval_loader
        loss, error = cotraining_solver.run_epoch(phase='eval')
        logger_main.info('Epoch {} - ave {} err: {:.3f}, ave {} loss: {:.3f}'.format(config['best_epoch'], split, error, split, loss))
        config[f'best_{split}_err'] = error
    update_config(path, args.config.split('/')[-1], config)
