import functools
import yaml
import argparse
import os

class join(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = '!join'
    @classmethod
    def from_yaml(cls, loader, node):
        return functools.reduce(lambda a, b: a.value + b.value, node.value)

def get_runner_args():
    parser = argparse.ArgumentParser(description='Argument Parser for the anything ...')

    # general configuration
    parser.add_argument('--config', '-c', default="config/cotraining.yaml")
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--dev', default=False)

    args = parser.parse_args()
    print('Loading config from {}'.format(args.config))
    yaml.add_constructor('!join', join)
    config = yaml.safe_load(open(args.config))

    # change config file
    if 'path' in config:
        path = config['path']
    else:
        prefix = config['prefix']
        num = 0
        while os.path.isdir(prefix + '/' + str(num)):
            num += 1
        path = os.path.join(prefix, str(num))
        config['path'] = path

    os.makedirs(os.path.join(path, 'ckpt'), exist_ok=True)

    return config, args, path
