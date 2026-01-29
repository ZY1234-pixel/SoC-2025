import argparse

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mthv2.yaml',type=str)
    return parser