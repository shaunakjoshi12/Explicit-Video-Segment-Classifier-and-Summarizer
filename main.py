import torch
import argparse
import torch.nn as nn
from dataset import VideoClipDataset
from torch.utils.data import DataLoader

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')

