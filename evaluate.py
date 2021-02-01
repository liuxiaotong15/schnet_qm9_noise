import os
import schnetpack as spk
from schnetpack import AtomsData
from schnetpack.datasets import QM9
import random
import numpy as np
import argparse

from ase.units import kcal, mol

parser = argparse.ArgumentParser(description='test from xxx.db')
parser.add_argument('--test_db', '-t', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--folder', '-f', required = True,
        help='give a folder which contain best_model file')

args = parser.parse_args()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)

qm9data = QM9(args.test_db, download=False, load_only=[QM9.G])

test_loader = spk.AtomsLoader(qm9data, batch_size=100, shuffle=True)

import torch
best_model = torch.load(os.path.join(args.folder, 'best_model'))

device = 'cuda'
err = 0
print(len(test_loader))
for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}
    # apply model
    pred = best_model(batch)
    # calculate absolute error
    tmp = torch.sum(torch.abs(pred[QM9.G]-batch[QM9.G]))
    tmp = tmp.detach().cpu().numpy() # detach from graph & convert to numpy
    err += tmp
    # log progress
    percent = '{:3.2f}'.format(count/len(test_loader)*100)
    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

err /= len(qm9data)
print('Test MAE', np.round(err, 2), 'eV =', np.round(err / (kcal/mol), 2), 'kcal/mol')

