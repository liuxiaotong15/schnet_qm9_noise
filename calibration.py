import os
import schnetpack as spk
from schnetpack import AtomsData
from schnetpack.datasets import QM9
import random
import numpy as np
import argparse

from ase.units import kcal, mol

parser = argparse.ArgumentParser(description='calibration a db from a model')
parser.add_argument('--input_db', '-i', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--output_db', '-o', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--folder', '-f', required = True,
        help='give a folder which contain best_model file')

args = parser.parse_args()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)

available_properties = [
        QM9.A,
        QM9.B,
        QM9.C,
        QM9.mu,
        QM9.alpha,
        QM9.homo,
        QM9.lumo,
        QM9.gap,
        QM9.r2,
        QM9.zpve,
        QM9.U0,
        QM9.U,
        QM9.H,
        QM9.G,
        QM9.Cv,
        ]


qm9data = QM9(args.input_db, download=False, load_only=[QM9.G])

import torch
best_model = torch.load(os.path.join(args.folder, 'best_model'))

device = 'cpu'
err = 0

converter = spk.data.AtomsConverter(device=device)

at_lst = []
props_lst = []

for i in range(len(qm9data)):
    at, props = qm9data.get_properties(idx=i)
    for k, v in props.items():
        props[k] = v.numpy()

    input_atoms = converter(at)
    # apply model
    pred = best_model(input_atoms)
    props[QM9.G][0] = (props[QM9.G][0] + pred[QM9.G].detach().cpu().numpy()[0, 0]) / 2
    at_lst.append(at)
    props_lst.append(props)


new_dataset = AtomsData(args.output_db, available_properties=[QM9.G]) #available_properties)

new_dataset.add_systems(at_lst, props_lst)
