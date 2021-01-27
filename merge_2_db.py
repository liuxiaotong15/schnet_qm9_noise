from schnetpack import AtomsData
from schnetpack.datasets import QM9
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='add x% random noise on xxx.db')
parser.add_argument('--first_db', '-f', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--second_db', '-s', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--output_filename', '-o', required = True,
        help='xxx.db, output filename please')
args = parser.parse_args()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)

qm9data1 = QM9(args.first_db, download=False)
qm9data2 = QM9(args.second_db, download=False)
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

new_dataset = AtomsData(args.output_filename, available_properties=available_properties)

at_lst = []
props_lst = []

for i in range(len(qm9data1)):
    at2, props = qm9data.get_properties(idx=i)
    for k, v in props.items():
        props[k] = v.numpy()
    at_lst.append(at2)
    props_lst.append(props)

for i in range(len(qm9data2)):
    at2, props = qm9data.get_properties(idx=i)
    for k, v in props.items():
        props[k] = v.numpy()
    at_lst.append(at2)
    props_lst.append(props)

new_dataset.add_systems(at_lst, props_lst)

