from schnetpack import AtomsData
from schnetpack.datasets import QM9
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='add x% random noise on xxx.db')
parser.add_argument('--noise', '-n', required = True, type = float,
        help='x% noise, just input x is enough, I will add % in the code')
parser.add_argument('--input_filename', '-i', required = True,
        help='xxx.db, input filename please')
parser.add_argument('--output_filename', '-o', required = True,
        help='xxx.db, output filename please')
args = parser.parse_args()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)

qm9data = QM9(args.input_filename, download=False)
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

for i in range(len(qm9data)):
    at2, props = qm9data.get_properties(idx=i)
    for k, v in props.items():
        props[k] = v.numpy()
    if(random.random() > 0.5):
        props[QM9.G][0] *= (1+args.noise*0.01)
    else:
        props[QM9.G][0] *= (1-args.noise*0.01)
    at_lst.append(at2)
    props_lst.append(props)

new_dataset.add_systems(at_lst, props_lst)

print('-' * 100)
example = qm9data[0]
print('Properties:')
for k, v in example.items():
    print('-', k, ':', v.tolist())
qm9data_new = QM9(args.output_filename, download=False)
example_new = qm9data_new[0]
print('Properties:')
for k, v in example_new.items():
    print('-', k, ':', v.tolist())
