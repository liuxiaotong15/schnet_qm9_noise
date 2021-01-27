from schnetpack import AtomsData
from schnetpack.datasets import QM9

seed = 1234

import random
import numpy as np

random.seed(seed)
np.random.seed(seed)

q1_cnt = 120000
q2_cnt = 10000

qm9data = QM9('./qm9.db', download=True)

idx_lst = list(range(len(qm9data)))
random.shuffle(idx_lst)

# for p in qm9data.available_properties:
#     print('-', p)
# 
# example = qm9data[0]
# 
# for k, v in example.items():
#     print('-', k, ':', v.shape)
#     print('-', k, ':', v.tolist())

at = qm9data.get_atoms(idx=0)


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


q1 = AtomsData('./Q1.db', available_properties=available_properties)
at_lst = []
props_lst = []
for idx in idx_lst[:q1_cnt]:
    at2, props = qm9data.get_properties(idx=idx)
    for k, v in props.items():
        props[k] = v.numpy()
    at_lst.append(at2)
    props_lst.append(props)
q1.add_systems(at_lst, props_lst)

q2 = AtomsData('./Q2.db', available_properties=available_properties)
at_lst = []
props_lst = []
for idx in idx_lst[q1_cnt:q1_cnt + q2_cnt]:
    at2, props = qm9data.get_properties(idx=idx)
    for k, v in props.items():
        props[k] = v.numpy()
    at_lst.append(at2)
    props_lst.append(props)
q2.add_systems([at2], [props])

q3 = AtomsData('./Q3.db', available_properties=available_properties)
at_lst = []
props_lst = []
for idx in idx_lst[q1_cnt + q2_cnt:]:
    at2, props = qm9data.get_properties(idx=idx)
    for k, v in props.items():
        props[k] = v.numpy()
    at_lst.append(at2)
    props_lst.append(props)
q3.add_systems([at2], [props])


# new_dataset.add_systems([at2], [props])
# print('-' * 100)
# qm9data_new = QM9('./new_dataset.db', download=True)
# example = qm9data_new[0]
# print('Properties:')
# 
# for k, v in example.items():
#     print('-', k, ':', v.tolist())
# 
