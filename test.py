from schnetpack import AtomsData

from schnetpack.datasets import QM9

qm9data = QM9('./qm9.db', download=True)

print('Number of reference calculations:', len(qm9data))
print('Available properties:')

for p in qm9data.available_properties:
    print('-', p)

example = qm9data[0]
print('Properties:')

for k, v in example.items():
    print('-', k, ':', v.shape)
    print('-', k, ':', v.tolist())


at = qm9data.get_atoms(idx=0)
print('Atoms object:', at)

at2, props = qm9data.get_properties(idx=0)
print('Atoms object (not the same):', at2)
print('props type:', type(props))
print('Equivalent:', at2 == at, '; not the same object:', at2 is at)

print('Total energy at 0K:', props[QM9.U0])
print('HOMO:', props[QM9.homo])
print('Free energy:', props[QM9.G])


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

for k, v in props.items():
    props[k] = v.numpy()

props[QM9.G][0] += 1

new_dataset = AtomsData('./new_dataset.db', available_properties=available_properties)

# from ase.visualize import view
# view(at2)

new_dataset.add_systems([at2], [props])
print('-' * 100)
qm9data_new = QM9('./new_dataset.db', download=True)
example = qm9data_new[0]
print('Properties:')

for k, v in example.items():
    print('-', k, ':', v.tolist())

