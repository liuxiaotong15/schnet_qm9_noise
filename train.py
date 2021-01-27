import os
import schnetpack as spk
from schnetpack import AtomsData
from schnetpack.datasets import QM9
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='train from xxx.db')
parser.add_argument('--train_db', '-t', required = True,
        help='xxx.db, input filename please')
args = parser.parse_args()
print(args)

seed = 1234
random.seed(seed)
np.random.seed(seed)

foldername = args.train_db.replace('.', '')

if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    print('folder existed, delete it please...')
    exit(0)

qm9data = QM9(args.train_db, download=False, load_only=[QM9.G])

train, val, test = spk.train_test_split(
        data=qm9data,
        num_train=int(0.8*len(qm9data)),
        num_val=len(qm9data) - int(0.8*len(qm9data)),
        split_file=os.path.join(foldername, "split.npz"),
        )

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

atomrefs = qm9data.get_atomref(QM9.G)
means, stddevs = train_loader.get_statistics(
        # QM9.G, get_atomwise_statistics=True, single_atom_ref=atomrefs
        QM9.G, single_atom_ref=atomrefs
        )

schnet = spk.representation.SchNet(
        n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
        cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
        )

output_G = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs[QM9.G], property=QM9.G,
                                           mean=means[QM9.G], stddev=stddevs[QM9.G])
model = spk.AtomisticModel(representation=schnet, output_modules=output_G)


from torch.optim import Adam

# loss function
def mse_loss(batch, result):
    diff = batch[QM9.U0]-result[QM9.U0]
    err_sq = torch.mean(diff ** 2)
    return err_sq

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

import schnetpack.train as trn

loss = trn.build_mse_loss([QM9.G])

metrics = [spk.metrics.MeanAbsoluteError(QM9.G)]
hooks = [
        trn.CSVHook(log_path=foldername, metrics=metrics),
        trn.ReduceLROnPlateauHook(
            optimizer,
            patience=5, factor=0.8, min_lr=1e-6,
            stop_after_min=True
            )
        ]

trainer = trn.Trainer(
        model_path=foldername,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        )

device = "cpu" # change to 'cpu' if gpu is not available
n_epochs = 300 # takes about 10 min on a notebook GPU. reduces for playing around
trainer.train(device=device, n_epochs=n_epochs)
