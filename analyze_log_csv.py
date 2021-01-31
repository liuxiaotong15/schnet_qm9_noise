import os
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol

import argparse

parser = argparse.ArgumentParser(description='analyze the csv file in folder')
parser.add_argument('--input_filename', '-i', required = True,
        help='xxx.db, input filename please')
args = parser.parse_args()
print(args)

results = np.loadtxt(os.path.join(args.input_filename, 'log.csv'), skiprows=1, delimiter=',')

time = results[:,0]-results[0,0]
learning_rate = results[:,1]
train_loss = results[:,2]
val_loss = results[:,3]
val_mae = results[:,4]

print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
              np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(time, val_loss, label='Validation')
plt.plot(time, train_loss, label='Train')
plt.yscale('log')
plt.ylabel('Loss [eV]')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(1,2,2)
plt.plot(time, val_mae)
plt.ylabel('mean abs. error [eV]')
plt.xlabel('Time [s]')
plt.show()
