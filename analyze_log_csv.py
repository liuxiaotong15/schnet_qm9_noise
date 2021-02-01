import os
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol

import argparse

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

parser = argparse.ArgumentParser(description='analyze the csv file in folder')
parser.add_argument('--folder', '-f', required = True,
        help='input folder please')
args = parser.parse_args()
print(args)

results = np.loadtxt(os.path.join(args.folder, 'log.csv'), skiprows=1, delimiter=',')

time = results[:,0]-results[0,0]
learning_rate = results[:,1]
train_loss = results[:,2]
val_loss = results[:,3]
val_mae = results[:,4]

print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
              np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')

# plt.figure(figsize=(14,5))
# plt.subplot(1,2,1)
plt.plot(time/3600, val_loss, label='Validation')
plt.plot(time/3600, train_loss, label='Train')
plt.yscale('log')
plt.ylabel('Loss/eV', font_axis)
plt.xlabel('Time/(h)', font_axis)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
# plt.subplot(1,2,2)
# plt.plot(time, val_mae)
# plt.ylabel('mean abs. error [eV]')
# plt.xlabel('Time [s]')

plt.subplots_adjust(bottom=0.12, right=0.986, left=0.124, top=0.986)
plt.show()
