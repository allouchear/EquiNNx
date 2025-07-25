import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
 
def getR2(y_pred, y_true):
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	return r2
 
if len(sys.argv)<2:
        print("Usage :")
        print("      give the name of npz file , idx dipole (x, y, z) and the prefix name of the output file")
        print("      Example ", sys.argv[0], " train_dir/metrics/best_valid_valid_set.npz ", " x ", " dipole ")
        exit(1)

d={'x':0, 'y':1, 'z':2}
sidx=sys.argv[2]
sidx=sidx.lower()
i=d[sidx[0]]
#j=d[sidx[1]]

data_file   = np.load(sys.argv[1],allow_pickle=True)
values=data_file['values'].item()
predict=values['edipole_predict'][:,i].reshape(-1)
target=values['edipole_target'][:,i].reshape(-1)
mae=np.mean(np.abs(predict - target))
r2=getR2(predict,target)
rmse = np.sqrt(np.mean((predict - target) ** 2))
df = pd.DataFrame({'predict':predict, 'target':target})
print(df)
fnout=sys.argv[3]
f, ax = plt.subplots()
plt.scatter(target,predict)
plt.plot(target,target)
s='MAE   = {:0.5}\nRMSE = {:0.5}\nR2     = {:0.5}'.format(mae,rmse,r2)
print(s)
plt.text(0.1, 0.9, s,  horizontalalignment='left', 
verticalalignment='center', 
bbox=dict(facecolor='lightsteelblue', alpha=0.5), 
transform=ax.transAxes)
plt.title(fnout)
plt.xlabel("Target")
plt.ylabel("Predict")

filename=fnout+'.pdf'
plt.savefig(filename)
plt.show()
print("Voir fichier ", filename)

