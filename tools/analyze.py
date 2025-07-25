import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def build_figure_stat(df, fnout):
	df.set_index('type',inplace=True)
	for s in df.columns:
		f, ax = plt.subplots(figsize=(12, 6))
		fs=20
		font = {'size': fs}
		plt.rc('font', **font)
		#plt.ylim(0, 1.1)
		plt.scatter(df.index,df[s],s=fs*10)
		#current_values = plt.gca().get_yticks()
		#plt.gca().set_yticklabels(['{:,.4f}'.format(x) for x in current_values])

		#ax.yaxis.grid() # horizontal lines
		ax.xaxis.grid() # vertical lines
		ax.grid(True,which='major', axis='x', linestyle='--')
		plt.xticks(rotation=45, ha='right')
		plt.title(s)
		#plt.xlabel("Direction")
		#plt.ylabel(s)
		filename=fnout+"_"+s+'.pdf'
		plt.show()
		plt.savefig(filename)
		plt.close()
		print("See file ", filename)

def build_figure_target_predict(predict, target, fnout, title, mae, rmse, r2):
	f, ax = plt.subplots(figsize=(6, 6))
	#font = {'size': 10}
	#plt.rc('font', **font)
	plt.scatter(target,predict)
	plt.plot(target,target)
	s='MAE   = {:0.5}\nRMSE = {:0.5}\nR2     = {:0.5}'.format(mae,rmse,r2)
	plt.text(0.1, 0.9, s,  horizontalalignment='left', 
		verticalalignment='center', 
		bbox=dict(facecolor='lightsteelblue', alpha=0.5), 
		transform=ax.transAxes)
	plt.title(title)
	plt.xlabel("Target")
	plt.ylabel("Predict")

	filename=fnout+'.pdf'
	plt.show()
	plt.savefig(filename,format='pdf')
	plt.close()

 
def getR2(y_pred, y_true):
	ss_res = np.sum((y_true - y_pred) ** 2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	r2 = 1 - (ss_res / ss_tot)
	return float(r2)

def build_all_values(values, name_predict, name_target,  df_stat, fnout, build_figure=False):
	dname='all'
	predict=values[name_predict][:].reshape(-1)
	target=values[name_target][:].reshape(-1)

	mae=np.mean(np.abs(predict - target))
	r2=getR2(predict,target)
	rmse = np.sqrt(np.mean((predict - target) ** 2))
	std_target=float(np.std(target))
	meanabs_target=float(np.mean(np.abs(target)))
	mean_target=float(np.mean(target))
	min_target=float(np.min(target))
	max_target=float(np.max(target))

	df = pd.DataFrame({'mae':mae, 'R2':r2, 'rmse':rmse, 'std_target':std_target, 'meanabs_target':meanabs_target, 'mean_target':mean_target,'min_target':min_target, 'max_target':max_target}, index=[dname])
	df_stat = pd.concat([df_stat, df])
	if build_figure :
		fout=fnout+'_'+name_predict.split('_')[0]+'_'+dname
		title=name_predict.split('_')[0]+'_'+dname
		build_figure_target_predict(predict, target, fout, title, mae, rmse, r2)
	return df_stat
def build_one_case(values, name_predict, name_target,  df_stat, df_values, dname, fnout, i=None, j=None, k=None, build_figure=False):
	if i is not None and j is not None and k is not None:
		predict=values[name_predict][:,i,j,k].reshape(-1)
		target=values[name_target][:,i,j,k].reshape(-1)
	elif i is not None and j is not None :
		predict=values[name_predict][:,i,j].reshape(-1)
		target=values[name_target][:,i,j].reshape(-1)
	elif i is not None:
		predict=values[name_predict][:,i].reshape(-1)
		target=values[name_target][:,i].reshape(-1)
	else:
		predict=values[name_predict][:].reshape(-1)
		target=values[name_target][:].reshape(-1)

	mae=np.mean(np.abs(predict - target))
	r2=getR2(predict,target)
	rmse = np.sqrt(np.mean((predict - target) ** 2))
	std_target=float(np.std(target))
	meanabs_target=float(np.mean(np.abs(target)))
	mean_target=float(np.mean(target))
	min_target=float(np.min(target))
	max_target=float(np.max(target))

	df = pd.DataFrame({'mae':mae, 'R2':r2, 'rmse':rmse, 'std_target':std_target, 'meanabs_target':meanabs_target, 'mean_target':mean_target,'min_target':min_target, 'max_target':max_target}, index=[dname])
	df_stat = pd.concat([df_stat, df])
	df_values[name_predict+'_'+dname] = predict
	df_values[name_target+'_'+dname] = target
	if build_figure :
		fout=fnout+'_'+name_predict.split('_')[0]+'_'+dname
		title=name_predict.split('_')[0]+'_'+dname
		build_figure_target_predict(predict, target, fout, title, mae, rmse, r2)
	return df_stat, df_values

def getProps(data_file, name, fnout, build_figure=None):
	values=data_file['values'].item()
	name_predict=name+"_predict"
	name_target=name+"_target"
	if len(values['ID'].shape)==1:
		ID=values['ID'][:].reshape(-1)
	elif len(values['ID'].shape)==2:
		ID = np.array([f"{a}-{b}" for a, b in values['ID']])
	l=len(values[name_target].shape)-1
	mae_all={}
	r2_all={}
	rmse_all={}
	df_values = pd.DataFrame()
	df_stat = pd.DataFrame()
	if l==0:
		df_values['ID']=ID # scalar: energy, alfa, ...
		dl=['s']
		for i,dname in enumerate(dl):
			df_stat, df_values = build_one_case(values, name_predict, name_target,  df_stat, df_values, dname, fnout, build_figure=build_figure)
	elif l==1:
		dl=['x','y','z']
		n=values[name_predict].shape[0]//ID.shape[0]
		if n>1:
			#forces
			natoms= np.sum(values['N'])
			ID = []
			for i,N in enumerate(values['N']):
				ID.extend([int(values['ID'][i])]*int(values['N'][i]))
			df_values['ID']=np.asarray(ID) # forces
		else:
			df_values['ID']=ID # edipole or forces
		for i,dname in enumerate(dl):
			df_stat, df_values = build_one_case(values, name_predict, name_target,  df_stat, df_values, dname, fnout, i=i, build_figure=build_figure)
	elif l==2:
		d=['x','y','z']
		df_values['ID']=ID # alpha tensor
		for i,di in enumerate(d):
			for j,dj in enumerate(d):
				dname=di+dj
				df_stat, df_values = build_one_case(values, name_predict, name_target,  df_stat, df_values, dname, fnout, i=i,j=j, build_figure=build_figure)
	elif l==3:
		d=['x','y','z']
		df_values['ID']=ID # beta tensor
		for i,di in enumerate(d):
			for j,dj in enumerate(d):
				for k,dk in enumerate(d):
					dname=di+dj+dk
					df_stat, df_values = build_one_case(values, name_predict, name_target,  df_stat, df_values, dname, fnout, i=i,j=j,k=k, build_figure=build_figure)

	df_stat.loc['mean']=df_stat.mean(axis=0, numeric_only=True)
	dname="all"
	df_stat = build_all_values(values, name_predict, name_target,  df_stat, fnout, build_figure=build_figure)
	print(df_stat.to_string())
	print(df_values)
	return df_stat, df_values

 
if len(sys.argv)<2:
        print("Usage :")
        print("      give the name of npz file, the prefix name of the output file and an integer : 1=build pdf figures, 0 if not")
        print("      Example ", sys.argv[0], " train_dir/metrics/best_valid_valid_set.npz ", " result ")
        exit(1)

data_file   = np.load(sys.argv[1],allow_pickle=True)
fnout=sys.argv[2]
dkeys=data_file['values'].item().keys()
print(dkeys)
build_figure=int(sys.argv[3])==1
nc=180
nbc=60
for key in dkeys:
	if '_predict' in key:
		name=key.split('_')[0]
		print('-'*nbc,name, '-'*(nc-nbc-len(name)))
		df_stat , df_values= getProps(data_file, name, fnout, build_figure=build_figure)
		df_stat['type']=df_stat.index
		fout_stat=fnout+'_stat_'+name+'.csv'
		df_stat.to_csv(fout_stat, index=False)  
		print("See ", fout_stat,' file')
		fout_values=fnout+'_values_'+name+'.csv'
		df_values.to_csv(fout_values, index=False)  
		print("See ", fout_values,' file')
		build_figure_stat(df_stat, fnout+'_'+name)
print('-'*180)
print("See ", fnout+'*.pdf', ' file')
print('-'*180)


