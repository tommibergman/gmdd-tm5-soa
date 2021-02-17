from scipy.special import erf
import matplotlib.pyplot as plt 
import nappy
import datetime
import glob
import netCDF4 as nc
import re
import pandas as pd
from mpl_toolkits.basemap import Basemap
from general_toolbox import get_gridboxarea,lonlat,write_netcdf_file,parse_nas_normalcomments,NMB
import os
from scipy.stats import pearsonr
import numpy as np
import emep_read
import improve_plots
import logging
import matplotlib as mpl
from settings import *
def get_cutoffs():
	cutoff={}
	cutoff['CA0100R']=['TSI_3022A_WHI',7e-9]
	cutoff['CA0102R']=['TSI_3775_ETL',4e-9]
	cutoff['CA0420G']=['TSI_3010_ALT',10e-9]
	cutoff['CH0001G']=['TSI_3772_JFJ_dry',10e-9]
	cutoff['DE0043G']=['TSI_CPC_3025',3e-9]
	cutoff['DE0043G']=['TSI_CPC_3772',10e-9]
	cutoff['DE0043G']=['TSI_CPC_3772',10e-9]
	cutoff['DE0060G']=['CPC_3022A_NMY',7e-9]
	cutoff['ES0018G']=['CPC_3025A_SN1160_amb',3e-9]
	cutoff['FI0023R']=['CPC_VAR_01',np.nan]
	cutoff['FI0050R']=['CPC_HYY_01',np.nan]
	cutoff['FI0096G']=['cpc',np.nan]
	cutoff['FR0030R']=['TSI_3010_PUY',10e-9]
	cutoff['GB0036R']=['cpc_GB04',np.nan]
	cutoff['IE0031R']=['CPC_01',np.nan]
	cutoff['IT0009R']=['TSI_3772_CMN',10e-9]
	cutoff['KR0101R']=['TSI_3776_GSN',2.5e-9]
	cutoff['LT0015R']=['UF02_Preila',np.nan]
	cutoff['NO0042G']=['ZEP-CPC1',np.nan]
	cutoff['PR0100C']=['TSI_3022_CPR',7e-9]
	cutoff['TW0100R']=['TSI_3010_LLN',10e-9]
	cutoff['US0008R']=['TSI_3010_BRW',10e-9]
	cutoff['US0035R']=['TSI_3760_BND',11e-9]
	cutoff['US1200R']=['TSI_3760_MLO',11e-9]
	cutoff['US3446C']=['TSI_3760_APP',11e-9]
	cutoff['US6001R']=['TSI_3010_SMO',10e-9]
	cutoff['US6002C']=['TSI_3010_SGP',10e-9]
	cutoff['US6004G']=['TSI_3760_SPO',11e-9]
	cutoff['US6005G']=['TSI_3760_THD',11e-9]
	cutoff['US9050R']=['TSI_3010_SPL',10e-9]
	cutoff['ZA0001G']=['TSI_3781_CPT',6e-9]
	return cutoff

def list_stations(indict):
	paperpath=''
	tabletex=open(paperpath+'sitelist_CN.tex','w')
	tabletex.write("%Name & Station code&Longitude& Latitude &Height\\\\\n")
	sortdict={}
	for i in sorted(indict.keys()):
		#print i,indict[i]
		#print len(indict[i][0])
		print ('%6s,%35s,%4s,%4s,%4s')%(i,indict[i]['name'],indict[i]['lon'],indict[i]['lat'],indict[i]['ele'])
		stationname=indict[i]['name']
		#print stationname
		stationname=stationname.replace("#","\\#")
		#print stationname
		sortdict[stationname]=[i,indict[i]['lon'],indict[i]['lat'],indict[i]['ele']]
	for j in sorted(sortdict):
		#print ('%-35s,%6s,%4s,%4s,%4s')%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3])
		tabletex.write("%s&%6s&%4s&%4s&%4s\\\\\n"%(j,sortdict[j][0],sortdict[j][1],sortdict[j][2],sortdict[j][3]))
	tabletex.close()

def monthly_aggregation(sitedata,dataname='CN'):
	meanmodelmon={}
	meanmodelmon={k:np.zeros(12) for k in sitedata.keys()}
	kk=0
	for i in sitedata:
		timedata=sitedata[i]['time']
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		monthdata=np.full(12,np.NAN)
		modelmonthdata=np.full(12,np.NAN)
		for j in timedata:
			monthindices[j.month].append(timedata.index(j))
		for k in monthindices:
			monthdata[k-1]=np.nanmean([sitedata[i][dataname][ijk] for ijk in monthindices[k]])
		meanmodelmon[i]=monthdata

	return meanmodelmon

def read_num(filein):
	fhandle=nc.Dataset(filein,'r')
	output={}
	for imode in ['N_NUS','N_AIS','N_ACS','N_COS','N_AII','N_ACI','N_COI']:
		output[imode]=fhandle[imode][:]
	return output
def read_rwet(filein):
	fhandle=nc.Dataset(filein,'r')
	output={}
	for imode in ['RWET_NUS','RWET_AIS','RWET_ACS','RWET_COS','RWET_AII','RWET_ACI','RWET_COI']:
		output[imode]=fhandle[imode][:]
	return output
def read_timeaxis(filein):
	fhandle=nc.Dataset(filein,'r')
	return fhandle.variables['time'][:]
def colocate_time(gpdata,timeaxis,stime,etime,obsdata,errorvalue=999999):
	timeaxis=timeaxis-np.floor(timeaxis[0])
	modelstep=0
	model1_mean=[]
	model1_std=[]
	time=[]
	for obsstep in range(len(stime)):
		N=0
		temp=[]
		#check for not going out of time eaxis

		if modelstep>(np.size(timeaxis)-1):
			continue
		# increase a until model data corresponds to obsdata
		while(timeaxis[modelstep]<stime[obsstep]):
			modelstep+=1
			if  modelstep>(np.size(timeaxis)-1):
				break
		#check that we are still within model data
		if modelstep>(np.size(timeaxis)-1):
			continue
		# amodel_mean data until model time exceeds current 
		# obs period (etime)
		while(timeaxis[modelstep]>=stime[obsstep] and timeaxis[modelstep]<etime[obsstep]):
			if obsdata[obsstep]<errorvalue:
				N+=1
				temp.append(gpdata[modelstep])
				modelstep+=1
				if modelstep==np.size(timeaxis):
					break
			elif np.isnan(obsdata[obsstep]):
				N+=1
				temp.append(np.nan)
				modelstep+=1
				if modelstep==np.size(timeaxis):
					break
			else:
				modelstep+=1
			# again check taht we are within model data
			if  modelstep>=(np.size(timeaxis)-1):
				break
		if N>0:
			model_mean=np.mean(temp)*1e-6
			model_std=np.std(temp)*1e-6
			time1=(timeaxis[modelstep-N]+timeaxis[modelstep-1])/2

		else:
			print 'ERROR:  no data points found in modeldata, '
			print obsstep,modelstep
			continue	
		model1_mean.append(model_mean)
		model1_std.append(model_std)
		time.append(nc.num2date(time1,'days since 2010-01-01 00:00:00',calendar='standard'))
	return model1_mean,model1_std,time
def cnRead(filein='/Users/bergmant/Documents/obs-data/Ebas_200120_1014_CN/*.nas'):
	datadict={}
	for file in glob.glob(filein):
		sitedict={}
		#print file
		reader=nappy.openNAFile(file)
		#print reader.getNormalComments()
		# Get the Normal Comments (SCOM) lines.
		norm_comms = reader.getNormalComments()
		# parse header stuff
		lon,lat,stationname,site,elev,sdate,matrix,unit=parse_nas_normalcomments(norm_comms)
		# save for site
		sitedict['name']=stationname
		sitedict['lat']=lat
		sitedict['lon']=lon
		sitedict['ele']=elev
		sitedict['unit']=unit
		sitedict['matrix']=matrix
		#print sdate
		XX=reader.getIndependentVariables()
		variables=reader.getVariables()
		nadict=reader.getNADict()
		reader.readData()
		floats = [np.float(e) for e in reader["V"][1]]
		pltdata=np.array(floats)
		# if np.any(pltdata>10000):
		# 	for i in pltdata:
		# 		print i
		# 		print stationname,pltdata[i],np.count_nonzero(pltdata>10000)
		pltdata[pltdata>10000]=np.nan
		sitedict['CN']=pltdata
		#print reader["V"][0]
		#print reader["X"]
		endtime=[float(b) for b in reader["V"][0]]
		starttime=[float(e) for e in reader["X"]]
		stamp=[nc.num2date((float(b)+float(e))/2,units='days since 2010-01-01 00:00:00',calendar='standard') for b,e in zip(reader["V"][0],reader["X"])]
		#print stamp
		sitedict['time']=stamp
		sitedict['starttime']=starttime
		sitedict['endtime']=endtime
		datadict[site]=sitedict
	return datadict
def collocate():
	for isite in observed_data:
		gpdata,lat,lon=emep_read.select_gp(test2,observed_data[isite]['lat'],observed_data[isite]['lon'])
		print gpdata
		sadf
		m1m,m1s,t1=colocate_time(gpdata,timeax,observed_data[isite]['starttime'],observed_data[isite]['endtime'],observed_data[isite]['CN'])
		#gpdata,mlat,mlon=select_gp(modeldata,lat,lon)
		plt.figure()
		plt.title(isite)
		plt.plot(observed_data[isite]['time'],observed_data[isite]['CN'],lw=3)
		plt.plot(observed_data[isite]['time'],m1m,'r',lw=1)
		plt.xlim(0,365)
		ax_sc.plot(np.nanmean(observed_data[isite]['CN']),np.nanmean(m1m),'or')
def read_N(filein="/Users/bergmant/Documents/tm5-soa/output/raw/newsoa-ri/general_TM5_newsoa-ri_2010.lev1.nc"):
	modesigma={}
	modesigma['NUS']=1.59
	modesigma['AIS']=1.59
	modesigma['ACS']=1.59
	modesigma['COS']=2.00
	modesigma['AII']=1.59
	modesigma['ACI']=1.59
	modesigma['COI']=2.00
	rad=2.5
	testidata=read_num(filein)
	rwetdata=read_rwet(filein)
	timeax=read_timeaxis(filein)
	outdata=np.zeros_like(testidata['N_NUS'][:])
	#return testidata,rwetdata,timeax	
	for jj in testidata:
		rmode='RWET_'+jj[-3:]
		smode=jj[-3:]

		hr2=(0.5*np.sqrt(2.0))
		cmedr2mmedr= np.exp(3.0*(np.log(modesigma[smode])**2))
		rdata=np.where(rwetdata[rmode]<1e-20,1e-10,rwetdata[rmode][:])
		z=(np.log(rad)-np.log(rdata*1e9*cmedr2mmedr)/np.log(modesigma[smode]))
		modfrac=1-(0.5+0.5*erf(z*hr2))
		outdata+=testidata[jj][:]*modfrac
		#if jj=='N_NUS':
		#	raw_input()
	return outdata,timeax
def read_N_R(filein="/Users/bergmant/Documents/tm5-soa/output/raw/newsoa-ri/general_TM5_newsoa-ri_2010.lev1.nc"):
	testidata=read_num(filein)
	rwetdata=read_rwet(filein)
	timeax=read_timeaxis(filein)
	outdata=np.zeros_like(testidata['N_NUS'][:])
	return testidata,rwetdata,timeax	

def modal_frac(rad,rwet,mode):
	modesigma={}
	modesigma['NUS']=1.59
	modesigma['AIS']=1.59
	modesigma['ACS']=1.59
	modesigma['COS']=2.00
	modesigma['AII']=1.59
	modesigma['ACI']=1.59
	modesigma['COI']=2.00
	modfrac=0
	hr2=(0.5*np.sqrt(2.0))
	cmedr2mmedr= np.exp(3.0*(np.log(modesigma[mode])**2))
	rdata=np.where(rwet[:]<1e-20,1e-10,rwet[:])
	z=(np.log(rad)-np.log(rdata*1e9*cmedr2mmedr)/np.log(modesigma[mode]))
	#print z, rad,rdata,cmedr2mmedr
	modfrac=1-(0.5+0.5*erf(z*hr2))
	return modfrac
def read_collocate_modeldata(observed_data):
	cutoffs=get_cutoffs()

	modeldata={}
	data_monthly={}
	for experiment in EXPS:
		model={}
		#test2,timeax=read_N(basepathraw+exp+'/general_TM5_'+exp+'_2010.lev1.nc')
		Number_modes,radii_modes,timeax=read_N_R(basepathraw+experiment+'/general_TM5_'+experiment+'_2010.lev1.nc')
		for isite in observed_data:
			# if cutoff is not defined use 5um (diameter of 10um)
			if np.isnan(cutoffs[isite][1]):
				sitecutoff=5
			else: # otherwise use the value
				sitecutoff=cutoffs[isite][1]*1e9/2
			sitecutoff=5 #radius -> Dp=10nm
			#each site at a time
			CN_site_data=np.zeros_like(timeax)
			for N_mode in Number_modes:
				gridpoint_Nmode_data,lat,lon=emep_read.select_gp(Number_modes[N_mode],observed_data[isite]['lat'],observed_data[isite]['lon'])
				radius_mode_varname='RWET_'+N_mode[-3:]
				gridpoint_R_data,lat,lon=emep_read.select_gp(radii_modes[radius_mode_varname],observed_data[isite]['lat'],observed_data[isite]['lon'])
				modal_fraction= modal_frac(sitecutoff,gridpoint_R_data,radius_mode_varname[-3:])
				CN_site_data[:]+=gridpoint_Nmode_data*modal_fraction
				
			#m1m,m1s,t1=colocate_time(gpdata,timeax,observed_data[isite]['starttime'],observed_data[isite]['endtime'],observed_data[isite]['CN'])
			CN,std,col_time=colocate_time(CN_site_data,timeax,observed_data[isite]['starttime'],observed_data[isite]['endtime'],observed_data[isite]['CN'])
			modeldata_site={'CN':CN,'std':std,'time':col_time,'lon':lon,'lat':lat}
			model[isite]=modeldata_site
		modeldata[experiment]=model


		data_monthly[experiment]=monthly_aggregation(modeldata[experiment])
	data_monthly['obs']=monthly_aggregation(observed_data)
	return modeldata
def	annual_aggregate(modeldata,observed_data):
	annual_data={}
	for i in EXPS:
		annual_data[i]=np.zeros(len(observed_data))
	annual_data['obs']=np.zeros(len(observed_data))

	for exp in EXPS:
		for i,site in enumerate(modeldata['newsoa-ri']):
			annual_data[exp][i]=np.nanmean(modeldata[exp][site]['CN'])
	for i,site in enumerate(observed_data):
		annual_data['obs'][i]=np.nanmean(observed_data[site]['CN'])
	return annual_data
def scatter_plot_fig8(annual_data):
	R1= pearsonr(annual_data['obs'],annual_data['newsoa-ri'])
	R2= pearsonr(annual_data['obs'],annual_data['oldsoa-bhn'])

	f_sc,ax_sc=plt.subplots(1,figsize=(6,6))
	ax_sc.plot(annual_data['obs'],annual_data['newsoa-ri'],'or')
	ax_sc.loglog(annual_data['obs'],annual_data['oldsoa-bhn'],'xb')
	ax_sc.annotate(('NEWSOA: %6.2f')%R1[0],xy=(.01,0.9),xycoords='axes fraction')
	ax_sc.annotate(('OLDSOA: %6.2f')%R2[0],xy=(.01,0.85),xycoords='axes fraction')
	ax_sc.plot([0,10000],[0,10000],'-k')
	ax_sc.plot([0,5000],[0,10000],'--g')
	ax_sc.plot([0,10000],[0,5000],'--g')
	ax_sc.set_xlabel('Observed number concentration [cm-3]')
	ax_sc.set_ylabel('Modelled number concentration [cm-3]')
	ax_sc.set_xlim(5e1,1e4)
	ax_sc.set_ylim(5e1,1e4)
	ax_sc.set_aspect('equal', 'box')
	f_sc.savefig(output_png_path+'article/fig8_scatter_CN.png',dpi=pngdpi)
def main():
	observed_data=cnRead()
	#list_stations(observed_data)


	modeldata=read_collocate_modeldata(observed_data)
	annual_data=annual_aggregate(modeldata,observed_data)

	increase_count = np.count_nonzero((annual_data['newsoa-ri']>annual_data['oldsoa-bhn']))
	decrease_count = np.count_nonzero((annual_data['newsoa-ri']<annual_data['oldsoa-bhn']))

	scatter_plot_fig8(annual_data)
	NMB1=NMB(annual_data['obs'],annual_data['newsoa-ri'])
	NMB2=NMB(annual_data['obs'],annual_data['oldsoa-bhn'])
	R1= pearsonr(annual_data['obs'],annual_data['newsoa-ri'])
	R2= pearsonr(annual_data['obs'],annual_data['oldsoa-bhn'])
	print 'NEWSOA:',NMB1*100,R1
	print 'OLDSOA:',NMB2*100,R2
	print 'decrease at N stations:',decrease_count
	print 'increase at N stations:',increase_count
	plt.show()

if __name__ == '__main__':
	main()
