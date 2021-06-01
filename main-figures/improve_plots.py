from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
#import sys
#sys.path.append(r'/Users/bergmant/Documents/python/tm5/')
#from lonlat import lonlat
from general_toolbox import get_gridboxarea,lonlat,write_netcdf_file
import matplotlib.gridspec as gridspec
#from netcdfutils import write_netcdf_file
import glob
import xarray as xr
from scipy.stats import pearsonr
import os
import pandas as pd
import improve_tools as improve_tools
import logging
#import plot_m7
from settings import *
SMALL_SIZE=12
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels# def site_type():
# 	# categorization from http://vista.cira.colostate.edu/improve/wp-content/uploads/2016/08/IMPROVE_V_FullReport.pdf
# 	types={}
# 	types['urban']=[10730023]
# 	types['suburban']
# 	types['rurla']
def list_stations(indict):
	tabletex=open(basepath+'/paper/sitelist_improve.tex','w')
	tabletex.write("%Name & Station code&Longitude& Latitude &Height\\\\\n")
	sortdict={}
	for i in sorted(indict.keys()):
		print len(indict[i][0])
		print ('%6s,%35s,%4s,%4s,%4s')%(i,indict[i][0],indict[i][1],indict[i][2],indict[i][3])
		stationname=indict[i][0]
		#print stationname
		stationname=stationname.replace("#","\\#")
		#print stationname
		sortdict[stationname]=[i,indict[i][1],indict[i][2],indict[i][3]]
	for j in sorted(sortdict):
		print ('%-35s,%6s,%4s,%4s,%4s')%(j,sortdict[j][0],sortdict[j][2],sortdict[j][1],sortdict[j][3])
		tabletex.write("%s&%6s&%4s&%4s&%4s\\\\\n"%(j,sortdict[j][0],sortdict[j][2],sortdict[j][1],sortdict[j][3]))
	tabletex.close()

def str_months():
	return ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
def monthly_aggregation(sitedata):
	meanmodelmon={}
	meanmodelmon_pom={}
	meanmodelmon_C={}
	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_pom[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_C[exp]=np.zeros((len(sitedata.keys()),12))
	meanmodelmon['obs']=np.zeros((len(sitedata.keys()),12))
	kk=0
	for i in sitedata:
		#print i
		#sitedata		
		obs=sitedata[i][6][1][:,1]
		timedata=sitedata[i][6][0][:]
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		monthdata=np.full(12,np.NAN)
		modelmonthdata=np.full(12,np.NAN)
		modelmonthdata_pom=np.full(12,np.NAN)
		modelmonthdata_soa=np.full(12,np.NAN)
		for j in timedata:
			monthindices[j.month].append(timedata.index(j))
		for k in monthindices:
			monthdata[k-1]=np.mean(sitedata[i][6][1][monthindices[k],1])
		meanmodelmon['obs'][kk,:]=monthdata[:]
		
		for exp in EXPS:
			model=nc.Dataset(basepathprocessed+'/improve_col/'+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(basepathprocessed+'/improve_col/'+exp+'_pom_'+i+'.nc','r').variables[i][:]
			for kmod in monthindices:
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdata_pom[kmod-1]=np.mean(model_pom[monthindices[kmod]])
			meanmodelmon[exp][kk,:]=modelmonthdata
			meanmodelmon_pom[exp][kk,:]=modelmonthdata_pom
			if len(obs)!=len(model):
				logger.debug('Site %s number of obs: %i model: %i',i,len(obs),len(model))
				exit()
			mfb=improve_tools.MFB(obs,model)
			mfe=improve_tools.MFE(obs,model)
			nmb=improve_tools.NMB(obs,model)
			nme=improve_tools.NME(obs,model)
			rmse=improve_tools.RMSE(obs,model)
			r=pearsonr(monthdata,modelmonthdata)
			mfbmon=improve_tools.MFB(monthdata,modelmonthdata)
			mfemon=improve_tools.MFE(monthdata,modelmonthdata)
			nmbmon=improve_tools.NMB(monthdata,modelmonthdata)
			nmemon=improve_tools.NME(monthdata,modelmonthdata)
			rmsemon=improve_tools.RMSE(monthdata,modelmonthdata)
			rmon=pearsonr(monthdata,modelmonthdata)
			print i,exp,'R: ',r
			#model=model_site[i]['OM']		
		kk+=1

	return meanmodelmon,meanmodelmon_pom
def calc_dens(inputdata):
	data=nc.Dataset(inputdata,'r')
	modes=['NUS','AIS','ACS','COS']
	comps=['SOA','POM','SO4','BC','DU','SS']
	densities={}
	densities['SOA']=1.300
	densities['POM']=1.300
	densities['SO4']=1.841
	densities['BC']=1.800
	densities['SS']=2.165
	densities['DU']=2.65
	roo_a={}
	#h2o=['NUS','AIS','ACS','COS']
	for m in modes:
		mass=np.zeros_like(data.variables['M_SOAAIS'])
		roo=np.zeros_like(data.variables['M_SOAAIS'])

		for c in comps:
			indexi='M_'+c+m 
			if indexi in data.variables:
				mass+=data.variables[indexi][:]
				roo+=data.variables[indexi][:]*densities[c]
		indexi='aerh2o3d_'+m
		mass+=data.variables[indexi][:]
		roo+=data.variables[indexi][:]*1.0 #denisty of water
		radius_index='RWET_'+m
		volume=(4/3*np.pi*(data.variables[radius_index][:]**3)*data.variables['N_'+m][:])
		dens=mass/volume
		roo=roo/mass
		roo_a[m]=roo
		
	return roo_a			

def main():
	logger=logging.getLogger('improve_plots_main')
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
	# add formatter to ch
	ch.setFormatter(formatter)

	# add ch to logger
	logger.addHandler(ch)
	sitedata=improve_tools.improve()
	list_stations(sitedata)
	EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']
	for exp in EXPS:
		improve_tools.read_model_data(exp,sitedata,logger)

	meanmodelmon={}
	meanmodelmon_pom={}
	meanmodelmon_C={}
	for exp in EXPS:
		meanmodelmon[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_pom[exp]=np.zeros((len(sitedata.keys()),12))
		meanmodelmon_C[exp]=np.zeros((len(sitedata.keys()),12))
	meanmodelmon['obs']=np.zeros((len(sitedata.keys()),12))
	kk=0
	paperpath=""
	tablex=open(paperpath+'stats_improve.txt','w')
	tablex.write("Name, R, NMB\n")
	tabletex=open(paperpath+'stats_improve.tex','w')
	tabletex.write("%Name & Obs&Model& R &NMB (\%)\n")
	table_layout2_tex=open(paperpath+'stats_improve.tex','w')
	table_layout2_tex.write("%Name & Obs&Model& R &NMB (\%)\n")
	for i in sitedata:
		#sitedata		
		obs=sitedata[i][6][1][:,1]
		timedata=sitedata[i][6][0][:]
		monthindices={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
		monthdata=np.zeros([12])
		monthdata[:]=np.NAN
		modelmonthdata=np.zeros([12])
		modelmonthdata[:]=np.NAN
		modelmonthdata_pom=np.zeros([12])
		modelmonthdata_pom[:]=np.NAN
		modelmonthdata_soa=np.zeros([12])
		modelmonthdata_soa[:]=np.NAN
		# save indices for a given (jth) month indices in list
		for j in timedata:
			monthindices[j.month].append(timedata.index(j))
		# get monthly data from sitedata based on inidices
		for k in monthindices:
			monthdata[k-1]=np.mean(sitedata[i][6][1][monthindices[k],1])
		#kk=index(site)
		meanmodelmon['obs'][kk,:]=monthdata[:]

		for exp in EXPS:
			model=nc.Dataset(path_improve_col+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(path_improve_col+exp+'_pom_'+i+'.nc','r').variables[i][:]
			for kmod in monthindices:
				#print kmod
				modelmonthdata[kmod-1]=np.mean(model[monthindices[kmod]])
				modelmonthdata_pom[kmod-1]=np.mean(model_pom[monthindices[kmod]])
			meanmodelmon[exp][kk,:]=modelmonthdata
			meanmodelmon_pom[exp][kk,:]=modelmonthdata_pom
			if len(obs)!=len(model):
				logger.debug('Site %s number of obs: %i model: %i',i,len(obs),len(model))
				exit()
			mfb=improve_tools.MFB(obs,model)
			mfe=improve_tools.MFE(obs,model)
			nmb=improve_tools.NMB(obs,model)
			nme=improve_tools.NME(obs,model)
			rmse=improve_tools.RMSE(obs,model)
			r=pearsonr(monthdata,modelmonthdata)
			mfbmon=improve_tools.MFB(monthdata,modelmonthdata)
			mfemon=improve_tools.MFE(monthdata,modelmonthdata)
			nmbmon=improve_tools.NMB(monthdata,modelmonthdata)
			nmemon=improve_tools.NME(monthdata,modelmonthdata)
			rmsemon=improve_tools.RMSE(monthdata,modelmonthdata)
			rmon=pearsonr(monthdata,modelmonthdata)
			
		kk+=1
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	# fmean,amean=plt.subplots(ncols=2,figsize=(12,4))
	colors=['red', 'blue','black'] # newsoa, oldsoa, obs
	shadingcolors=['#ff000033', '#00ff0033','#0000ff33','#55555533']
	for n,exp in enumerate(EXPS[:],0):	
		if n==0:
			# amean[n].set_title('NEWSOA')
			labeli='NEWSOA'
		elif n==1:
			# amean[n].set_title('OLDSOA')
			labeli='OLDSOA'
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		nmbmean=improve_tools.NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		obsmean=np.nanmean(meanmodelmon['obs'])
		expmean=np.nanmean(meanmodelmon[exp])

		tablex.write(" %6s , %6.2f, %6.2f, %6.0f\\%% , %6.2f\n"%(labeli, obsmean, expmean,nmbmean*100, rmean[0]))
		tabletex.write("& %6s & %6.2f & %6.2f& %6.0f\\%% & %6.2f\\\\\n"%(labeli, obsmean, expmean,nmbmean*100, rmean[0]))
		table_layout2_tex.write("& %6s & %6.2f & %6.2f& %6.0f\\%% & %6.2f\\\\\n"%(labeli, obsmean, expmean,nmbmean*100, rmean[0]))
		#\\unit{\\mu gm^{-3}}
		print exp,n,obsmean,expmean
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)

	k=-1
	yearmean_model={}
	yearmean_model_pom={}
	for exp in EXPS:
		k+=1
		yearmean_obs=[]
		# f,ax=plt.subplots(1)
		for i in sitedata:
			#print i
			model=nc.Dataset(path_improve_col+exp+'_'+i+'.nc','r').variables[i][:]
			model_pom=nc.Dataset(path_improve_col+exp+'_pom_'+i+'.nc','r').variables[i][:]
			#model=model_site[i]['OM']		
			##### trying to get yearly means in dict
			if exp not in yearmean_model.keys():
				yearmean_model[exp]=[]
				yearmean_model[exp].append(np.mean(model))
				yearmean_model_pom[exp]=[]
				yearmean_model_pom[exp].append(np.mean(model))
			else:
				yearmean_model[exp].append(np.mean(model))
				yearmean_model_pom[exp].append(np.mean(model))
			
			####
			#yearmean_model[exp].append(np.mean(model))
			yearmean_obs.append(np.mean(sitedata[i][6][1][:,1]))
			#print type(model)
			if not 'temp_mod' in locals():
				temp_mod=model.copy()
			else:			
				temp_mod=np.concatenate((all_model,model))
			N_mod=np.shape(temp_mod)
			all_model=temp_mod.copy()
			if not 'temp_obs' in locals():
				temp_obs=sitedata[i][6][1][:,1].copy()
			else:			
				temp_obs=np.concatenate((all_obs,sitedata[i][6][1][:,1]))
			N_obs=np.shape(temp_obs)
			if N_obs!=N_mod:
				print N_mod,N_obs
				exit()
			all_obs=temp_obs.copy()
			# ax.loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
			# axb[k].loglog(sitedata[i][6][1][:,1],model,'or',ms=2)
			
			xmax=max(max(model),max(sitedata[i][6][1][:,1]))
			xmin=min(min(model),min(sitedata[i][6][1][:,1]))
			ymax=xmax
			ymin=xmin
		print 'N '+exp+':',len(yearmean_model[exp]),len(yearmean_obs)
		if len(yearmean_model[exp])>1 and len(yearmean_obs)>1 and False:

			# f,ax=plt.subplots(1)
			print yearmean_model.keys,len(yearmean_model[exp]),len(yearmean_obs)
			# ax.loglog(yearmean_obs,yearmean_model[exp],'ob')
			# ax2[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
			# if k<2:
			# 	ax2b[k].loglog(yearmean_obs,yearmean_model[exp],'or',ms=2)
			xmax=max(max(yearmean_model[exp]),max(yearmean_obs))
			xmin=min(min(yearmean_model[exp]),min(yearmean_obs))
			mfb=improve_tools.MFB(yearmean_obs,yearmean_model[exp])
			mfe=improve_tools.MFE(yearmean_obs,yearmean_model[exp])
			nmb=improve_tools.NMB(yearmean_obs,yearmean_model[exp])
			nme=improve_tools.NME(yearmean_obs,yearmean_model[exp])
			rmse=improve_tools.RMSE(yearmean_obs,yearmean_model[exp])
			r=pearsonr(yearmean_obs,yearmean_model[exp])
			r_all=pearsonr(all_obs,all_model)
			mfb_all=improve_tools.MFB(all_obs,all_model)
			mfe_all=improve_tools.MFE(all_obs,all_model)
			nmb_all=improve_tools.NMB(all_obs,all_model)
			nme_all=improve_tools.NME(all_obs,all_model)
			rmse_all=improve_tools.RMSE(all_obs,all_model)
			ymax=xmax
			ymin=xmin
			print exp,'rmse: ',rmse,rmse_all
			print exp,'mfb: ',mfb,mfb_all
			print exp,'mfe: ',mfe,mfe_all
			print exp,'nmb: ',nmb,nmb_all
			print exp,'nme: ',nme,nme_all
	fmean,amean=plt.subplots(ncols=2,nrows=2,figsize=(12,8))
	colors=['red', 'blue','black']
	shadingcolors=['#ff000033', '#0000ff33','#00ff0033','#55555533']
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)+std,color=shadingcolors[2])
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0)-std,color=shadingcolors[2])
	#amean[0,0].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0),color=colors[2])
	amean[0,0].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
	amean[0,0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,0].set_ylim([0,2.0])

	amean[0,1].errorbar(np.linspace(1,12,12),np.nanmean(meanmodelmon['obs'],axis=0), yerr=[std, std], fmt='--o',color=shadingcolors[3],label='observations')
		#amean[2].set_title('Observations')
	amean[0,1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
	amean[0,1].set_ylim([0,2.0])
	letters=[['a','b'],['c','d']]
	for n,exp in enumerate(EXPS[:],0):	
		if n==0:
			labeli='NEWSOA'
			amean[0,n].set_title('NEWSOA')
			colori='r'
		elif n==1:
			labeli='OLDSOA'
			amean[0,n].set_title('OLDSOA')
			colori='b'
		print n, exp
		std=np.nanstd(meanmodelmon[exp],axis=0)
		maxi=np.nanmax(meanmodelmon[exp],axis=0)
		mini=np.nanmin(meanmodelmon[exp],axis=0)
		amean[0,n].fill_between(np.linspace(1,12,12), np.nanmean(meanmodelmon[exp],axis=0)-std, np.nanmean(meanmodelmon[exp],axis=0)+std, color=shadingcolors[n],alpha=0.3)
		
		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon[exp],axis=0),color=colors[n],label=labeli+' POA+SOA')
		amean[0,n].plot(np.linspace(1,12,12),np.nanmean(meanmodelmon_pom[exp],axis=0),color=colors[n],ls='--',label=labeli+' POA')
		amean[0,n].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
		amean[0,n].set_xticklabels(str_months())
		if n==0:
			amean[0,n].set_ylim([0,2.0])
		else:
			amean[0,n].set_ylim([0,2.0])
		#amean[n].set_title(exp)
		print np.shape(meanmodelmon['obs'])
		print np.shape(yearmean_obs)

		nmbmean=improve_tools.NMB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		mbmean=improve_tools.MB(np.nanmean(meanmodelmon['obs'],axis=0),np.nanmean(meanmodelmon[exp],axis=0))
		rmean=pearsonr(np.nanmean(meanmodelmon[exp],axis=0),np.nanmean(meanmodelmon['obs'],axis=0))
		amean[0,n].annotate(('NMB (MB): %6.1f %% (%4.2f)')%(nmbmean*100,mbmean),xy=(0.01,0.95),xycoords='axes fraction',fontsize=12)
		amean[0,n].annotate(('R: %6.2f')%(rmean[0]),xy=(0.01,0.9),xycoords='axes fraction',fontsize=12)
		amean[0,n].set_xlabel('Month',fontsize=12)
		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
		amean[0,n].set_ylabel('OM[pm25][ug m-3]',fontsize=12)
		amean[0,n].annotate(('%s)')%(letters[0][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		amean[0,n].legend(loc='upper right',fontsize=12)
		#ax.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
		print np.nanmean(meanmodelmon[exp],axis=0)
		print 'teststste',n,exp
		amean[1,n].loglog(yearmean_obs,yearmean_model[exp],'o',c=colori,ms=3)
		amean[1,n].plot([0.0001,1000],[0.0001,1000],'k')
		amean[1,n].plot([0.0001,1000],[0.001,10000],'k--')
		amean[1,n].plot([0.001,10000],[0.0001,1000],'k--')
		amean[1,n].set_ylim([.5e-2,3e0])
		amean[1,n].set_xlim([.5e-2,3e0])
		amean[1,n].set_xlabel('IMPROVE OM[pm25][ug m-3]',fontsize=12)
		#amean[0,n].set_ylabel('TM5:'+exp+' OM[pm25][ug m-3]')
		amean[1,n].set_ylabel(EXP_NAMEs[n] + ' OM[pm25][ug m-3]',fontsize=12)
		amean[1,n].set_aspect('equal')			
		#amean[1,n].legend(loc=4)
		nmbmean=improve_tools.NMB(yearmean_obs,yearmean_model[exp])
		mbmean=improve_tools.MB(yearmean_obs,yearmean_model[exp])
		rmean=pearsonr(yearmean_model[exp],yearmean_obs)
		rlogmean=pearsonr(np.log(yearmean_model[exp]),np.log(yearmean_obs))
		amean[1,n].annotate(('NMB (MB): %5.1f %% (%4.2f)')%(nmbmean*100,mbmean),xy=(0.20,0.06),xycoords='axes fraction',fontsize=12)
		amean[1,n].annotate(('R (R log): %6.2f, (%4.2f) ')%(rmean[0],rlogmean[0]),xy=(0.20,0.01),xycoords='axes fraction',fontsize=12)
		amean[1,n].annotate(('%s)')%(letters[1][n]),xy=(0.0,1.02),xycoords='axes fraction',fontsize=18)
		plt.tight_layout()

	
	std=np.nanstd(meanmodelmon['obs'],axis=0)
	maxi=np.nanmax(meanmodelmon['obs'],axis=0)
	mini=np.nanmin(meanmodelmon['obs'],axis=0)
	#fmean.suptitle('IMPROVE',fontsize=12)
	fmean.savefig(output_png_path+'/article/fig10_scatter-seasonal-IMPROVE-2x2.png',dpi=600)
	fmean.savefig(output_pdf_path+'/article/fig10_scatter-seasonal-IMPROVE-2x2.pdf')
	fmean.savefig(output_jpg_path+'/article/fig10_scatter-seasonal-IMPROVE-2x2.jpg',dpi=600)
	plt.show()


	
if __name__=='__main__':
	main()
