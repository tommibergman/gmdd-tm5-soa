import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
#import matplotlib as mpl
import sys
import os
#sys.path.append("/Users/bergmant/Documents/Project/ifs+tm5-validation/scripts")
from colocate_aeronet import do_colocate
import matplotlib as mpl
import read_colocation_aeronet as ra
import datetime
import scipy.stats as stats
import glob
import subprocess
from bivariate_fit import bivariate_fit
from matplotlib.colors import LogNorm
import xarray as xr
import re
import cis
from matplotlib.patches import Polygon
from settings import *
from general_toolbox import str_months,RMSE,MFE,MFB,NMB,NME
import string
# def RMSE(obs,model):
# 	N=0
# 	RMSE=0.0
# 	for o,m in zip(obs,model):	
# 		if not (np.isnan(o) or np.isnan(m)):
# 			#print m,o
# 			RMSE+=(m-o)**2
# 			N+=1
# 	return np.sqrt(RMSE/N)
# def MFE(obs,model):
# 	N=0
# 	MFE=0.0
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			MFE+=np.abs(m-o)/((o+m)/2)
# 			N+=1
# 	return MFE/N
# def MFB(obs,model):
# 	N=0
# 	MFB=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			MFB+=(m-o)/((o+m)/2)
# 			N+=1
# 	return MFB/N
# def NMB(obs,model):
# 	N=0
# 	nom=0.0
# 	denom=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		#print o,m,type(o),type(m)
# 		if not (np.isnan(o) or np.isnan(m)):
# 			nom+=(m-o)
# 			denom+=o
# 			N+=1
# 	NMB=nom/denom
# 	return NMB
# def NME(obs,model):
# 	N=0
# 	nom=0.0
# 	denom=0.0
# 	#print obs,model
# 	for o,m in zip(obs,model):
# 		if not (np.isnan(o) or np.isnan(m)):
# 			nom+=np.abs(m-o)
# 			denom+=o
# 			N+=1
# 	NME=nom/denom
# 	return NME

def concatenate_sites(data1,data2,data3):
	TM5=[]
	TM5_OLD=[]
	Aeronet=[]
	count1=0
	std=[]
	stderr=[]
	relstderr=[]

	for i in data1:
		#print i, data1
		#print data1[i][5],data2[i][5],data3[i][5]
		#print '1',data1[i][4]
		#print 'd',data3[i][4]
		#print type(data1[i][4])
		#print data1[i][4][1]

		if type(data1[i][4]) != int:
			#if len(data1[i][4]) != len(data3[i][4]):
			#	print len(data1[i][4])
			#	print len(data3[i][4])
			#print np.std(data1[i][4]),np.mean(data1[i][4]),np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))),(np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))/np.mean(data1[i][4]),len(TM5data[i][4])
			#print stats.bayes_mvs(data1[i][4])

			std.append(np.std(data1[i][4]))
			stderr.append(np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))
			relstderr.append((np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))/np.mean(data1[i][4]))

			#print len(data1[i]),len(data1[i][4])
			#print len(data3[i]),len(data3[i][4])
			if all(data1[i][4]) is not np.ma.masked:
				for j in range(len(data1[i][4])):
					#if 'Nauru' in i:
					#	print len(data1[i][4])
					#	print j,data1[i][4][j],data2[i][4][j],data3[i][4][j]
					#print i,j, len(data1[i])
					if data1[i][4][j] is not np.ma.masked and data2[i][4][j] is not np.ma.masked and data3[i][4][j] is not np.ma.masked and data1[i][4][j] >= 0 and data2[i][4][j] >= 0 and data3[i][4][j]>=0 :			
						TM5.append(data1[i][4][j])
						TM5_OLD.append(data2[i][4][j])
						Aeronet.append(data3[i][4][j])
					else:
						#print j
						#print data1[i][4][j]
						if data3[i][4][j] is not np.ma.masked: 	
							print 'ERROR'
							exit()
							pass
							#print 'tm5',i,j,data1[i][4][j],data1[i][5],data1[i][3][j]
							#print 'tm5 aero',i,j,data3[i][4][j],data3[i][5],data3[i][3][j]
							#raw_input()
						count1+=1
			else:
				print data1[i][4][j],data2[i][4][j],data3[i][4][j]
				
		else: #Yearly data, only 1 data point in data[i][4], 
			#print data1[i][4] ,data2[i][4] ,data3[i][4] 
			if data1[i][4] is not np.ma.masked:
				#for j in range(len(data1[i][4])):
				if data1[i][4] is not np.ma.masked and data2[i][4] is not np.ma.masked and data3[i][4] is not np.ma.masked and data1[i][4] >= 0 and data2[i][4] >= 0 and data3[i][4]>=0 :			
					TM5.append(data1[i][4])
					TM5_OLD.append(data2[i][4])
					Aeronet.append(data3[i][4])
				else:
					if data3[i][4] is not np.ma.masked: 	
						print 'ERROR'
						exit()
						pass
						#print 'tm5',i,j,data1[i][4],data1[i][5],data1[i][3]
						#print 'tm5 aero',i,j,data3[i][4],data3[i][5],data3[i][3]
						#raw_input()
					count1+=1

	return TM5,TM5_OLD,Aeronet
def concatenate_sites2(data):
	outdata=[]
	count1=0
	std=[]
	stderr=[]
	relstderr=[]

	count2=0
	for i in data:
		#print data[i][4]#,len(data3[i][4])
		#print np.std(data1[i][4]),np.mean(data1[i][4]),np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))),(np.std(data1[i][4])/np.sqrt(float(len(data1[i][4]))))/np.mean(data1[i][4]),len(TM5data[i][4])
		#print stats.bayes_mvs(data1[i][4])

		std.append(np.std(data[i][4]))
		stderr.append(np.std(data[i][4])/np.sqrt(float(len(data[i][4]))))
		relstderr.append((np.std(data[i][4])/np.sqrt(float(len(data[i][4]))))/np.mean(data[i][4]))

		count1=0
		#raw_input()
		for j in range(len(data[i][4])):
			#if data[i][4][j] is not np.ma.masked and data[i][4][j] > 0 :			
			if not np.ma.is_masked(data[i][4][j]):			
				outdata.append(data[i][4][j])
				count1+=1
				#if 'Nauru' in i:
				#	print len(data[i][4])
				#	print i,data[i][4][j]

			else:
				#print 'tm5',i,j,data[i][4][j],data[i][5]
				pass
				#count1+=1
		count2+=count1
		#print i,count1
	#print count2
	return outdata
def sites2numpy(data):
	outdata=[]
	count1=0
	std=[]
	stderr=[]
	relstderr=[]
	#print np.shape(data)
	#print np.shape(data[0])
	sitesnumpy=np.zeros(np.shape(data))
	for i in range(len(data)):
		
		print data[i][:]
		sitesnumpy[i,:]=data[i][:]
		#raw_input()
		# for j in range(len(data[i][4])):
		# 	print j
		# 	if data[i][4][j] is not np.ma.masked and data[i][4][j] > 0 :			
		# 		outdata.append(data[i][4][j])
		# 	else:
		# 		print 'tm5',i,j,data[i][4][j],data[i][5]
		# 		count1+=1
	print sitesnumpy
	asdfa
	return outdata


def scatter_heat(obs,model,obsname=None,modelname=None):
	plt.figure()
	cm=plt.get_cmap('BuPu')
	if obsname==None:
		obsname='Aeronet'
	if modelname==None:
		modelname='TM5'
	print np.shape(obs),np.shape(model)
	plt.hist2d(obs,model,bins=100,norm=LogNorm(),cmap=cm)
	XX=np.max([np.max(obs),np.max(model)])
	XX=1.0
	plt.xlim(0,XX)
	plt.ylim(0,XX)
	l2,=plt.plot([0,XX],[0,XX],'--k',lw=3)
	fit = np.polyfit(obs, model, 1)
	YY=np.poly1d(fit)
	#aa,bb,SS=bivariate_fit(Aeronetb,TM5b,0.01,0.003+0.001*TM5b,0.0,1.0)
	print fit
	slope,intercept,r,p,stderr=stats.linregress(obs, model)
	print slope,intercept,r,p,stderr
	xxx=np.logspace(-3,1,1000)
	l3,=plt.plot(xxx,YY(xxx),'-b',lw=4)
	#l55,=plt.plot(xxx,bb*xxx+aa,'-g',lw=4)
	plt.title('Individual observations',fontsize=14)
	plt.xlabel(obsname+' 2010 per point',fontsize=14)
	plt.ylabel(modelname+' collocated 2010 per point',fontsize=14)
	plt.legend([l2,l3],['1:1 line','Fitted'],loc=2,numpoints=1,fontsize=14)
	#plt.legend([l1,l2,l3],['Colocated annual mean','1:1 line','Fitted'],loc=2,numpoints=1)
	plt.annotate('Slope:         %6.2f\nintercept:   %6.2f\nR:               %6.2f'%(slope,intercept,r), 
				xy=(0.02, 0.67), xycoords='axes fraction'
				,fontsize=14)
	#plt.annotate('Slope:         %6.2f\nintercept:   %6.2f\nR:               %6.2f'%(bb,aa,-1), 
	#            xy=(0.02, 0.27), xycoords='axes fraction'
	#            ,fontsize=14)
	cb=plt.colorbar()
	cb.set_label('Number of observations')
	#plt.figure()
def scatter_dot(obs,model,**kwargs):
	if 'ax' in kwargs:
		ax=kwargs['ax']
	else:
		fig,ax=plt.subplots(1)
	if 'ms' in kwargs:
		msize=kwargs['ms']
	else:
		msize=5
	if 'col' in kwargs:
		col=kwargs['col']
	else:
		col='b'
	if 'obsname' in kwargs:
		obsname=kwargs['obsname']
	else:
		obsname='Aeronet'
	if 'modelname' in kwargs:
		modelname=kwargs['modelname']
	else:
		modelname='TM5'

	if 'label' in kwargs:
		label=kwargs['label']
	else:
		label=None
	print label
	#print obs,model
	ax.scatter(obs,model,c=col,s=msize, label=label)
	XX=np.max([np.max(obs),np.max(model)])
	XX=1.0
	ax.set_xlim(0.01,XX)
	ax.set_ylim(0.01,XX)
	ax.set_xlim(0.01,1.0)
	ax.set_ylim(0.01,1.0)
	l2,=ax.plot([0,XX],[0,XX],'--k',lw=1.5)
	mask = ~np.isnan(obs) & ~np.isnan(model)
	fit = np.polyfit(obs[mask], model[mask], 1)
	YY=np.poly1d(fit)
	#aa,bb,SS=bivariate_fit(Aeronetb,TM5b,0.01,0.003+0.001*TM5b,0.0,1.0)
	print fit
	# mask out nans in obs and model for both arrays
	mask = ~np.isnan(obs) & ~np.isnan(model)
	slope,intercept,r,p,stderr=stats.linregress(obs[mask], model[mask])
	print slope,intercept,r,p,stderr
	xxx=np.logspace(-3,1,1000)
	#l3,=plt.plot(xxx,YY(xxx),'-b',lw=4)
	#l55,=plt.plot(xxx,bb*xxx+aa,'-g',lw=4)
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	ax.set_title('Annual means',fontsize=14)
	ax.set_xlabel(obsname+' 2010 ',fontsize=14)
	ax.set_ylabel(modelname+' collocated 2010 ',fontsize=14)
	#ax.legend([l2,l3],['1:1 line','Fitted'],loc=2,numpoints=1,fontsize=14)
	#plt.legend([l1,l2,l3],['Colocated annual mean','1:1 line','Fitted'],loc=2,numpoints=1)
	#plt.annotate('Slope:         %6.2f\nintercept:   %6.2f\nR:               %6.2f'%(slope,intercept,r), 
	#			xy=(0.02, 0.67), xycoords='axes fraction'
	#			,fontsize=14)
	#plt.figure()
	return slope,intercept,r,p,stderr
def get_regions():
	regions={}
	#regions['US']=[(49,-129),(24,-65)]
	regions['NA']=[(75,-155),(24,-55)]
	regions['EU']=[(70,-16),(39,45)]
	regions['SA']=[(0,-86),(-35,-36)]
	regions['AUS']=[(-12,110),(-40,156)]
	regions['SEAS']=[(45,70),(10,122)]
	regions['SIB']=[(70,60),(50,166)]
	regions['SAH']=[(34,-18),(13,61)]
	regions['AFR']=[(13,-18),(-34,51)]
	return regions
def get_region_names():
	regions={}
	regions['US']='USA'
	regions['BONA']='Boreal North America'
	regions['NA']='North America'
	regions['EU']='Eurasia'
	regions['SA']='South America'
	regions['AUS']='Australia'
	regions['SEAS']='South Eastern Asia'
	regions['SIB']='Boreal Russia'
	regions['SAH']='Norhern Africa/Middle East'
	regions['AFR']='Sub-Saharan Africa'
	regions['ALL']='All Stations'
	return regions
def group_by_region(data):
	regions=get_regions()
	outdata={}
	#outdata['US']={}
	#outdata['BONA']={}
	outdata['NA']={}
	outdata['EU']={}
	outdata['SA']={}
	outdata['AUS']={}
	outdata['SEAS']={}
	outdata['SIB']={}
	outdata['SAH']={}
	outdata['AFR']={}
	#print data
	for i in data:
		#print data[i]
		if not np.ma.is_masked(data[i]):
			#print data[i]
			for reg in regions:
				#print regions[reg][0][1],  data[i][0],regions[reg][1][1] ,regions[reg][1][0], data[i][1],regions[reg][0][0]
				#print regions[reg][0][1]<  data[i][0]<regions[reg][1][1] and regions[reg][1][0]< data[i][1]<regions[reg][0][0]
				#print regions[reg][0][1]<  data[i][0]<regions[reg][1][1] , regions[reg][1][0]< data[i][1]<regions[reg][0][0]
				if regions[reg][0][1]<  data[i][0]<regions[reg][1][1] and regions[reg][1][0]< data[i][1]<regions[reg][0][0]: 
					outdata[reg][i]=data[i]	
		else:
			print 'data is not masked'

			# if regions['US'][0][1]<  data[i][0]<regions['US'][1][1] and regions['US'][1][0]< data[i][1]<regions['US'][0][0]: 
			# 	print regions['US'][0][1],  data[i][1],regions['US'][1][1]
			# 	print regions['US'][0][0], data[i][2],regions['US'][1][0]
			# 	outdata['US'].append(data[i])	
			# elif regions['EU'][0][1]<  data[i][0]<regions['EU'][1][1] and regions['EU'][1][0]< data[i][1]<regions['EU'][0][0]: 
			# 	print regions['EU'][0][1],  data[i][1],regions['EU'][1][1]
			# 	print regions['EU'][0][0], data[i][2],regions['EU'][1][0]
			# 	outdata['EU'].append(data[i])	
			# elif regions['SA'][0][1]<  data[i][0]<regions['SA'][1][1] and regions['SA'][1][0]< data[i][1]<regions['SA'][0][0]: 
			# 	print regions['SA'][0][1],  data[i][1],regions['SA'][1][1]
			# 	print regions['SA'][0][0], data[i][2],regions['SA'][1][0]
			# 	outdata['SA'].append(data[i])	
	return outdata
def plot_rectangle(bmap, lonmin,lonmax,latmin,latmax,color):
	# Andrew Straw in stackOverflow: 
	# https://stackoverflow.com/questions/12251189/how-to-draw-rectangles-on-a-basemap
    xs = [lonmin,lonmax,lonmax,lonmin,lonmin]
    ys = [latmin,latmin,latmax,latmax,latmin]
    x,y=m([lonmin,lonmin,lonmax,lonmax],[latmin,latmax,latmax,latmin])
    xy=zip(x,y)
    #bmap.plot(xs, ys,latlon = True)
    poly = Polygon( xy, facecolor=color)
    plt.gca().add_patch(poly)
def biasmap(data,aerodata,bounds=None,**kwargs):
	if not 'ax' in kwargs:
		f,ax=plt.subplots(1,figsize=(10,7))
	else:
		ax=kwargs['ax']

	m = Basemap(projection='robin',lon_0=0)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,120.,30.))
	m.drawmeridians(np.arange(0.,3060.,60.))
	mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize
	if bounds==None:
		bounds = [-1.1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1]
	norm = mpl.colors.BoundaryNorm(bounds, 11)
	count=0
	for i in range(len(aerodata)):
		#print i[0][0],i[1][0]
		x1,y1=m(aerodata[i][0][0],aerodata[i][1][0])
		
		diff=(np.ma.mean(data[i][4][:])-np.ma.mean(aerodata[i][4][:]))/np.ma.mean(aerodata[i][4][:])
		if np.ma.min(data[i][4]) is np.ma.masked:
			count+=1
			continue
		m.scatter(x1,y1,marker='o',c=diff,s=300*(np.mean(aerodata[i][4][:])),norm=norm,cmap = mycmap )
		xx,yy=m(-160,-70)
		m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
		xx,yy=m(-150,-70)
		plt.text(xx,yy,'AOD 1.0')
		xx,yy=m(-160,-60)
		m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
		xx,yy=m(-150,-60)
		plt.text(xx,yy,'AOD 0.5')
	
	c = plt.colorbar(orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
	c.ax.tick_params(labelsize=10)
	c.set_label('AOD relative bias [(TM5-AERONET)/AERONET]')
	return f,ax,c
def map_bias(obs,model,**kwargs):
	if 'ax'not in kwargs:
		f,axs=plt.subplots(1,figsize=(10,7))
	else:
		axs=kwargs['ax']
	if 'boundmax'in kwargs:
		boundmax=kwargs['boundmax']
	#print 'axs ',axs
	m = Basemap(projection='robin',lon_0=0,ax=axs)
	m.drawcoastlines()
	m.drawparallels(np.arange(-90.,91.,30.))
	m.drawmeridians(np.arange(-180.,181.,60.))
	mycmap=plt.get_cmap('coolwarm',11) 
	# define the bins and normalize

	bounds = [-0.45,-0.35,-0.25,-0.15,-0.05,-0.02,0.02,0.05,0.15,0.25,0.35,0.45]
	bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
	#print obs
	if 'boundmax' not in kwargs:
		temp=0.0
		boundmax=0.0
		for i in obs:
			#print i
			#print model
			#print obs
			temp=max(np.amax(np.mean(model[i][4][:])-np.mean(obs[i][4][:])),np.abs(np.amin(np.mean(model[i][4][:])-np.mean(obs[i][4][:]))))
			#temp=max(np.amax((np.mean(model[i][4][:])-np.mean(obs[i][4][:]))/np.mean(obs[i][4][:])),np.abs(np.amin((np.mean(model[i][4][:])-np.mean(obs[i][4][:]))/np.mean(obs[i][4][:]))))
			if temp >boundmax:
				boundmax=temp
		if boundmax< 0.375:
			bounds=np.linspace(-boundmax,boundmax,12)
		#boundmax=1.1
	if 'bounds' not in kwargs:
		bounds=np.linspace(-boundmax,boundmax,12)
	else:
		bounds=kwargs['bounds']
	norm = mpl.colors.BoundaryNorm(bounds, len(bounds)-1)
	##
	#intitialise
	if 'name1' in kwargs:
		name1=kwargs['name1']
	if 'name2' in kwargs:
		name2=kwargs['name2']
	#print obs[0][4]
	annual_obs=[]
	annual_model=[]
	if 'ref' in kwargs:
		ref=kwargs['ref']
	else:
		ref=obs
	for i in obs:
		#print i[0][0],i[1][0]
		#print i
		x1,y1=m(obs[i][0][0],obs[i][1][0])
		#print np.shape(model),np.shape(obs)
		#if kwargs['lrelative']:

		diff=np.mean(model[i][4][:])-np.mean(obs[i][4][:])	
		reldiff=(np.mean(model[i][4][:])-np.mean(obs[i][4][:]))/np.mean(obs[i][4][:])	
		if reldiff>boundmax:
			print i,reldiff
		#print model[i][4][:],np.shape(model[i][4][:]),model[i][5]
		#for j in model[i][4]:
		#	print type(j)
		print 'no. datapoints with data: ',np.ma.count(model[i][4][:])
		days_data=np.ma.count(model[i][4][:])
		if np.ma.min(model[i][4]) is np.ma.masked:
			continue
		#print i,diff,model[i][4][:]
		annual_model.append(np.mean(model[i][4][:]))
		annual_obs.append(np.mean(obs[i][4][:]))
		#print obs[i][4]
		#m.scatter(x1,y1,marker='o',c=diff,s=300*(np.mean(obs[i][4][:])),norm=norm,cmap = mycmap )
		cs=m.scatter(x1,y1,marker='o',c=reldiff,s=300*(np.mean(ref[i][4][:])),norm=norm,cmap = mycmap )


	

	#patches=[]
	#homeplate = np.array([[-138,60],[-122,60],[-122,30],[-138,30]])
	#patches.append(mpl.patches.Polygon(homeplate))
	#ax.add_collection(mpl.collections.PatchCollection(patches,facecolor='green',edgecolor='k',linewidths=10,zorder=1))
	#plot_rectangle(m,-138,-122,30,60,'r')
	c = plt.colorbar(cs,orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.85,extend='both',ax=axs)
	c.ax.tick_params(labelsize=10)
	c.set_label('AOD relative change [('+name2+'-'+name1+')/'+name1+']')
	#c.cmap.set_over('k')
	#c.cmap.set_under('g')
	xx,yy=m(-160,-70)
	m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
	xx,yy=m(-150,-70)
	plt.text(xx,yy,'AOD 1.0')
	print xx,yy
	xx,yy=m(-160,-60)
	m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
	xx2,yy2=m(-150,-60)
	plt.text(xx2,yy2,'AOD 0.5')
	#x2, y2 = m(-130,-40)
	#plt.annotate('Barcelona', xy=(x2, y2),  xycoords='data',
    #            )
	return axs
def categorize(obs,model):
	overestimatemodel=[]
	overestimateobs=[]
	underestimatemodel=[]
	underestimateobs=[]
	goodmodel=[]
	goodobs=[]
	#print diff
	for i in obs:
		print i, model[i][4].data
		diff=np.mean(model[i][4].data)-np.mean(obs[i][4].data)
			
		if diff >0.1:
			overestimateobs.append(obs[i])
			overestimatemodel.append(model[i])
		if diff >-0.025 and diff <0.025:
			goodobs.append(obs[i])
			goodmodel.append(model[i])
		if diff <-0.1:
			underestimateobs.append(obs[i])
			underestimatemodel.append(model[i])
	return underestimatemodel,goodmodel,overestimatemodel,underestimateobs,goodobs,overestimateobs

def read_aggregated(inpath=None):
	import numpy as np 
	import netCDF4 as nc
	#from netCDF4 import netcdftime
	import glob
	#print inpath
	if inpath==None:
		print 'No input file'
		quit()
	outputdata={}
	print 'reading aggregated data from '+inpath
	for datafile in  glob.iglob(inpath+'*.nc'):
		#print 'path:',aeronet
		site=datafile.split('/')[-1].split('.')[0]
		site= re.sub('[0-9]{6}_[0-9]{6}_','',site)
		if 'South_Pole_Obs_NOAA' in datafile:
			print datafile
			print 'skipping south pole'
			continue
		ncdata=nc.Dataset(datafile,'r')
		#print ncdata.variables
		if 'AOT_500' in ncdata.variables:
			aotname='AOT_500'
		else:
			aotname='od550aer'
		stdname=aotname+'_std_dev'
		numname=aotname+'_num_points'
		lon=ncdata.variables['longitude'][:]
		lat=ncdata.variables['latitude'][:]
		alt=ncdata.variables['altitude'][:]
		aot=ncdata.variables[aotname][:]
		if stdname in ncdata.variables:
			aotstd=ncdata.variables[stdname][:]
		else:
			aotstd=None
		if numname in ncdata.variables:
			aotnum=ncdata.variables[numname][:]
		else:
			aotnum=None
		time=ncdata.variables['time'][:]
		unit_temps = ncdata.variables['time'].units 
		#print ncdata.variables['time'].ncattrs()

		if 'calendar' in ncdata.variables['time'].ncattrs():
			cal_temps = ncdata.variables['time'].calendar
		else:
			cal_temps='gregorian'
		datevar = []
		datevar.append(nc.num2date(time,units = unit_temps,calendar = cal_temps))
		#if aot[0] is not np.ma.masked: 
		outputdata[site]=lon,lat,alt,time,np.squeeze(aot),datafile,aotstd,aotnum,np.squeeze(datevar)
	return outputdata

def read_aeronet_all(inpath=None):
	import numpy as np 
	import netCDF4 as nc
	#from netCDF4 import netcdftime
	import glob
	if inpath==None:
		print 'no inpath'
		quit()
	TM5data=[]
	TM5_collocated=inpath
	for TM5 in  glob.iglob(TM5_collocated+'*'):
		#print TM5
		ncTM5=nc.Dataset(TM5,'r')
		if 'longitude' in ncTM5.variables:
			lon=ncTM5.variables['longitude'][:]
		else:	
			lon=ncTM5.variables['lon'][:]
		if 'latitude' in ncTM5.variables:
			lat=ncTM5.variables['latitude'][:]
		else:
			lat=ncTM5.variables['lat'][:]
		if 'altitude' in ncTM5.variables:
			alt=ncTM5.variables['altitude'][:]
		aot=ncTM5.variables['od550aer'][:]
		time=ncTM5.variables['time'][:]
		#if aot[0] is not np.ma.masked: 
		TM5data.append([lon,lat,alt,time,aot,TM5])
	return TM5data


def do_colocate(AODmodel = 'od550aer',AODdata = 'AOT_500',input_TM5_data = '/Users/bergmant/Documents/codes/anaconda/data/newdata/aerocom3_TM5_AP3-CTRL2016_global_2010_hourly.od550aer.nc',output= 'Collocated_aeronet_ctrl2016_lin/',aeronet='aeronet_2010/',year=2010):
	from datetime import timedelta
	model=cis.read_data(input_TM5_data,AODmodel)
	#for outputdata in glob.iglob(aeronet+'*'):
	wavelength=np.float(AODdata[0].split('_')[1])
	aeronet_out_path=aeronet_path+str(year)+'/AOT_550/'
	for aeronetdata in glob.iglob(aeronet_path+'/AOT_Level2_All_Points/AOT/LEV20/ALL_POINTS/*'):
		
			#outputname=os.path.basename(outputdata)[14:]
		# Strip the beginning code away.
		outputname=os.path.basename(aeronetdata)[14:]+'.nc'

		#aeronetinname=os.path.basename(aeronetdata)[14:]
		#print os.path.isfile(output+outputname)
		# read in the aeronet data for a given station
		# from existing NC-file if it is available
		#print 'testing aeronet '+outputname
		aeronet_aot=None

		# using nc-files does not work, cis will collocate also for missing data
		#if os.path.isfile('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname):
		#	aeronet_aot=cis.read_data('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname,AODdata)
		# or from original files 
		# needs a switch not to redo everytime
		#else:
			#print aeronet_aot
		print aeronet_out_path+'/all/'
		if not os.path.isdir(aeronet_out_path+'/all/'):
			os.mkdir(aeronet_out_path)
			os.mkdir(aeronet_out_path+'/all/')
		aeronet_aot=cis.read_data(aeronetdata,AODdata)
		#Subset wanted year
		aeronet_aot=aeronet_aot.subset(t=cis.time_util.PartialDateTime(year))
		# needed for interpolation of 550
		aeronet_ang=cis.read_data(aeronetdata,'440-675Angstrom')
		aeronet_ang=aeronet_ang.subset(t=cis.time_util.PartialDateTime(year))
		if aeronet_aot !=None:
			print wavelength
			print aeronet_aot.data
			print #aeronet_ang.data
			print #(550.0/500.0)**(-aeronet_ang.data)
			print aeronet_aot.data*(550.0/wavelength)**(-aeronet_ang.data)
			print #(aeronet_aot.data*(550.0/500.0)**(-aeronet_ang.data))/aeronet_aot.data
			aeronet_aot.data=aeronet_aot.data*(550.0/wavelength)**(-aeronet_ang.data)
			tempdata=cis.utils.create_masked_array_for_missing_data(aeronet_aot.data,aeronet_aot.metadata.missing_value)
			aeronet_aot.data=tempdata
 			aeronet_aot.save_data(aeronet_out_path+'/all/'+outputname)
		#if aeronet_ang !=None:
			#print aeronet_ang.data
		#	tempdata=cis.utils.create_masked_array_for_missing_data(aeronet_ang.data,aeronet_ang.metadata.missing_value)
		#	aeronet_ang.data=tempdata
 			#aeronet_ang.save_data('/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/all/'+outputname+'ang')

		# write aeronet subset out for comparisons

		#if not os.path.isfile(output+outputname):
		if aeronet_aot !=None:		

			#if not os.path.isfile(output+outputname):
			# make sure subset exists
				# collocate
			print 'processing:', aeronetdata
			print 'to:',output+'/all/',' as: '+outputname

			if not os.path.isdir(output+'/all/'):
				os.mkdir(output+'/all')
			collocated=model.collocated_onto(aeronet_aot)
			# save collocated data (could be also saved into dataholder...) 
			# although file preserves speeds up future use
			collocated.save_data(output+'/all/'+outputname)

			periods=[1,365]
			for period in periods:
				if period==1:
					pname='daily'
				elif period==365:
					pname='yearly'
				elif period==30:
					pname='monthly'
				print 'aggregating: ', outputname, ' with period ',period#, ' to: ',outputname
				if not os.path.isdir(output+'/'+pname):
					os.mkdir(output+'/'+pname)

				#print 'aggregating: ', input_data, ' with period ',period
				print 'to: ',output+'/'+pname+'/'
				print 'station:',outputname
				#subprocess.call("cis aggregate "+AODmodel+":"+input_TM5_data+" t=[2010-01-01,2010-12-31,"+period+"] -o "+output+outputname,shell=True)
				#test=subprocess.call(command,shell=True)
				#data=cis.read_data(input_data,AODvariable)
				#print data
				aggregated_data=collocated.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
				aggregated_data.save_data(output+'/'+pname+'/'+outputname)
				if not os.path.isdir(aeronet_out_path+'/'+pname):
					os.mkdir(aeronet_out_path+'/'+pname)
				print os.path.exists(aeronet_out_path+'/'+pname+'/')

				if os.path.exists(aeronet_out_path+'/'+pname+'/'):
					# aggregate aeronetdata as well
					aggregated_data=aeronet_aot.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
					aggregated_data.save_data(aeronet_out_path+'/'+pname+'/'+outputname)
		else:
			print 'no aeronet data for year: '+str(year)


		#else:
		#	print 'done previously: '+output+outputname

def do_aggregate(AODvariable = 'od550aer',input_data="" ,output="",period=1):
	from datetime import timedelta
	print input_data
	for input_data in glob.iglob(input_data+'/*.nc'):
		
		outputname=os.path.basename(input_data)
		
		if not os.path.isfile(output+"/"+outputname):
			print 'aggregating: ', input_data, ' with period ',period, ' to: ',outputname
			#print 'aggregating: ', input_data, ' with period ',period
			print 'to: ',output
			print 'station:',outputname
			#subprocess.call("cis aggregate "+AODmodel+":"+input_TM5_data+" t=[2010-01-01,2010-12-31,"+period+"] -o "+output+outputname,shell=True)
			#test=subprocess.call(command,shell=True)
			data=cis.read_data(input_data,AODvariable)
			#print data
			#print period
			aggregated_data=data.aggregate(t=[cis.time_util.PartialDateTime(2010),timedelta(days=period)] )
			aggregated_data.save_data(output+outputname)
def kk_agre(indata):
	jindex=0
	kkdata_sum=np.zeros((12))
	kkdata_std=np.zeros((12))
	kkdata2_sum=np.zeros((12,1000000))
	kkdata2_sum[:]=np.nan
	kk_numsites=np.zeros((12))
	kk_numsites2=np.zeros((12))
	for i in indata:
		kkdata=np.zeros((12))
		for kk in range(12):
			data=[]
			for jj,itime in enumerate(indata[i][8][:]):
				#print type(itime),itime
				if itime.month==kk+1 and not np.ma.is_masked(indata[i][4][jj]):
					data.append(indata[i][4][jj])
					kkdata2_sum[kk,jindex]=indata[i][4][jj]
					jindex+=1
				#[data.append(d) for d in outputdatadict['daily'][i][4][:] if kk+1 == itime.month]
				else:
					continue
				#print data
			kkdata[kk]=np.mean(np.array(data))
			if not np.isnan(kkdata[kk]):
				kkdata_sum[kk]+=kkdata[kk]
				kk_numsites[kk]+=1
		#plt.figure()
		#plt.plot(kkdata_sum/kk_numsites)
		#plt.title(i)
	mean_kkdata=kkdata_sum/kk_numsites
	for i in range(12):
		#print np.nanstd(kkdata2_sum[i,:])
		#print kkdata2_sum[i,:]
		kkdata_std[i]=np.nanstd(kkdata2_sum[i,:])
	return mean_kkdata,kk_numsites,kkdata_std

#if __name__="__main__":
AODmodel = 'od550aer'
aero_wavelength=550
AODdata = ['AOT_500']
year=2010
aeronet_out_path='/Users/bergmant/Documents/obs-data/aeronet/'+str(year)+'/AOT_'+str(aero_wavelength)+'/'
#input_TM5_data = '/Users/bergmant/Documents/Project/tm5-SOA/output/general_TM5_soa-riccobono_2010.od550aer.nc'
#input_TM5_data = '/Volumes/NTFS/general_TM5_soa-riccobono_2010.lev0.od550aer.nc'
#output= 'AERONET/col_aeronet_soa-riccobono-5nm-2010/'
aeronet='/Users/bergmant/Documents/obs-data/aeronet/aeronet_2010-550_all/'
#if not os.path.isdir(output):
#	print "creating output directory: "+ output
#	os.mkdir(output)       
EXPS=['newsoa-ri','oldsoa-final','nosoa']
EXPS=['newsoa-ri','oldsoa-bhn']#,'nosoa']
EXPS={}
EXPS['newsoa-ri']=None
EXPS['oldsoa-bhn']=None
l_aggregate=False
#l_aggregate=True
l_collocate=False
#l_collocate=True
for i in EXPS:
	input_TM5_data = '/Users/bergmant/Documents/tm5-soa/output/raw/general_TM5_'+i+'_2010.lev0.od550aer.nc'
	output_col= '/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_'+i+'-2010-550/'

	EXPS[i]={'inputdata':input_TM5_data,'collocated':output_col}
	if l_collocate:
		print 'collocating', i
		if not os.path.isdir(output_col+'/all/'):
			print "creating output directory: "+ output_col + '/all/'
			if not os.path.isdir(output_col):
				os.mkdir(output_col)
			os.mkdir(output_col+'/all')

		do_colocate(AODmodel,AODdata,input_TM5_data,output_col,aeronet)
	if l_aggregate:
		output_aggre=output_col+'/daily/'
		if not os.path.isdir(output_aggre):
			print "creating output directory: "+ output_aggre
			os.mkdir(output_aggre)
		do_aggregate('od550aer',output_col+'/all/',output_aggre,1)
	
		# output_aggre=output_col+'/monthly'
		# if not os.path.isdir(output_aggre):
		# 	print "creating output directory: "+ output_aggre
		# 	os.mkdir(output_aggre)
		# do_aggregate('od550aer',output_col+'/all/',output_aggre,'P1M')
		
		output_aggre=output_col+'/yearly/'
		if not os.path.isdir(output_aggre):
			print "creating output directory: "+ output_aggre
			os.mkdir(output_aggre)
		do_aggregate('od550aer',output_col+'/all/',output_aggre,365)
		#subprocess.call("cis col "+AODmodel+":"+input_TM5_data+" "+outputdata+':variable='+AODdata+',collocator=lin -o '+output+outputname,shell=True)
		   #aggregate AOT_500:${i} t=[2010-01-01,2010-12-31,P1M]  -o aeronet_2010_monthly/$(basename $i .nc).monthly.nc
	output_aggre_aeronet='/Users/bergmant/Documents/obs-data/aeronet/2010/daily/'
	input_aeronet_all='/Users/bergmant/Documents/obs-data/aeronet/2010/all/'
	if not os.path.isdir(output_aggre_aeronet):
		print "creating output directory: "+ output_aggre_aeronet
		os.mkdir(output_aggre_aeronet)
	print 'aggregating aeronet'
	do_aggregate('AOT_500',input_aeronet_all,output_aggre_aeronet,1)

	# output_aggre_aeronet='/Users/bergmant/Documents/obs-data/aeronet/2010/yearly/'
	# input_aeronet_all='/Users/bergmant/Documents/obs-data/aeronet/2010/all/'
	# if not os.path.isdir(output_aggre_aeronet):
	# 	print "creating output directory: "+ output_aggre_aeronet
	# 	os.mkdir(output_aggre_aeronet)
	# do_aggregate('AOT_500',input_aeronet_all,output_aggre_aeronet,365)
	print ("collocation and aggregation done!")


#output_png_path='{}/Documents/tm5-soa/figures/png/AERONET/'.format(os.environ['HOME'])

#indata='/Users/bergmant/Documents/Project/ifs+tm5-validation/TM5-AERONET/Collocated_aeronet_ctrl2016_lin/'
#indata=None
#'/Users/bergmant/Documents/Project/tm5-SOA/SOA2010_daily'
#outputdata,TM5data=read_aeronet_all(indata)
outputdataym=read_aggregated(aeronet_out_path+'/yearly/')
outputdatadm=read_aggregated(aeronet_out_path+'/daily/')
print aeronet_out_path+'/yearly/',aeronet_out_path+'/daily/'
#outputdatamm=read_aggregated('/Users/bergmant/Documents/obs-data/aeronet/aeronet_2010_monthly/')
#=read_aggregated('/Users/bergmant/Documents/obs-data/aeronet/2010/all/')
print EXPS['newsoa-ri']['collocated']+'/yearly/'
outputdatadict={}
TM5NEWdatadict={}
TM5OLDdatadict={}
dataperiods=['yearly','monthly','daily','all']
for period in dataperiods:
	outputdatadict[period]=read_aggregated(aeronet_out_path+'/'+period+'/')
	TM5NEWdatadict[period]=read_aggregated(EXPS['newsoa-ri']['collocated']+'/'+period+'/')
	TM5OLDdatadict[period]=read_aggregated(EXPS['oldsoa-bhn']['collocated']+'/'+period+'/')
tm5new=group_by_region(TM5NEWdatadict['yearly'])
tm5old=group_by_region(TM5OLDdatadict['yearly'])
#print outputdatadict['yearly']
aeronet=group_by_region(outputdatadict['yearly'])

print len(tm5old)
print len(tm5old['SA'])
print type(tm5old)
print len(aeronet)
print len(aeronet['SA'])
print type(aeronet)
aeronet_table=open(paper+"aeronet_table.txt","w")
aeronet_table_latex=open(paper+"aeronet_table.tex","w")
# f_old,ax_old=plt.subplots(1)
# f_new,ax_new=plt.subplots(1)
#f_comb,ax_comb=plt.subplots(1)
f_scat_comb,ax_scat_comb=plt.subplots(nrows=2,ncols=2,figsize=(16,16))#,tight_layout=True)
colordict={'NA':'r','EU':'b','SA':'k','SIB':'g','SEAS':'y','AUS':'m','AFR':'brown','SAH':'c'}
m = Basemap(projection='cyl',lon_0=0,ax=ax_scat_comb[1,1])
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,3060.,60.))
regions=get_regions()
for index,reg in enumerate(colordict):
	aero_concat=[]#np.empty(len(aeronet[reg]))
	#print (len(aeronet[reg]))
	tm5old_concat=[]#np.empty(len(tm5old[reg]))
	tm5new_concat=[]#np.empty(len(tm5new[reg]))
	#raw_input()
	print '----reg---'
	print reg
	print len(aeronet)
	print len(aeronet[reg])
	if len(aeronet[reg])==0:
			continue
	#print aeronet[reg][0]
	lonmin=min(regions[reg][0][1],regions[reg][1][1])
	lonmax=max(regions[reg][0][1],regions[reg][1][1])
	latmin=min(regions[reg][0][0],regions[reg][1][0])
	latmax=max(regions[reg][0][0],regions[reg][1][0])

	plot_rectangle(m,lonmin,lonmax,latmin,latmax,color=colordict[reg])
	x, y = m(lonmin,latmin)
	ax_scat_comb[1,1].annotate(reg, xy=(x, y), xycoords='data', xytext=(x, y), textcoords='data',color='w')
	for i in aeronet[reg]:
		#print aeronet
		#print i
		#aero_concat[i]=aeronet[reg][i][4]
		#tm5old_concat[i]=tm5old[reg][i][4]
		#tm5new_concat[i]=tm5new[reg][i][4]
		#print i
		#print aeronet[reg][i][4]
		#print  not np.ma.is_masked(aeronet[reg][i][4])
		#print tm5old[reg][i]
		#raw_input()
		#print i,aeronet[reg],aeronet[reg][i]
		#print i,tm5old[reg],tm5old[reg][i]
		#print i,tm5new[reg],tm5new[reg][i]
		if not np.ma.is_masked(aeronet[reg][i][4]):
			aero_concat.append(aeronet[reg][i][4])
			tm5old_concat.append(tm5old[reg][i][4])
			tm5new_concat.append(tm5new[reg][i][4])
		else:
			print aeronet[reg][i][4],tm5old[reg][i][4],tm5new[reg][i][4]
			continue
		if np.isnan(tm5new[reg][i][4]):
			print tm5new[reg][i][5],aeronet[reg][i][5]
			print tm5new[reg][i][4],aeronet[reg][i][4]
			
	aero_concat=np.array(aero_concat)
	#print aero_concat
	#print tm5new_concat
	#print tm5old_concat

	tm5old_concat=np.array(tm5old_concat)
	tm5new_concat=np.array(tm5new_concat)
	#print aero_concat
	#print tm5old_concat
	print reg,np.shape(tm5new_concat), np.mean(aero_concat),np.mean(tm5new_concat),np.mean(tm5old_concat)
	print reg,np.shape(tm5new_concat), np.shape(aero_concat)
	#print aero_concat,tm5new_concat
	#print aero_concat,tm5new_concat
	#print reg,np.shape(tm5new_concat), NMB(aero_concat,tm5new_concat),NMB(aero_concat,tm5old_concat)
	#print reg,np.shape(tm5new_concat), RMSE(aero_concat,tm5new_concat),RMSE(aero_concat,tm5old_concat)
	#plt.scatter(aero_concat,tm5new_concat,c=colordict[reg],s=1.5)
	if len(aero_concat)==0:
		continue
	print reg
	# slope,interp,r,p,stderr=scatter_dot(aero_concat,tm5old_concat,col=colordict[reg],ax=ax_old,modelname='OLDSOA',label=reg,ms=10)
	# ax_old.annotate(reg+' N: %5.2f MB: %5.2f, R: %5.2f, stderr: %5.2f'%(len(tm5old_concat),np.ma.mean(np.array(tm5old_concat)-np.array(aero_concat)),r,stderr),xy=(0.02,0.9-index*0.05),xycoords='axes fraction',fontsize=10)
	# handles, labels = ax_old.get_legend_handles_labels()
	# print handles,labels
	#plt.scatter(aero_concat,tm5_concat,c=colordict[reg],s=1.5)
	# slope,interp,r,p,stderr=scatter_dot(aero_concat,tm5new_concat,col=colordict[reg],ax=ax_new,modelname='NEWSOA',label=reg,ms=10)
	# ax_new.annotate(reg+' N: %5.2f MB: %5.2f, R: %5.2f, stderr: %5.2f'%(len(tm5new_concat),np.ma.mean(np.array(tm5new_concat)-np.array(aero_concat)),r,stderr),xy=(0.02,0.9-index*0.05),xycoords='axes fraction',fontsize=10)
	NMB_new=NMB(aero_concat,tm5new_concat)*100
	NMB_old=NMB(aero_concat,tm5old_concat)*100
	slope,interp,rnew,p,stderrnew=scatter_dot(aero_concat,tm5new_concat,col=colordict[reg],ax=ax_scat_comb[0,0],modelname='NEWSOA',label=reg,ms=10)
	#ax_scat_comb[0].annotate('%-6.6s N: %2d MB: %5.2f, R: %5.2f, SE: %5.2f'%(reg,len(tm5old_concat),np.ma.mean(np.array(tm5new_concat)-np.array(aero_concat)),r,stderr),xy=(0.02,0.95-index*0.05),xycoords='axes fraction',fontsize=10)

	slope,interp,rold,p,stderrold=scatter_dot(aero_concat,tm5old_concat,col=colordict[reg],ax=ax_scat_comb[0,1],modelname='OLDSOA',label=reg,ms=10)
	#ax_scat_comb[1].annotate('%-6.6s N: %2d MB: %5.2f, R: %5.2f, SE: %5.2f'%(reg,len(tm5old_concat),np.ma.mean(np.array(tm5old_concat)-np.array(aero_concat)),r,stderr),xy=(0.02,0.95-index*0.05),xycoords='axes fraction',fontsize=10)
	rmse_old=RMSE(aero_concat,tm5old_concat)
	rmse_new=RMSE(aero_concat,tm5new_concat)


	aeronet_table.write('%-6.6s\t%2d\t%6.3f\t%6.3f\t%6.3f\t%5.1f\t%5.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\\\\ \n'%(reg,len(tm5old_concat),np.ma.mean(np.array(aero_concat)),np.ma.mean(np.array(tm5new_concat)),np.ma.mean(np.array(tm5old_concat)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
	aeronet_table_latex.write('%-6.6s&%2d&%6.3f&%6.3f&%6.3f&%5.1f&%5.1f&%6.3f&%6.3f&%6.3f&%6.3f\\\\ \n'%(reg,len(tm5old_concat),np.ma.mean(np.array(aero_concat) ),np.ma.mean(np.array(tm5new_concat)),np.ma.mean(np.array(tm5old_concat)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
	
for i in outputdatadict['yearly']:
	#print outputdatadict['yearly'][i][0],outputdatadict['yearly'][i][1]
	px,py=m(outputdatadict['yearly'][i][0],outputdatadict['yearly'][i][1])
	#px,py=m(10,10)
	print px,py
	m.scatter(px,py,5,marker='o',color='w',edgecolor='k',linewidth=0.5,zorder=100)
nkk,kkn,astd=kk_agre(outputdatadict['all'])
nkk_new,kkn_old,nstd=kk_agre(TM5NEWdatadict['all'])
nkk_old,kkn_new,ostd=kk_agre(TM5OLDdatadict['all'])
ax_scat_comb[1,0].plot(np.linspace(0,11,12),nkk,'k',label='AERONET')
ax_scat_comb[1,0].plot(np.linspace(0,11,12),nkk_new,'r',label=EXP_NAMEs[0])
ax_scat_comb[1,0].plot(np.linspace(0,11,12),nkk_old,'b',label=EXP_NAMEs[1])
ax_scat_comb[1,0].set_xticks(np.linspace(0,11,12))
ax_scat_comb[1,0].set_xticklabels(str_months())
ax_scat_comb[1,0].set_xlabel('Month')
ax_scat_comb[1,0].set_ylabel('AOD [1]')

ax_scat_comb[0,0].legend(loc=4)
ax_scat_comb[1,0].legend(loc=3)
ax_scat_comb[0,1].legend(loc=4)
print (outputdatadict['all'].keys())
aero_R=[]
TM5n_R=[]
TM5o_R=[]
for site in outputdatadict['yearly']:
	#print outputdatadict['yearly'][site][4]
	#print type(outputdatadict['yearly'][site][4])
	#print outputdatadict['yearly'][site][4]==None
	#print np.isnan(outputdatadict['yearly'][site][4])
	if not np.ma.is_masked(outputdatadict['yearly'][site][4]):			
		aero_R.append(outputdatadict['yearly'][site][4])
		TM5n_R.append(TM5NEWdatadict['yearly'][site][4])
		TM5o_R.append(TM5OLDdatadict['yearly'][site][4])
print aero_R
print TM5n_R
print TM5o_R

ax_scat_comb[0,0].annotate('R (log R): %5.2f (%5.2f)'%(stats.pearsonr(np.array(TM5n_R),np.array(aero_R))[0],stats.pearsonr(np.log(np.array(TM5n_R)),np.log(np.array(aero_R)))[0]),xy=(0.05,0.95),xycoords='axes fraction',fontsize=12)
ax_scat_comb[0,0].annotate('MB : %5.3f '%(np.mean(np.array(TM5n_R))-np.mean(np.array(aero_R))),xy=(0.05,0.90),xycoords='axes fraction',fontsize=12)
ax_scat_comb[0,1].annotate('R (log R): %5.2f (%5.2f)'%(stats.pearsonr(np.array(TM5o_R),np.array(aero_R))[0],stats.pearsonr(np.log(np.array(TM5o_R)),np.log(np.array(aero_R)))[0]),xy=(0.05,0.95),xycoords='axes fraction',fontsize=12)
ax_scat_comb[0,1].annotate('MB : %5.3f'%(np.mean(np.array(TM5o_R))-np.mean(np.array(aero_R))),xy=(0.05,0.90),xycoords='axes fraction',fontsize=12)
k=0
for i in range(2):
	for j in range(2):
		if k==3:
			ax_scat_comb[i,j].annotate(string.ascii_lowercase[k]+')',xy=(0.01,1.51),xycoords='axes fraction',fontsize=18)
		else:
			ax_scat_comb[i,j].annotate(string.ascii_lowercase[k]+')',xy=(0.01,1.01),xycoords='axes fraction',fontsize=18)
		k+=1
f_scat_comb.savefig(output_png_path+'AERONET/scatter_categorized_log_seasonal_map.png',dpi=600)
f_scat_comb.savefig(output_pdf_path+'AERONET/scatter_categorized_log_seasonal_map.pdf')
# f_old.legend()
# f_new.legend()
#TM5dataym=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_newsoa-ri-2010/yearly/')
#TM5datadm=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_newsoa-ri-2010/daily/')
#TM5datamm=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_newsoa-ri-2010/monthly/')
#TM5data=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_newsoa-ri-2010/all/')
#TM5OLDdatadict={}
#for period in dataperiods:
#	TM5OLDdatadict[period]=read_aggregated(EXPS['oldsoa-bhn']['collocated']+'/'+period+'/')

#TM5_OLD_dataym=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_oldsoa-bhn-2010/yearly/')
#TM5_OLD_datadm=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_oldsoa-bhn-2010/daily/')
#TM5_OLD_datamm=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_oldsoa-bhn-2010/monthly/')
#TM5_OLD_data=read_aggregated('/Users/bergmant/Documents/tm5-soa/output/processed/col_aeronet_oldsoa-bhn-2010/all/')
#print len(TM5datamm),len(outputdatamm)
#sites2numpy(TM5dataym)
#TM5n,TM5o,aero=concatenate_sites(TM5datadm,TM5_OLD_datadm,outputdatadm)
#TM5n2=concatenate_sites2(TM5datadm)
#TM5o2=concatenate_sites2(TM5_OLD_datadm)
#aero2=concatenate_sites2(outputdatadm)
TM5n2=concatenate_sites2(TM5NEWdatadict['daily'])
TM5o2=concatenate_sites2(TM5OLDdatadict['daily'])
aero2=concatenate_sites2(outputdatadict['daily'])

print np.shape(TM5n2),np.shape(TM5o2),np.shape(aero2)
#print np.shape(TM5n),np.shape(TM5o),np.shape(aero)

#scatter_heat(TM5o,TM5n)
# plt.xlim(0,2.0)
# plt.ylim(0,2.0)
# plt.title('both')
#scatter_heat(aero,TM5n)
#TM5n,TM5o,aero=concatenate_sites(TM5datamm,TM5_OLD_datamm,outputdatamm)
#scatter_heat(TM5o,TM5n)
# plt.title('NEWSOA')
scatter_heat(aero2,TM5n2)
#TM5n,TM5o,aero=concatenate_sites(TM5datamm,TM5_OLD_datamm,outputdatamm)
#scatter_heat(TM5o,TM5n)
#plt.title('NEWSOA')
#scatter_heat(aero2,TM5o2)
#TM5n,TM5o,aero=concatenate_sites(TM5datamm,TM5_OLD_datamm,outputdatamm)
scatter_heat(TM5o2,TM5n2)
plt.title('OLDSOAvsNEWSOA')
plt.xlim(0,2.0)
plt.ylim(0,2.0)
#plt.figure()
#plt.scatter(aero2,TM5n2,s=1)
#plt.title('NEWSOA oldway plotting')
#plt.xlim(0,1.)
#plt.ylim(0,1.)
plt.figure()
plt.scatter(aero2,TM5o2,s=1)
plt.xlim(0,1.)
plt.ylim(0,1.)
plt.title('OLDSOA')
plt.figure()
plt.scatter(aero2,TM5n2,s=1)
plt.xlim(0,1.75)
plt.ylim(0,1.75)
#TM5n,TM5o,aero=concatenate_sites(TM5dataym,TM5_OLD_dataym,outputdataym)
# aero=[]
# TM5n=[]
# TM5o=[]
# for i in TM5dataym:
# 	print outputdataym[i][4]
# 	aa=outputdataym[i][4]
# 	print aa
# 	print np.ma.getdata(outputdataym[i][4])
# 	print 
# 	#aero.append(outputdataym[i][4])
# 	#print outputdataym[i][4][mask]
# 	#aero.append(float(aa))
# 	#TM5o.append(float(TM5_OLD_dataym[i][4]))
# 	#TM5n.append(float(TM5dataym[i][4]))
# 	print np.ma.is_masked(aa)
# 	if not np.ma.is_masked(aa):
# 		aero.append(aa)
# 		TM5o.append(TM5_OLD_dataym[i][4])
# 		TM5n.append(TM5dataym[i][4])

aero=[]
TM5n=[]
TM5o=[]
for i in TM5NEWdatadict['yearly']:
	#aa=outputdataym[i][4]
	aa=outputdatadict['yearly'][i][4]
	#print aa
	#print np.ma.getdata(outputdatadict['yearly'][i][4])
	print 
	#aero.append(outputdataym[i][4])
	#print outputdataym[i][4][mask]
	#aero.append(float(aa))
	#TM5o.append(float(TM5_OLD_dataym[i][4]))
	#TM5n.append(float(TM5dataym[i][4]))
	print np.ma.is_masked(aa)
	print np.isnan(aa)
	if np.isnan(aa) or np.isnan(TM5NEWdatadict['yearly'][i][4]):
		raw_input('nnannnanann')
	if not np.ma.is_masked(aa):
		aero.append(aa)
		TM5o.append(TM5OLDdatadict['yearly'][i][4])
		TM5n.append(TM5NEWdatadict['yearly'][i][4])
	else:
		print 'problem'
		#print aa,TM5OLDdatadict['yearly'][i][4],TM5NEWdatadict['yearly'][i][4]
		#print outputdataym[i][5],TM5OLDdatadict['yearly'][i][5],TM5NEWdatadict['yearly'][i][5]
		
print len(outputdataym)
#print aero
#print TM5n
plt.figure()
f_scat_2panels,a=plt.subplots(nrows=1,ncols=2,figsize=(12,6))
a[0].plot([0,1],[0,1],'-k')
a[0].scatter(aero,TM5o,s=1.5,c='b')
a[0].annotate('NEWSOA MB: %5.2f'%(np.ma.mean(np.array(TM5n)-np.array(aero))),xy=(0.05,0.9),xycoords='axes fraction',fontsize=12)
print stats.pearsonr(np.array(TM5n),np.array(aero)),np.shape(TM5n)
print stats.pearsonr(np.log(np.array(TM5n)),np.log(np.array(aero))),np.shape(TM5n)
a[0].annotate('NEWSOA  R: %5.2f (%5.2f)'%(stats.pearsonr(np.array(TM5n),np.array(aero))[0], stats.pearsonr(np.log(np.array(TM5n)),np.log(np.array(aero)))[0]),xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)
a[0].set_xlim(0,1.0)
a[0].set_ylim(0,1.0)

a[0].set_ylabel('NEWSOA',fontsize=14)
a[0].set_xlabel('AERONET',fontsize=14)

a[1].plot([0,1],[0,1],'-k')
a[1].scatter(aero,TM5n,s=1.5,c='r')
a[1].annotate('OLDSOA MB: %5.2f'%(np.ma.mean(np.array(TM5o)-np.array(aero))),xy=(0.05,0.9),xycoords='axes fraction',fontsize=12)
print stats.pearsonr(np.array(TM5o),np.array(aero)),np.shape(TM5o)
print stats.pearsonr(np.log(np.array(TM5o)),np.log(np.array(aero))),np.shape(TM5o)
a[1].annotate('OLDSOA  R: %5.2f (%5.2f)'%(stats.pearsonr(np.array(TM5o),np.array(aero))[0],stats.pearsonr(np.log(np.array(TM5o)),np.log(np.array(aero)))[0]),xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)
a[1].set_xlim(0,1.0)
a[1].set_ylim(0,1.0)
a[1].set_ylabel('OLDSOA',fontsize=14)
a[1].set_xlabel('AERONET',fontsize=14)
f_scat_2panels.savefig(output_png_path+'AERONET/scatter_linear_2panel.png',dpi=200)


f_scat_log_2panels,a=plt.subplots(nrows=1,ncols=2,figsize=(12,6))
a[0].plot([1e-2,1],[1e-2,1],'-k')
a[0].scatter(aero,TM5o,s=1.5,c='b')
a[0].annotate('NEWSOA MB: %5.2f'%(np.ma.mean(np.array(TM5n)-np.array(aero))),xy=(0.05,0.9),xycoords='axes fraction',fontsize=12)
print stats.pearsonr(np.array(TM5n),np.array(aero)),np.shape(TM5n)
print stats.pearsonr(np.array(TM5n),np.array(aero)),np.shape(TM5n)
a[0].annotate('NEWSOA  R: %5.2f'%(stats.pearsonr(np.array(TM5n),np.array(aero))[0]),xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)
a[0].set_xlim(1e-2,1.0)
a[0].set_ylim(1e-2,1.0)
a[0].set_xscale('log')
a[0].set_yscale('log')
a[0].set_ylabel('NEWSOA',fontsize=14)
a[0].set_xlabel('AERONET',fontsize=14)
a[1].plot([1e-2,1],[1e-2,1],'-k')
a[1].scatter(aero,TM5n,s=1.5,c='r')
a[1].annotate('OLDSOA MB: %5.2f'%(np.ma.mean(np.array(TM5o)-np.array(aero))),xy=(0.05,0.9),xycoords='axes fraction',fontsize=12)
a[1].annotate('OLDSOA  R: %5.2f'%(stats.pearsonr(np.array(TM5o),np.array(aero))[0]),xy=(0.05,0.85),xycoords='axes fraction',fontsize=12)
a[1].set_xlim(1e-2,1.0)
a[1].set_ylim(1e-2,1.0)
a[1].set_ylabel('OLDSOA',fontsize=14)
a[1].set_xlabel('AERONET',fontsize=14)
a[1].set_xscale('log')
a[1].set_yscale('log')
f_scat_log_2panels.savefig(output_png_path+'AERONET/scatter_log_2panel.png',dpi=200)

modelu,modelg,modelo,obsu,obsg,obso=categorize(outputdatadict['yearly'],TM5NEWdatadict['yearly'])
print 'OLDSOA MB: %6.3f'%(np.ma.mean(np.array(TM5o)-np.array(aero)))
print 'OLDSOA NMB: %6.3f'%(NMB(TM5o,aero))
print 'NEWSOA MB: %6.3f'%(np.ma.mean(np.array(TM5n)-np.array(aero)))
print 'NEWSOA NMB: %6.3f'%(NMB(TM5n,aero))
TM5n=np.array(TM5n)
TM5o=np.array(TM5o)
aero=np.array(aero)
NMB_new=NMB(TM5n,aero)
NMB_old=NMB(TM5o,aero)
mask = ~np.isnan(aero) & ~np.isnan(TM5o)
print  TM5n,type(TM5n),np.shape(TM5n),mask
slope,intercept,rold,p,stderr=stats.linregress(aero[mask], TM5o[mask])	
slope,intercept,rnew,p,stderr=stats.linregress(aero[mask], TM5n[mask])	
rmse_old=RMSE(aero,TM5o)
rmse_new=RMSE(aero,TM5n)
print rnew,rold
aeronet_table.write('%-6.6s\t%2d\t%6.3f\t%6.3f\t%6.3f\t%5.1f\t%5.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\\\\ \n'%('All',len(TM5o),np.ma.mean(np.array(aero)),np.ma.mean(np.array(TM5n)),np.ma.mean(np.array(TM5o)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))
aeronet_table_latex.write('%-6.6s&%2d&%6.3f&%6.3f&%6.3f&%5.1f&%5.1f&%6.3f&%6.3f&%6.3f&%6.3f\\\\ \n'%('All',len(TM5o),np.ma.mean(np.array(aero) ),np.ma.mean(np.array(TM5n)),np.ma.mean(np.array(TM5o)),NMB_new,NMB_old,rnew,rold,rmse_new,rmse_old))

aeronet_table.close()
aeronet_table_latex.close()

#map_bias(outputdatadm,TM5datadm)
#map_bias(outputdatadm,TM5_OLD_datadm)
f,axit=plt.subplots(1)
map_bias(outputdatadict['daily'],TM5NEWdatadict['daily'],ax=axit,boundmax=1.1,name1='AERONET',name2='NEWSOA')
f,axit=plt.subplots(1)
map_bias(outputdatadict['daily'],TM5OLDdatadict['daily'],ax=axit,boundmax=1.1,name1='AERONET',name2='OLDSOA')
f,axit=plt.subplots(1)
map_bias(TM5OLDdatadict['daily'],TM5NEWdatadict['daily'],ax=axit,boundmax=0.11,name1='OLDSOA',name2='NEWSOA')
f13,axit=plt.subplots(nrows=1,ncols=2,figsize=(20,6))
print np.shape(axit)

#axit[0]=
map_bias(TM5OLDdatadict['all'],TM5NEWdatadict['all'],ax=axit[0],boundmax=0.5,name1='OLDSOA',name2='NEWSOA',bounds=[-0.5,-0.35,-0.15,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.15,0.35,0.5],ref=outputdatadict['all'])
axit[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
#biasmap(TM5NEWdatadict['daily'],TM5OLDdatadict['daily'])
map_bias(outputdatadict['all'],TM5NEWdatadict['all'],ax=axit[1],boundmax=1.1,name1='AERONET',name2='NEWSOA')
#axit[0].annotate(('NMB (MB): %5.1f %% (%4.2f)')%(nmbmean*100,mbmean),xy=(0.45,0.06),xycoords='axes fraction',fontsize=10)
axit[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
#axit[1]=

f13.savefig(output_png_path+'/AERONET/map-2panel-aeronet-oldsoa.png',dpi=600)
f,axit=plt.subplots(1)
map_bias(outputdatadict['all'],TM5NEWdatadict['all'],ax=axit,boundmax=1.1,name1='AERONET',name2='NEWSOA')


jindex=0
#for n,exp in enumerate(EXPS):
# kkdata2=np.zeros((12))
# kknum2=np.zeros((12))
# for i in outputdatadict['daily']:
# 	kkdata=np.zeros((12))
# 	for kk in range(12):
# 		data=[]
# 		for jj,itime in enumerate(outputdatadict['daily'][i][8][:]):
# 			print type(itime),itime
# 			if itime.month==kk+1 and not np.ma.is_masked(outputdatadict['daily'][i][4][jj]):
# 				data.append(outputdatadict['daily'][i][4][jj])
# 			#[data.append(d) for d in outputdatadict['daily'][i][4][:] if kk+1 == itime.month]
# 			else:
# 				continue
# 			print data
# 		kkdata[kk]=np.mean(np.array(data))
# 		if not np.isnan(kkdata[kk]):
# 			kkdata2[kk]+=kkdata[kk]
# 			kknum2[kk]+=1
# 	print kkdata
# 	#plt.figure()
# 	#plt.plot(kkdata)
# 	#plt.show()
# newkk=kkdata2/kknum2	

print astd
#print newkk
print nkk

f,ax=plt.subplots(1)
ax.plot(np.linspace(0,11,12),nkk,'k')
ax.plot(np.linspace(0,11,12),nkk+astd,'--k')
ax.plot(np.linspace(0,11,12),nkk-astd,'--k')
ax.plot(np.linspace(0,11,12),nkk_new,'r')
ax.plot(np.linspace(0,11,12),nkk_new+nstd,'--r')
ax.plot(np.linspace(0,11,12),nkk_new-nstd,'--r')
ax.plot(np.linspace(0,11,12),nkk_old,'b')
ax.plot(np.linspace(0,11,12),nkk_old+ostd,'--b')
ax.plot(np.linspace(0,11,12),nkk_old-ostd,'--b')
ax.set_xticks(np.linspace(0,11,12))
ax.set_xticklabels(str_months())
ax.set_xlabel('Month')
ax.set_ylabel('AOD [1]')
f.savefig(output_png_path+'/AERONET/seasonal_all.png',dpi=600)
plt.show()
map_bias(obsu,modelu)
map_bias(obsg[i],modelg[i])



# for i in range(len(modelu)):
# 	plt.figure()
# 	plt.title(modelu[i][5].split('/')[-1])
# 	plt.plot(modelu[i][-1],modelu[i][4],'or')
# 	plt.plot(obsu[i][-1],obsu[i][4],'ob')
# plt.show()
# map_bias(obso,modelo)
# for i in range(len(modelo)):
# 	plt.figure()
# 	plt.title(modelo[i][5].split('/')[-1])
# 	plt.plot(modelo[i][4],'r')
# 	plt.plot(obso[i][4],'b')
# plt.show()
# map_bias(obsg,modelg)
# for i in range(len(modelg)):
# 	plt.figure()
# 	plt.title(modelg[i][5].split('/')[-1])
# 	plt.plot(modelg[i][4],'r')
# 	plt.plot(obsg[i][4],'b')
plt.show()

TM5data_old=TM5data

plt.figure(figsize=(10,7))

m = Basemap(projection='robin',lon_0=0)
m.drawcoastlines()
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,3060.,60.))
mycmap=plt.get_cmap('coolwarm',11) 
# define the bins and normalize
bounds = [-0.45,-0.35,-0.25,-0.15,-0.05,-0.02,0.02,0.05,0.15,0.25,0.35,0.45]
bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
norm = mpl.colors.BoundaryNorm(bounds, 11)
##
#intitialise
overestimateTM5=[]
overestimateAERONET=[]
underestimateTM5=[]
underestimateAERONET=[]
goodTM5=[]
goodAERONET=[]
#print outputdata[0][4]
annual_aeronet=[]
annual_TM5=[]
for i in outputdatadict['all']:
	#print i[0][0],i[1][0]
	print i
	x1,y1=m(outputdatadict['all'][i][0][0],outputdata[i][1][0])
	print np.shape(TM5NEWdatadict['all']),np.shape(outputdata)
	diff=np.mean(TM5NEWdatadict['all'][i][4][:])-np.mean(outputdatadict['all'][i][4][:])
	#print diff
	print outputdatadict['all'][i][5]
	if diff >0.1:
		overestimateAERONET.append(outputdatadict['all'][i])
		overestimateTM5.append(TM5NEWdatadict['all'][i])
	if diff >-0.025 and diff <0.025:
		goodAERONET.append(outputdatadict['all'][i])
		goodTM5.append(TM5NEWdatadict['all'][i])
	if diff <-0.1:
		underestimateAERONET.append(outputdatadict['all'][i])
		underestimateTM5.append(TM5NEWdatadict['all'][i])
	#print TM5data[i][4][:],np.shape(TM5data[i][4][:]),TM5data[i][5]
	#for j in TM5data[i][4]:
	#	print type(j)
	print 'no. datapoints with data: ',np.ma.count(TM5NEWdatadict['all'][i][4][:])
	days_data=np.ma.count(TM5NEWdatadict['all'][i][4][:])
	if np.ma.min(TM5NEWdatadict['all'][i][4]) is np.ma.masked:
		continue
	#print i,diff,TM5data[i][4][:]
	annual_TM5.append(np.mean(TM5NEWdatadict['all'][i][4][:]))
	annual_aeronet.append(np.mean(outputdatadict['all'][i][4][:]))
	#print outputdata[i][4]
	m.scatter(x1,y1,marker='o',c=diff,s=300*(np.mean(outputdatadict['all'][i][4][:])),norm=norm,cmap = mycmap )
	xx,yy=m(-160,-70)
	m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
	xx,yy=m(-150,-70)
	plt.text(xx,yy,'AOD 1.0')
	xx,yy=m(-160,-60)
	m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
	xx,yy=m(-150,-60)
	plt.text(xx,yy,'AOD 0.5')

c = plt.colorbar(orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
c.ax.tick_params(labelsize=10)
c.set_label('AOD bias [TM5-AERONET]')

i='newsoa-ri'
#indata='/Volumes/Utrecht/'+i+'/general_TM5_'+i+'_2010.lev0.od550aer.nc'
indata='/Users/bergmant/Documents/tm5-soa/output/raw/general_TM5_'+i+'_2010.lev0.od550aer.nc'
print indata
outputdata,TM5data=read_aeronet_all(indata)


# plt.figure(figsize=(10,7))
# TM5data_new=TM5data
# m = Basemap(projection='robin',lon_0=0)
# m.drawcoastlines()
# m.drawparallels(np.arange(-90.,120.,30.))
# m.drawmeridians(np.arange(0.,3060.,60.))
# mycmap=plt.get_cmap('coolwarm',11) 
# # define the bins and normalize
# bounds = [-0.45,-0.35,-0.25,-0.15,-0.05,-0.02,0.02,0.05,0.15,0.25,0.35,0.45]
# bounds = [-0.375,-0.3,-0.225,-0.15,-0.075,-0.025,0.025,0.075,0.15,0.225,0.30,0.375]
# norm = mpl.colors.BoundaryNorm(bounds, 11)
# ##
# #intitialise
# overestimateTM5=[]
# overestimateAERONET=[]
# underestimateTM5=[]
# underestimateAERONET=[]
# goodTM5=[]
# goodAERONET=[]
# #print outputdata[0][4]
# annual_aeronet=[]
# annual_TM5=[]
# for i in range(len(outputdata)):
# 	#print i[0][0],i[1][0]
# 	print i
# 	x1,y1=m(outputdata[i][0][0],outputdata[i][1][0])
# 	print np.shape(TM5data),np.shape(outputdata)
# 	diff=np.mean(TM5data[i][4][:])-np.mean(outputdata[i][4][:])
# 	#print diff
# 	print outputdata[i][5]
# 	if diff >0.1:
# 		overestimateAERONET.append(outputdata[i])
# 		overestimateTM5.append(TM5data[i])
# 	if diff >-0.025 and diff <0.025:
# 		goodAERONET.append(outputdata[i])
# 		goodTM5.append(TM5data[i])
# 	if diff <-0.1:
# 		underestimateAERONET.append(outputdata[i])
# 		underestimateTM5.append(TM5data[i])
# 	#print TM5data[i][4][:],np.shape(TM5data[i][4][:]),TM5data[i][5]
# 	#for j in TM5data[i][4]:
# 	#	print type(j)
# 	print 'no. datapoints with data: ',np.ma.count(TM5data[i][4][:])
# 	days_data=np.ma.count(TM5data[i][4][:])
# 	if np.ma.min(TM5data[i][4]) is np.ma.masked:
# 		continue
# 	#print i,diff,TM5data[i][4][:]
# 	annual_TM5.append(np.mean(TM5data[i][4][:]))
# 	annual_aeronet.append(np.mean(outputdata[i][4][:]))
# 	#print outputdata[i][4]
# 	m.scatter(x1,y1,marker='o',c=diff,s=300*(np.mean(outputdata[i][4][:])),norm=norm,cmap = mycmap )
# 	xx,yy=m(-160,-70)
# 	m.scatter(xx,yy,marker='o',c=0,s=300,norm=norm,cmap = mycmap)
# 	xx,yy=m(-150,-70)
# 	plt.text(xx,yy,'AOD 1.0')
# 	xx,yy=m(-160,-60)
# 	m.scatter(xx,yy,marker='o',c=0,s=150,norm=norm,cmap = mycmap)
# 	xx,yy=m(-150,-60)
# 	plt.text(xx,yy,'AOD 0.5')

# c = plt.colorbar(orientation='horizontal',ticks=bounds,aspect=30,pad=0.05,shrink=0.7)
# c.ax.tick_params(labelsize=10)
# c.set_label('AOD bias [TM5-AERONET]')

plt.figure()
for i in TM5NEWdatadict['all']:
	plt.scatter(TM5OLDdatadict['all'][i][4][:],TM5NEWdatadict['all'][i][4][:]-TM5OLDdatadict['all'][i][4][:])

#scatter_heat()
plt.show()

