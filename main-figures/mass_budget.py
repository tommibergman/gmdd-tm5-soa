import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from pylab import *
from settings import *
from general_toolbox import get_gridboxarea,lonlat
import matplotlib.gridspec as gridspec
import string
days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])

def read_data(filein,species):
	ff=nc.Dataset(filein)
	#for i in ff.variables:
	#	if species in ff.variables[i]:
	data=[]
	if 'emi'+species in ff.variables:
		data.append(ff.variables['emi'+species][:])
	else:
		data.append(np.zeros_like(ff.variables['wet'+species][:]))
	data.append(ff.variables['wet'+species][:])
	data.append(ff.variables['dry'+species][:])
	data.append(ff.variables['load'+species][:])
	return data

def read_data_type(filein,species,budtype):
	"""
	read data from filein for give data type
	"""
	ff=nc.Dataset(filein)
	#for i in ff.variables:
	#	if species in ff.variables[i]:
	data={}
	for spec in species:
		print spec
		print budtype,spec
		if budtype=='emi':
			if budtype+spec in ff.variables:
				data[budtype+spec]=ff.variables[budtype+spec][:]
			else:
				data[budtype+spec]=np.zeros_like(ff.variables['wet'+spec][:])
		else:
			data[budtype+spec]=ff.variables[budtype+spec][:]
	return data

def read_emi(filein,species=['bc','oa','soa','so4','ss','dust','so2','isop','terp']):
	"""
	read emission data for species
	"""
	emidata=read_data_type(filein,species,'emi') 
	print emidata.keys()
	return emidata
def read_dry(filein,species=['bc','oa','soa','so4','ss','dust']):
	"""
	read drydep data for species
	"""
	drydata=read_data_type(filein,species,'dry') 
	print drydata.keys()
	return drydata
	pass
def read_wet(filein,species=['bc','oa','soa','so4','ss','dust']):
	"""
	read wetdep data for species
	"""
	wetdata=read_data_type(filein,species,'wet') 
	print wetdata.keys()
	return wetdata
	pass
def read_load(filein,species=['bc','oa','soa','so4','ss','dust']):
	"""
	read burden data for species
	"""
	print filein
	loaddata=read_data_type(filein,species,'load') 
	print loaddata.keys()
	return loaddata
	pass
def read_old_SOA(fname='SOA.nc'):
	fsoa=nc.Dataset(fixeddata+'/'+fname,'r')
	if fname=='SOA.nc':
		olddata=fsoa.variables['FIELD'][:]
		gb1d=fsoa.variables['GRIDBOX_AREA'][:]
	elif fname=='MEGAN2-2010-SOA.nc':
		olddata=fsoa.variables['MEGAN2_SOA'][:]
		gb1d=fsoa.variables['cell_area'][:]
	return olddata*2.4/1.15,gb1d

def prod_data(filein,species):
	"""
	Read data for chemical production
	"""
	ff=nc.Dataset(filein)
	data=[]
	if (species=='soa' and 'prod_elvoc'in ff.variables):
		data.append(ff.variables['prod_elvoc'][:])
	if (species=='soa' and 'prod_svoc'in ff.variables):
		data.append(ff.variables['prod_svoc'][:])
	if (species=='so4' and ('prod_liq_so4' in ff.variables)):
		#print 'liq'
		data.append(ff.variables['prod_liq_'+species][:])
	if (species=='so4' and 'prod_gas_so4'in ff.variables):
		#print 'gas'
		data.append(ff.variables['prod_gas_'+species][:])
	
	return data
def prod_data_source(filein,species):
	"""
	Read data for chemical production by pathway
	"""
	ff=nc.Dataset(filein)
	data={}
	if (species=='soa' and 'p_el_ohterp'in ff.variables):
		#print np.sum(ff.variables['p_el_ohterp'][:])
		data['el_ohterp']=np.array(ff.variables['p_el_ohterp'][:])
	if (species=='soa' and 'p_el_o3terp'in ff.variables):
		#print np.sum(ff.variables['p_el_o3terp'][:])
		data['el_o3terp']=np.array(ff.variables['p_el_o3terp'][:])
	if (species=='soa' and 'p_el_ohisop'in ff.variables):
		#print np.sum(ff.variables['p_el_ohisop'][:])
		data['el_ohisop']=np.array(ff.variables['p_el_ohisop'][:])
	if (species=='soa' and 'p_el_o3isop'in ff.variables):
		#print np.sum(ff.variables['p_el_o3isop'][:])
		data['el_o3isop']=np.array(ff.variables['p_el_o3isop'][:])
	if (species=='soa' and 'p_sv_ohterp'in ff.variables):
		#print np.sum(ff.variables['p_sv_ohterp'][:])
		data['sv_ohterp']=np.array(ff.variables['p_sv_ohterp'][:])
	if (species=='soa' and 'p_sv_o3terp'in ff.variables):
		#print np.sum(ff.variables['p_sv_o3terp'][:])
		data['sv_o3terp']=np.array(ff.variables['p_sv_o3terp'][:])
	if (species=='soa' and 'p_sv_ohisop'in ff.variables):
		#print np.sum(ff.variables['p_sv_ohisop'][:])
		data['sv_ohisop']=np.array(ff.variables['p_sv_ohisop'][:])
	if (species=='soa' and 'p_sv_o3isop'in ff.variables):
		#print np.sum(ff.variables['p_sv_o3isop'][:])
		data['sv_o3isop']=np.array(ff.variables['p_sv_o3isop'][:])
	return data
def read_voc(filein):
	"""
	Read VOC emissions
	"""
	isop=nc.Dataset(filein,'r').variables['emiisop'][:]
	terp=nc.Dataset(filein,'r').variables['emiterp'][:]
	return terp,isop
def read_so2(filein):
	"""
	Read VOC emissions
	"""
	so2=nc.Dataset(filein,'r').variables['emiso2'][:]
	return so2
def read_dms(filein):
	"""
	Read VOC emissions
	"""
	dms=nc.Dataset(filein,'r').variables['emidms'][:]
	return dms
def mass_budget(filein):
	"""
	calculate the budgets
	"""

	data=np.zeros((6,6,12,90,120))
	pd=np.zeros((2,2,12,34,90,120))
	names=['bc','oa','soa','so4','ss','dust']
	for i in range(len(names)):
		data[i,:4,:,:,:]=np.array(read_data(filein,names[i]))
	
	pd[0,:,:,:,:,:]=np.array(prod_data(filein,'so4'))
	pd[1,:,:,:,:,:]=np.array(prod_data(filein,'soa'))
	pdsource=prod_data_source(filein,'soa')

	days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])

	gb=get_gridboxarea('TM53x2')
	# calculate SOA production in the OLD scheme from 1x1 deg data
	for i in pdsource:
		temp=0
		for m in range(12):
				for j in range(34):
					temp+=np.sum(pdsource[i][m,j,:,:]*gb[:,:]*3600*24*days[m])
		print i,': ',temp*1e-9
	# calcualte the global budgetdata 
	dataglobal=np.zeros((6,4))
	for j in range(6):
		for i in range(4):
			for m in range(12):
				if i <3:
					dataglobal[j,i]+=np.sum(data[j,i,m,:,:]*gb[:,:]*3600*24*days[m])
				else:
					dataglobal[j,i]+=np.sum(data[j,i,m,:,:]*gb[:,:])*days[m]
	dataglobal=dataglobal
	# calculate chemical production of SOA and SO4
	proddata=np.zeros((2,2))
	for k in range(2):
		for i in range(2):
			for m in range(12):
				for j in range(34):
					proddata[k,i]+=np.sum(pd[k,i,m,j,:,:]*gb[:,:]*3600*24*days[m])
	proddata=proddata
	print filein
	C_ratio_elvoc= (10*16)/(10*16+16*1.+7*12)
	C_ratio_svoc= (10*16)/(10*16+16*1.+6*12)
	for i in range(6):
		print names[i]
		print 'emi',dataglobal[i,0]*1e-9
		if (i==2):#SOA
			print 'prod ELVOC',(proddata[1,0])*1e-9, (proddata[1,0])*1e-9*C_ratio_elvoc
			print 'prod SVOC',(proddata[1,1])*1e-9, (proddata[1,1])*1e-9*C_ratio_svoc
		elif (i==3):#SO4
			print 'prod liq',proddata[0,0]*1e-9
			print 'prod gas',proddata[0,1]*1e-9
		print 'wet',dataglobal[i,1]*1e-9
		print 'dry',dataglobal[i,2]*1e-9
		print 'load',dataglobal[i,3]*1e-9/np.sum(days)
		if (i==2):
			print 'total',(dataglobal[i,0]+proddata[1,0]+proddata[1,1]-(dataglobal[i,1]+dataglobal[i,2]))*1e-9
		elif (i==3):
			print 'total',(dataglobal[i,0]+proddata[0,0]+proddata[0,1]-(dataglobal[i,1]+dataglobal[i,2]))*1e-9
		else:
			print 'total',(dataglobal[i,0]-(dataglobal[i,1]+dataglobal[i,2]))*1e-9
#	for i in range(2):
#		print 'prod ',proddata[i,0]*1e-9
#		print 'prod ',proddata[i,1]*1e-9
	return dataglobal,proddata,data,pd
def budget_to_latex(data):
	gb=get_gridboxarea('TM53x2')
	olddata=load_original_production_soa()
	olddata_M2=load_original_production_soa('MEGAN2-2010-SOA.nc')
	oldTM5data=0
	oldTM5M2data=0
	for i,ndays in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
		oldTM5data=oldTM5data+(olddata[i,:,:]*3600*24*ndays*gb).sum()
		oldTM5M2data=oldTM5M2data+(olddata[i,:,:]*3600*24*ndays*gb).sum()
	#print oldTM5data/1e9
	totalremoval={}
	totalremoval['newsoa-ri']={}
	totalremoval['oldsoa-bhn']={}
	totalremoval['oldsoa-bhn-megan2']={}

	for k, kaero in enumerate(['BC','OA','SOA','SO4','SS','DU']):
		for i in data:
			totalremoval[i][kaero]=data[i][0][k][1]/1e9+data[i][0][k][2]/1e9

	days=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
	for k, kaero in enumerate(['BC','OA','SOA','SO4','SS','DU']):
		print kaero
		for j,jbug in enumerate(['Emission','Wet Deposition','Dry Deposition','Burden']):

			if jbug== 'Burden':
				print jbug,'\t',
			else:
				print jbug,
			for i in data:
				#print i
				if jbug=='Production':
					out=data[i][1][k]
				elif jbug=='Burden':
					out= proddata=data[i][0][k][j]/1e9/days.sum()	

				else:	
					out=data[i][0][k][j]/1e9
					if jbug=='Wet Deposition' or jbug=='Dry Deposition':
						outfrac=data[i][0][k][j]/1e9/totalremoval[i][kaero]
				if jbug=='Emission' and kaero=='SOA':
					print '\t',
				elif jbug=='Wet Deposition' or jbug=='Dry Deposition':
					print '\t{:6.2f}'.format(out),'\t{:6.2f}'.format(outfrac*100),
				else:
					print '\t{:6.2f}'.format(out),
			print 
			if jbug=='Emission' and (kaero=='SO4' or kaero=='SOA'):
				print 'Production',
				for i in  data:
					if kaero=='SO4':
						#SO4
						out = (data[i][1][0][0]+data[i][1][0][1])/1e9
					elif kaero=='SOA':
						#SOA
						if i == 'oldsoa-bhn':
							out = oldTM5data/1e9
						else:
							out = (data[i][1][1][0]+data[i][1][1][1])/1e9
					print '\t{:6.2f}'.format(out),
				print

		
def mapit_log(data,clevs,ax):
	"""
	Map data with logarithmic scale
	"""
	#plt.figure()
	lon,lat=lonlat('TM53x2')
	m = Basemap(projection='robin',lat_0=0,lon_0=0,ax=ax)
	m.drawcoastlines(linewidth=0.25)
	m.drawmeridians(np.arange(0,360,30))
	m.drawparallels(np.arange(-90,90,30))
	x, y = meshgrid(lon, lat)
	px,py=m(x,y)
	#clevs=[1e-14,1e-13,1e-12,1e-11,1e-10]
	cs = m.contourf(px,py,np.squeeze(data),clevs,norm=mpl.colors.LogNorm(clevs[0],clevs[-1]),shading='flat',cmap=plt.cm.Greens)
	#x1,y1=m(info[1],info[0])
	#m.scatter(x1,y1,c='r',marker='o',s=100)
	m.colorbar(cs,location='bottom')
def mapit_boundary(data,clevs,ax,diverging=False,**kwargs):
	"""
	Map data with boundary value scale
	"""
	#plt.figure()
	if diverging:
		mycmap=plt.get_cmap('RdBu_r',len(clevs)-1) 
		#mycmap=plt.get_cmap('bwr',len(clevs)) 
	else:
		mycmap=plt.get_cmap('Greens',len(clevs)-1) 
	lon,lat=lonlat('TM53x2')
	m = Basemap(projection='robin',lat_0=0,lon_0=0,ax=ax)
	m.drawcoastlines(linewidth=0.25)
	m.drawmeridians(np.arange(0,360,30))
	m.drawparallels(np.arange(-90,90,30))
	x, y = meshgrid(lon, lat)
	px,py=m(x,y)
	#clevs=[1e-14,1e-13,1e-12,1e-11,1e-10]
	if np.amax(data)>clevs[-1]:
		cs = m.contourf(px,py,np.squeeze(data),clevs,norm=mpl.colors.BoundaryNorm(clevs,mycmap.N),shading='flat', extend='max',cmap=mycmap)
	else:
		cs = m.contourf(px,py,np.squeeze(data),clevs,norm=mpl.colors.BoundaryNorm(clevs,mycmap.N),shading='flat',cmap=mycmap)
	#x1,y1=m(info[1],info[0])
	#m.scatter(x1,y1,c='r',marker='o',s=100)
	if diverging:
		cb=m.colorbar(cs,location='bottom',ticks=clevs)
	else:
		cb=m.colorbar(cs,location='bottom',ticks=clevs, extend='max')
	if 'cblabel' in kwargs:
		cblabel=kwargs['cblabel']
	else:
		cblabel=''
	cb.ax.set_xticklabels(clevs,fontsize=14)
	cb.set_label(cblabel, fontsize=16)
	plt.tight_layout()
def budgetonly(filename):
	"""
	calculate budget only
	"""
	print  'processing ' + filename
	print '\n'
	data1,prod1,alldata,allprod=mass_budget(filename)
	voc1,voc2=read_voc(filename)
	return data1,prod1,alldata,allprod,voc1,voc2
def dostuff(filename):
	data1,prod1,alldata,allprod=mass_budget(filename)
	voc1,voc2=read_voc(filename)
	ff=nc.Dataset(filename,'r')
	am=ff.variables['airmass']
	p=ff.variables['pressure']
	t=ff.variables['temp']
	load_soa=ff.variables['loadsoa'][:]
	wet_soa=ff.variables['wetsoa'][:]

	#print np.shape(voc1)
	elvo=ff.variables['GAS_ELVOC'][:]
	elvo2=ff.variables['prod_elvoc'][:]
	emiterp,emiisop=read_voc(filename)
	
	#print np.shape(emiterp)
	gg=nc.Dataset('output/general_TM5_SOA-s_2009.mm.so2.nc','r')
	wetso2=gg.variables['wetso2']
	dryso2=gg.variables['dryso2']
	
	gb=get_gridboxarea('TM53x2')
	load_soamm=np.zeros((12))
	wet_soamm=np.zeros((12))
	for i in range(12):
		load_soamm[i]=np.sum(load_soa[i,:,:]*gb*1e-9)
		wet_soamm[i]=np.sum(wet_soa[i,:,:]*gb*1e-9)
	#plt.figure()
	#plt.plot(load_soamm)
	#plt.figure()
	#plt.plot(wet_soamm)
	#plt.show()
	wetso2_sum=0
	dryso2_sum=0
	for i in range(12):
		wetso2_sum+=np.sum(wetso2[i,:,:]*gb)*1e-9
		dryso2_sum+=np.sum(dryso2[i,:,:]*gb)*1e-9
	print 'wetso2',wetso2_sum/12.0*3600*24*365/2
	print 'dryso2',dryso2_sum/12.0*3600*24*365/2
	#print 'emiterp',np.sum(emiterp*gb)*1e-9*3600*365*24/12,np.sum(emiterp*gb)*1e-9*3600*365*24*0.15
	#print 'emiisop',np.sum(emiisop*gb)*1e-9*3600*365*24/12,np.sum(emiisop*gb)*1e-9*3600*365*24*0.05
	print 'emiterp',np.sum(emiterp*gb)*1e-9*3600*365*24/12#,(prod1[1,0]*1e-9)/(np.sum(emiterp*gb)*1e-9*3600*365*24/12)
	print 'emiisop',np.sum(emiisop*gb)*1e-9*3600*365*24/12#,(prod1[1,0]*1e-9)/(np.sum(emiisop*gb)*1e-9*3600*365*24/12)
	elvoc=ff.variables['GAS_ELVOC']
	svoc=ff.variables['GAS_SVOC']

	lon,lat=lonlat('TM53x2')
	clevs=[1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6]
	clevs=[1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]
	mapit_log(np.squeeze(np.mean(np.sum(allprod[1,:,:,0,:,:],axis=0),axis=0)*gb),clevs)
	mapit_log(np.squeeze(np.mean(np.sum(allprod[1,:,:,3,:,:],axis=0),axis=0)*gb),clevs)
	clevs=[1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8]
	clevs=[1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1]
	mapit_log(np.squeeze(np.mean(elvoc[0,:,:,:]+svoc[0,:,:,:],axis=0)*1e9),clevs)
	plt.title('concentration of VOC [mg m-3]')
	mapit_log(np.squeeze(np.mean(elvoc[1,:,:,:]+svoc[1,:,:,:],axis=0)*1e9),clevs)
	plt.title('concentration of VOC [mg m-3]')
	mapit_log(np.squeeze(np.mean(elvoc[2,:,:,:]+svoc[2,:,:,:],axis=0)*1e9),clevs)
	plt.title('concentration of VOC [mg m-3]')
	mapit_log(np.squeeze(np.mean(elvoc[3,:,:,:]+svoc[3,:,:,:],axis=0)*1e9),clevs)
	plt.title('concentration of VOC [mg m-3]')
	mapit_log(np.squeeze(np.mean(elvoc[4,:,:,:]+svoc[4,:,:,:],axis=0)*1e9),clevs)
	plt.title('concentration of VOC [mg m-3]')
	plt.show()
def seasonal_maps(data,bounds,diverging=False,cblabel=''):
	"""
	Plot data seasonlly, input data needs to have dimension (12,:,:)
	"""
	f,a=plt.subplots(ncols=2,nrows=2,figsize=(12,8))
	lon,lat=lonlat('TM53x2')
	lons, lats = np.meshgrid(lon,lat)
	seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
	seas=['DJF','MAM','JJA','SON']
	jj=-1
	bounds_load=bounds#[-10,-5,-3,-2,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,2,3,5,10]
	cb=[]
	for i in range(4):
		ii=i
		kk=0
		if ii>1:
			ii=ii-2
			kk=1
			#jj+=1
		#for exp in EXPS:
		norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
		#mycmap=plt.cm.coolwarm
		if diverging:
			mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
		else:
			mycmap=plt.get_cmap('Greens',len(bounds_load)) 
		#a[kk,ii].set_title('Fractional change in {}'.format(seas[i]))
		m=Basemap(projection='robin',lon_0=0,ax=a[kk,ii])
		image=m.pcolormesh(lons,lats,data[seasonidx[i],:,:].mean(0),norm=norm,cmap=mycmap,latlon=True)
		m.drawparallels(np.arange(-90.,90.,30.))
		m.drawmeridians(np.arange(-180.,180.,60.))
		m.drawcoastlines()
		cb= m.colorbar(image,"bottom", size="5%", pad="2%")
		cb.set_label(cblabel)
		a[kk,ii].set_title(seas[i])
	return f,a,cb
	#f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_NEWSOA-NOSOA.png',dpi=600)

def convert_1x1_2_3x2(indata):
	""" 
	sum 1x1 degree data into 3x2 degree grid 
	"""

	outti=np.zeros((12,90,120))
	for kk in range(90):
		for jj in range(120):
			#print kk,kk+1,jj,jj+2
			outti[:,kk,jj] = indata[:,kk*2:kk*2+1,jj*3:jj*3+2].sum() 
	return outti

def load_original_production_soa(fname='SOA.nc'):
	# SOA production stored in C/gridbox (1x1 grid) 
	gb=get_gridboxarea('TM53x2')
	if fname=='MEGAN2-2010-SOA.nc':
		varname='MEGAN2_SOA'
		gbname='cell_area'
	else:
		varname='FIELD'
		gbname='GRIDBOX_AREA'

	fsoa=nc.Dataset(fixeddata+'/'+fname,'r')
	olddata=fsoa.variables[varname][:]

	gb1d=fsoa.variables[gbname][:]
	olddata=olddata#/gb1d/(24*30)
	print 'direct old SOA production globally (TgC)',olddata.sum()*1e-9,olddata.sum()*1e-9/.015
	oldTM5=np.zeros((12,90,120))
	#print oldTM5.shape

	# create 3x2 degree gridded production data
	for k in range(12):
		for i in range(90):
			for j in range(120):
				#print i*2,i*2+2,j*3,j*3+3
				oldTM5[k,i,j]=olddata[k,i*2:i*2+2,j*3:j*3+3].sum()
	testisumma=0

	# change units to same with NEWSOA
	for k in range(12):
		testisumma=testisumma+(oldTM5[k,:,:]).sum()
		#kg/gridbox->kg/(m2 s)
		oldTM5[k,:,:]=oldTM5[k,:,:]/(gb[:,:]*24.0*monthlengths[k]*3600)*2.4/1.15
	return oldTM5

if __name__=='__main__':

	#output_png_path='/Users/bergmant/Documents/tm5-soa/figures/png'
	#pdfpath='/Users/bergmant/Documents/tm5-soa/figures/pdf'
	#jpgpath='/Users/bergmant/Documents/tm5-soa/figures/jpg'
	EXPS=['newsoa-ri','oldsoa-bhn','oldsoa-bhn-megan2']
	data={}
	monthlengths=[31,28,31,30,31,30,31,31,30,31,30,31]
	#for exp in EXPS:
	for exp in ['newsoa-ri','oldsoa-bhn','oldsoa-bhn-megan2']:
		data[exp]=budgetonly(basepathraw+'/'+exp+'/general_TM5_'+exp+'_2010.mm.nc')
		#for kk in range(6):
			#	for jj in range(4):
			#print exp,np.shape(data[exp][kk])
		if exp=='oldsoa-bhn':
			#print data[exp]
			oldsoadata,gb1d=read_old_SOA()
			#print np.shape(oldsoadata)	
			oldsoadata=convert_1x1_2_3x2(oldsoadata)
			#print np.shape(oldsoadata)
			
			# for kk in range(6):
			# 	for jj in range(4):
			# 		print np.shape(data[exp])
			# 		print oldsoadata[kk,jj]

			#print oldsoadata.sum()/1e-9
			#print np.shape(oldsoadata)
			#print exp,data[exp]
			#print np.shape(data[exp][0])
			#print np.shape(data['newsoa-ri'][0])
			data[exp][3][0,0,:,0,:,:]=oldsoadata
			data[exp][0][2,0]=oldsoadata.sum()#*grid
		elif exp=='oldsoa-bhn-megan2':
			#print data[exp]
			oldsoadata,gb1d=read_old_SOA('MEGAN2-2010-SOA.nc')
			#print np.shape(oldsoadata)	
			oldsoadata=convert_1x1_2_3x2(oldsoadata)
			#print np.shape(oldsoadata)
			
			# for kk in range(6):
			# 	for jj in range(4):
			# 		print np.shape(data[exp])
			# 		print oldsoadata[kk,jj]

			#print oldsoadata.sum()/1e-9
			#print np.shape(oldsoadata)
			#print exp,data[exp]
			#print np.shape(data[exp][0])
			#print np.shape(data['newsoa-ri'][0])
			data[exp][3][0,0,:,0,:,:]=oldsoadata
			data[exp][0][2,0]=oldsoadata.sum()#*grid

	print EXPs[0],data[EXPs[0]][0],data[EXPs[0]][1]
	print EXPs[1],data[EXPs[1]][0],data[EXPs[1]][1]
	gb=get_gridboxarea('TM53x2')
	emiso2=read_so2(basepathraw+'/'+exp+'/general_TM5_'+exp+'_2010.mm.nc')	
	emidms=read_dms(basepathraw+'/'+exp+'/general_TM5_'+exp+'_2010.mm.nc')	
	emiso2_sum=0
	emidms_sum=0
	for i in range(12):
		emiso2_sum+=np.sum(emiso2[i,:,:]*gb)*1e-9*3600*24*monthlengths[i]
		emidms_sum+=np.sum(emidms[i,:,:]*gb)*1e-9*3600*24*monthlengths[i]
	print emiso2_sum
	print emidms_sum
	#budget_to_latex(data)
	lon,lat=lonlat('TM53x2')
	gb=get_gridboxarea('TM53x2')

	# f,a=plt.subplots(ncols=4,nrows=3)
	# #f2,a2=plt.subplots(1)
	# k=-1
	# gb=get_gridboxarea('TM53x2')
	# print 'emiterp',np.sum(np.sum(data[EXPS[0]][4],axis=0)*gb)*1e-9*3600*365*24/12#,(prod1[1,0]*1e-9)/(np.sum(emiterp*gb)*1e-9*3600*365*24/12)
	# print 'emiisop',np.sum(np.sum(data[EXPS[0]][5],axis=0)*gb)*1e-9*3600*365*24/12#,(prod1[1,0]*1e-9)/(np.sum(emiisop*gb)*1e-9*3600*365*24/12)
	# lon,lat=lonlat('TM53x2')
	# clevs=[0,50,100,300,500,700,1000,5000]
	# for i in range(4):
	# 	for j in range(3):
	# 		k+=1
	# 		print ( np.squeeze(np.sum(data[EXPS[0]][3][1,:,k,0,:,:],axis=0)*gb)).max(),a.shape,j,i
	# 		#mapit_boundary(np.squeeze(np.sum(data['soa-riccobono'][3][1,:,k,0,:,:],axis=0)*gb)*1e3,clevs,a[j,i])
	# 		#a[i,j].plot(data['soa-riccobono'][5].sum(1).sum(1))
	# 		#a[i,j].plot(data[exp][5].sum(1).sum(1))
	# 		#a[i,j].plot(data['nosoa'][5].sum(1).sum(1))
	# 		print np.squeeze(np.mean(np.sum(data[EXPS[0]][3][1,:,k,0,:,:],axis=0)*gb*1e3,axis=1)).shape
	# 		a[j,i].plot(np.squeeze(np.mean(np.sum(data[exp][3][1,:,k,0,:,:],axis=0)*gb*1e3,axis=1)))
	# 		a[j,i].plot(np.squeeze(np.mean(np.sum(data[exp][3][1,:,k,0,:,:],axis=0)*gb*1e3,axis=1)))
	# 		a[j,i].plot(np.squeeze(np.mean(np.sum(data[exp][3][1,:,k,0,:,:],axis=0)*gb*1e3,axis=1)))

	# print data[EXPS[0]][3].shape,data[EXPS[0]][3].max()*1e9*3600
	# #f,ax=plt.subplots(1)
	#mapit_boundary(np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0).mean(0))*1e6*3600*24*365,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75,150,250,500],ax)
	
	#f,a,cb=seasonal_maps(np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0))*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75,150,250,500],diverging=False,cblabel='Production of SOA [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/production/NEWSOA_production_SOA.png',dpi=600)
	f,a,cb=seasonal_maps(np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75,150,500],diverging=False,cblabel='Production of SOA [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/supplement/figS1_NEWSOA_production_SOA.png',dpi=600)
	#f,a,cb=seasonal_maps((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,1:,:,:],axis=(0,2)))-np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0)))*1e9*3600,[-50,-25,-10,-7.5,-5,-2.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50],diverging=True,cblabel='Diff sum(level 1-n)-level0 in Production of SOA [ug m-2 hr-1]')

	f,a,cb=seasonal_maps(np.squeeze(data[EXPS[0]][3][1,0,:,0,:,:])*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75],diverging=False,cblabel='Production of ELVOC [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/supplement/figS5_NEWSOA_production_ELVOC.png',dpi=600)
	f,a,cb=seasonal_maps(np.squeeze(data[EXPS[0]][3][1,1,:,0,:,:])*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75],diverging=False,cblabel='Production of SVOC [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/supplement/figS6_NEWSOA_production_SVOC.png',dpi=600)
	# f,a=plt.subplots(1)
	# print np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0)).mean(2).shape,np.linspace(0,33,34).shape,lat.shape,np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=0)).mean(0).mean(-1).max()*1e9*3600
	# levs,lats=np.meshgrid(np.linspace(0,33,34),lat)
	# #a.pcolormes(lat,np.linspace(0,33,34),np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=0)).mean(0).mean(-1)*1e9*3600)
	# bounds=[0,0.25,0.5,0.75,1.0,1.5,2,2.5,3,4,4.5,5,5.5]
	# mycmap=plt.cm.get_cmap('Greens',len(bounds))
	# normi = mpl.colors.BoundaryNorm(bounds, len(bounds))
	# im=a.pcolormesh(lats,levs,np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=0)).mean(0).mean(-1).transpose()*1e9*3600,norm=normi,cmap=mycmap)
	# plt.colorbar(im)
	oldTM5=load_original_production_soa()
	oldTM5M2=load_original_production_soa('MEGAN2-2010-SOA.nc')
	f,a,cb=seasonal_maps(np.squeeze(oldTM5)*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75,150,500],diverging=False,cblabel='Production of SOA [ug m-2 hr-1]')
	f.savefig(output_png_path+'/supplement/figS2_OLDSOA_production_SOA.png',dpi=600)
	
	f,a,cb=seasonal_maps(np.squeeze(oldTM5M2)*1e9*3600,[0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,75,150,250,500],diverging=False,cblabel='emis (OLDSOA-MEGAN2) of SOA [ug m-2 hr-1]')
	f.savefig('figSX_OLDSOA-MEGAN2_production_SOA.png',dpi=600)
	#print data[EXPS[0]][3][1,:,:,0,:,:].shape
	hdata=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0))*1e9*3600
	f,a,cb=seasonal_maps((hdata-np.squeeze(oldTM5)*1e9*3600),[-50,-25,-10,-7.5,-5,-2.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50],diverging=True,cblabel='emis (OLDSOA) of SOA [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/supplement/figS4_NEWSOA-OLDSOA_production_SOA.png',dpi=600)
	
	hdata=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))#*1e9*3600
	oldie=oldTM5.copy()
	oldie[oldTM5==0]=1
	f,a,cb=seasonal_maps((hdata-np.squeeze(oldTM5))/np.squeeze(oldie)*100,[-150,-100,-75,-50,-25,-10,-7.5,-5,5,7.5,10,25,50,75,100,150],diverging=True,cblabel='change in production [%]')
	#f.savefig(output_png_path+'/supplement/figS3_NEWSOA-OLDSOA_production_SOA_percentage.png',dpi=600)
	
	#hdata=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,1,:,:],axis=0))*1e9*3600
	#f,a,cb=seasonal_maps((hdata-np.squeeze(oldTM5)*1e9*3600),[-50,-25,-10,-7.5,-5,-2.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50],diverging=True,cblabel='emis (OLDSOA) of SOA [ug m-2 hr-1]')
	#hdata=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,2,:,:],axis=0))*1e9*3600
	#f,a,cb=seasonal_maps((hdata-np.squeeze(oldTM5)*1e9*3600),[-50,-25,-10,-7.5,-5,-2.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50],diverging=True,cblabel='emis (OLDSOA) of SOA [ug m-2 hr-1]')
	hdata2=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))*1e9*3600
	f,a,cb=seasonal_maps((hdata2-np.squeeze(oldTM5)*1e9*3600),[-150,-50,-25,-10,-7.5,-5,-2.5,-1.0,-0.5,-0.1,-0.01,0.01,0.1,0.5,1.0,2.5,5,7.5,10,25,50,150],diverging=True,cblabel='Difference in SOA production [ug m-2 hr-1]')
	#f.savefig(output_png_path+'/supplement/figS4_NEWSOA-OLDSOA_production_SOA.png',dpi=600)

	# print (np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,1:,:,:],axis=(0,2))).mean(0)*gb*3600*24*365).sum()	
	# print (np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2))).mean(0)*gb*3600*24*365).sum()
	# print (np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0)).mean(0)*gb*3600*24*365).sum()
	# prodlev=np.zeros(34)
	# prodlev_old=np.zeros(34)
	# for ilev in range(34):
	# 	#print (np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,ilev,:,:],axis=0)).mean(0)*gb*3600*24*365).sum()
	# 	prodlev[ilev]=(np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,ilev,:,:],axis=0)).mean(0)*gb*3600*24*365).sum()
	# 	#prodlev2[ilev]
	# f,a=plt.subplots(ncols=2)
	# a[0].plot(prodlev/1e9,np.linspace(1,35,34),'r')
	# prodlev_old[0]=(np.squeeze(oldTM5)*gb).sum()*3600*24*365*0.8/12
	# prodlev_old[1]=(np.squeeze(oldTM5)*gb).sum()*3600*24*365*0.2/12
	# a[0].plot(prodlev_old/1e9,np.linspace(1,35,34),'b')
	# a[0].set_xlabel('SOA production [Tg yr$^{-1}$]')
	# a[0].set_ylabel('Model level')
	# #plt.figure()
	# a[1].plot(prodlev/prodlev.sum()*100,np.linspace(1,35,34),'r')
	# a[1].plot(prodlev_old/prodlev_old.sum()*100,np.linspace(1,35,34),'b')
	# a[1].set_xlabel('Fraction of total SOA production [%]')
	# a[1].set_ylabel('Model level')
	# f.savefig(output_png_path+'/production/production_SOA_vertical.png',dpi=600)

	# plt.plot()
	# plt.figure()
	# plt.semilogx(prodlev,np.linspace(1,35,34))

	# #plt.show()
	# print prodlev[2:].sum()/prodlev.sum()*100
	# print prodlev_old[2:].sum()/prodlev_old.sum()*100
	


	# f,ax=plt.subplots(1)
	
	# levels=	[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2]
	# mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,levels,ax,True,cblabel='Difference in production [g m$^{-2}$ yr$^{-1}$]')
	# ax.set_title('Difference in production')
	# f.savefig(output_png_path+'/production/NEWSOA-OLDSOA_annual_production_SOA.png',dpi=600)
	f,ax=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
	#print np.shape(data[EXPS[0]][3])
	#NEWSOA
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))).mean(0)*1e3*3600*24*365,[0.05,0.1,0.25,0.5,1,2,3,4,5],ax[0,0],False,cblabel='Annual SOA production [g m$^{-2}$ yr$^{-1}$]')
	#OLDSOA
	mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[0.05,0.1,0.25,0.5,1,2,3,4,5],ax[0,1],False,cblabel='Annual SOA production [g m$^{-2}$ yr$^{-1}$]')
	#DIFF
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[1,0],True,cblabel='Absolute difference in production [[g m$^{-2}$ yr$^{-1}$]')
	#DIFF
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[1,1],True,cblabel='??????Difference in production [g m$^{-2}$ yr$^{-1}$]')
	#f.savefig(output_png_path+'/production/annual-production-SOA-2x2.png',dpi=600)
	f,ax=plt.subplots(nrows=3,ncols=2,figsize=(12,8))
	#print np.shape(data[EXPS[0]][3])
	#NEWSOA
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))).mean(0)*1e3*3600*24*365,[0.05,0.1,0.25,0.5,1,2,3,4,5],ax[0,0],False,cblabel='Annual SOA production [g m$^{-2}$ yr$^{-1}$]')
	#OLDSOA
	mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[0.05,0.1,0.25,0.5,1,2,3,4,5],ax[0,1],False,cblabel='Annual SOA production [g m$^{-2}$ yr$^{-1}$]')
	#DIFF
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[1,0],True,cblabel='Absolute difference in production [[g m$^{-2}$ yr$^{-1}$]')
	#DIFF
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[1,1],True,cblabel='??????Difference in production [g m$^{-2}$ yr$^{-1}$]')

	mapit_boundary((np.squeeze(oldTM5M2).mean(0)*1e3*3600*24*365-(np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365),[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[2,0],True,cblabel='??????Difference in production [g m$^{-2}$ yr$^{-1}$]')
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5M2)).mean(0)*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[2,1],True,cblabel='??????Difference in production [g m$^{-2}$ yr$^{-1}$]')
	f.savefig('annual-production-SOA-3x2.png',dpi=600)
	# f,ax=subplots(3)
	# print np.squeeze(np.mean(data[EXPS[0]][2][2,1,:,:,:],axis=(0))).shape
	# print type(np.squeeze(np.mean(data[EXPS[0]][2][2,1,:,:,:],axis=(0)))*1e3*3600*24*365)
	# mapit_boundary(np.squeeze(np.mean(data[EXPS[0]][2][2,1,:,:,:],axis=(0)))*1e3*3600*24*365,[0.01,0.05,0.1,0.5,1,2],ax[0],False,cblabel='wetsoa [g m$^{-2}$ yr$^{-1}$]')
	# mapit_boundary(np.squeeze(np.mean(data[EXPS[1]][2][2,1,:,:,:],axis=(0)))*1e3*3600*24*365,[0.01,0.05,0.1,0.5,1,2],ax[1],False,cblabel='wetsoa [g m$^{-2}$ yr$^{-1}$]')
	# mapit_boundary(np.squeeze(np.mean(data[EXPS[0]][2][2,1,:,:,:],axis=(0))-np.mean(data[EXPS[1]][2][2,1,:,:,:],axis=(0)))*1e3*3600*24*365,[-2,-1,-0.5,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.5,1,2],ax[2],True,cblabel='wetsoa [g m$^{-2}$ yr$^{-1}$]')
	
	terp=nc.Dataset(output+'/general_TM5_newsoa-ri_2010.mm.nc','r')['emiterp']
	isop=nc.Dataset(output+'/general_TM5_newsoa-ri_2010.mm.nc','r')['emiisop']
	dms=nc.Dataset(output+'/general_TM5_newsoa-ri_2010.mm.nc','r')['emidms']
	so2=nc.Dataset(output+'/general_TM5_newsoa-ri_2010.mm.nc','r')['emiso2']
	#print 'emidms',(dms[:]*gb[:]).sum()/gb.sum()/1e9/days.sum()
	#print 'emiso2',(so2*gb).sum()/gb.sum()/1e9/days.sum()
	###
	# f,aa=plt.subplots(1)
	# print terp.shape
	# terp2=terp[:,:,:]*gb
	# isop2=isop[:,:,:]*gb
	# terpzon=terp2[:,75:,:].sum(axis=(1,2))
	# isopzon=isop2[:,75:,:].sum(axis=(1,2))
	# oldterp2=oldTM5[:,:,:]*gb/0.15/2.4*1.15
	
	# oldterpzon=oldterp2[:,75:,:].sum(axis=(1,2))
	# print 'lat 75',lat[75]
	# aa.plot(terpzon,'r')
	# aa.plot(isopzon,'--r')
	# aa.plot(oldterpzon,'b')
	fmapvoc,ax=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
	oldterp=oldTM5[:,:,:]/0.15/2.4*1.15
	sumterp=np.zeros((12,90,120))
	sumisop=np.zeros((12,90,120))
	sumoldterp=np.zeros((12,90,120))
	for i in range(12):
		sumterp[i,:,:]=terp[i,:,:]*monthlengths[i]*3600*24
		sumoldterp[i,:,:]=oldterp[i,:,:]*monthlengths[i]*3600*24
		sumisop[i,:,:]=isop[i,:,:]*monthlengths[i]*3600*24
	sumterp=sumterp.sum(axis=0)#/0.15/2.4*1.15
	sumisop=sumisop.sum(axis=0)#/0.15/2.4*1.15
	sumoldterp=sumoldterp.sum(axis=0)#/0.15/2.4*1.15
	#print sumterp.max()
	#print sumterp.sum()
	mapit_boundary(np.squeeze(sumterp)*1e3,[0.05,0.1,0.25,0.5,1,2,3,4,5,6],ax[0,0],False,cblabel='Annual monoterpene emission [g m$^{-2}$ yr$^{-1}$]')
	mapit_boundary(np.squeeze(sumoldterp)*1e3,[0.05,0.1,0.25,0.5,1,2,3,4,5,6],ax[0,1],False,cblabel='Annual monoterpene emission [g m$^{-2}$ yr$^{-1}$]')
	mapit_boundary(np.squeeze(sumisop)*1e3,[0.05,0.1,0.25,0.5,1,2,3,4,5,8,10,15,25,35],ax[1,0],False,cblabel='Annual isoprene emission [g m$^{-2}$ yr$^{-1}$]')
	mapit_boundary(np.squeeze(sumterp)*1e3-np.squeeze(sumoldterp)*1e3,[-5,-2,-1,-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5,1,2,5],ax[1,1],True,cblabel='Difference in monoterpene emissions NEWSOA-OLDSOA [g m$^{-2}$ yr$^{-1}$]')
	k=0
	for i in range(2):
		for j in range(2):
			ax[i,j].annotate(string.ascii_lowercase[k]+')',xy=(0.01,0.95),xycoords='axes fraction',fontsize=24)
			k+=1

	#fmapvoc.savefig(output_png_path+'/article/fig2_annual_emi_terp_isop.png',dpi=600)
	#fmapvoc.savefig(output_pdf_path+'/article/fig2_annual_emi_terp_isop.pdf',dpi=600)

	# f,aa=plt.subplots(1)
	

	# prod3d1=np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2))
	# prod3d2=np.sum(data[EXPS[0]][3][1,:,:,0:2,:,:],axis=(0,2))
	# aa.plot(prod3d1[:,75,67],'r')
	# aa.plot(prod3d2[:,75,67],'--r')
	# aa.plot(oldTM5[:,75,67],'b')
	# f,aa=plt.subplots(1)
	
	# prod3d=np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2))*gb
	# prod3dzon=prod3d[:,70:,:].sum(axis=(1,2))
	# old=oldTM5[:,:,:]*gb
	# print
	# print old.sum()*3600*24*30
	# print prod3d.sum()*3600*24*30
	# oldzon=old[:,70:,:].sum(axis=(1,2))
	# aa.plot(oldzon,'b')
	# aa.plot(prod3dzon,'r')

	# print lon[67]
	# print lat[75]
	# plt.show()
	# f,ax=plt.subplots(1)
	# ax.plot(((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb).sum(1),lat,lw=2)
	# ax.set_xlabel('Absolute difference in zonal production [Gg]')
	# ax.set_ylabel('Latitude [$^\circ$]')
	# ax.set_title('Absolute difference in production')
	# f.savefig(output_png_path+'/production/NEWSOA-OLDSOA_annual_production_SOA_zonal.png',dpi=600)

	# f,ax=plt.subplots(1)	
	# print 'Difference between emis/prod:',((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*gb*3600*24*365).sum()
	# mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e3*3600*24*365,[0.1,2,5,10,15,25,50,100],ax,False,cblabel='SOA production [gm-2]')
	# ax.set_title('Annual production of OLDSOA in [g]')
	# f.savefig(output_png_path+'/production/OLDSOA_annual_production_SOA.png',dpi=600)


	# f,ax=plt.subplots(1)
	# mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))).mean(0)*1e3*3600*24*365,[0.1,2,5,10,15,25,50,100],ax,False,cblabel='SOA production [g m-2]')
	# ax.set_title('Annual production of NEWSOA in [g]')
	# f.savefig(output_png_path+'/production/NEWSOA_annual_production_SOA.png',dpi=600)

	#f,ax=plt.subplots(2,2,figsize=(16,16))
	#gs = ax[0, 1].get_gridspec()
	fig3 = plt.figure(figsize=(16,9),tight_layout=True)
	gs = gridspec.GridSpec(2, 2,figure=fig3)
	ax1 = fig3.add_subplot(gs[0, 0])
	newdata=np.zeros((90,120))
	oldTM5data=np.zeros((90,120))
	oldTM5M2data=np.zeros((90,120))
	#print np.shape(oldTM5)
	
	for i,ndays in enumerate([31,28,31,30,31,30,31,31,30,31,30,31]):
		#print i,ndays
		#print data[EXPS[0]][3]
		#print np.shape(data[EXPS[0]][3][1,:,i,:,:,:])
		#print data[EXPS[0]][3][1,:,i,:,:,:]*3600*24*ndays
		#print np.shape(np.sum(data[EXPS[0]][3][1,:,i,:,:,:],axis=(0,1))*3600*24*ndays)
		newdata=newdata+np.sum(data[EXPS[0]][3][1,:,i,:,:,:],axis=(0,1))*3600*24*ndays*gb
		oldTM5data=oldTM5data+oldTM5[i,:,:]*3600*24*ndays*gb
		oldTM5M2data=oldTM5M2data+oldTM5M2[i,:,:]*3600*24*ndays*gb

	#mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax1,False,cblabel='Total SOA production [Gg]')
	mapit_boundary((np.squeeze(newdata)*1e-6),[0.01,0.1,1,10,20,50,75,100,150,300],ax1,False,cblabel='Total SOA production [Gg]')
	ax1.set_title('Annual production of NEWSOA',fontsize=18)
	ax1.annotate('a)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#f.savefig(output_png_path+'/production/NEWSOA_annual_production_SOA_SUM.png',dpi=600)

	#f,ax=plt.subplots(1)
	ax2 = fig3.add_subplot(gs[0, 1])
	#mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax2,False,cblabel='Total SOA production [Gg]')
	mapit_boundary((np.squeeze(oldTM5data))*1e-6,[0.01,0.1,1,10,20,50,75,100,150,300],ax2,False,cblabel='Total SOA production [Gg]')
	ax2.set_title('Annual production of OLDSOA',fontsize=18)
	ax2.annotate('b)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#f.savefig(output_png_path+'/production/OLDSOA_annual_production_SOA_SUM.png',dpi=600)

	#f,ax=plt.subplots(1)
	#same size axes for last one
	ax3 = fig3.add_subplot(gs[1, :])
	#move the axes to middle of figure
	pos = ax3.get_position()
	#pos.x0 = 0.2+pos.x0       # for example 0.2, choose your value
	#pos.x1 = 0.2+pos.x1       # for example 0.2, choose your value
	ax3.set_position(pos)
	#plot
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb,[-150,-75,-50,-25,-10,-1,1,10,25,50,75,150],ax3,True,cblabel='Absolute difference in production [Gg]')
	ax3.set_title('Absolute difference in production',fontsize=18)
	ax3.annotate('c)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#fig3.savefig(output_png_path+'/article/fig4_3panel-'+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+'-diff_annual_production_SOA_SUM.png',dpi=600)
	#fig3.savefig(output_pdf_path+'/article/fig4_3panel-'+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+'-diff_annual_production_SOA_SUM.pdf',dpi=600)
	fig4 = plt.figure(figsize=(23,11),tight_layout=True)
	
	gs = gridspec.GridSpec(2, 3,figure=fig4)
	
	ax1 = fig4.add_subplot(gs[0, 0])
	#mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax1,False,cblabel='Total SOA production [Gg]')
	mapit_boundary((np.squeeze(newdata)*1e-6),[0.01,0.1,1,10,20,50,75,100,150,300],ax1,False,cblabel='Total SOA production [Gg]')
	ax1.set_title('Annual production of NEWSOA',fontsize=18)
	ax1.annotate('a)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#f.savefig(output_png_path+'/production/NEWSOA_annual_production_SOA_SUM.png',dpi=600)

	#f,ax=plt.subplots(1)
	ax2 = fig4.add_subplot(gs[0, 1])
	#mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax2,False,cblabel='Total SOA production [Gg]')
	mapit_boundary((np.squeeze(oldTM5data))*1e-6,[0.01,0.1,1,10,20,50,75,100,150,300],ax2,False,cblabel='Total SOA production [Gg]')
	ax2.set_title('Annual production of OLDSOA',fontsize=18)
	ax2.annotate('b)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#f.savefig(output_png_path+'/production/OLDSOA_annual_production_SOA_SUM.png',dpi=600)
	ax3 = fig4.add_subplot(gs[0, 2])
	#mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax2,False,cblabel='Total SOA production [Gg]')
	mapit_boundary((np.squeeze(oldTM5M2data))*1e-6,[0.01,0.1,1,10,20,50,75,100,150,300],ax3,False,cblabel='Total SOA production [Gg]')
	ax3.set_title('Annual production of OLDSOA-MEGAN2',fontsize=18)
	ax3.annotate('c)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	#f.savefig(output_png_path+'/production/OLDSOA_annual_production_SOA_SUM.png',dpi=600)

	#f,ax=plt.subplots(1)
	#same size axes for last one
	ax4 = fig4.add_subplot(gs[1, 0])
	#move the axes to middle of figure
	pos = ax4.get_position()
	
	print pos
	pos.x0 = 0.2+pos.x0       # for example 0.2, choose your value
	pos.x1 = 0.2+pos.x1       # for example 0.2, choose your value
	print pos
	#ax4.set_position(pos)
	#plot
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb,[-150,-75,-50,-25,-10,-1,1,10,25,50,75,150],ax4,True,cblabel='Absolute difference in production [Gg]')
	ax4.set_title('Absolute difference (NEWSOA - OLDSOA) in production',fontsize=14)
	ax4.annotate('d)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	
	ax5 = fig4.add_subplot(gs[1, 1])
	#move the axes to middle of figure
	pos = ax5.get_position()
	pos.x0 = 0.2+pos.x0       # for example 0.2, choose your value
	pos.x1 = 0.2+pos.x1       # for example 0.2, choose your value
	#ax5.set_position(pos)
	#plot
	mapit_boundary((np.squeeze(oldTM5)).mean(0)*1e-6*3600*24*365*gb-np.squeeze(oldTM5M2).mean(0)*1e-6*3600*24*365*gb,[-150,-75,-50,-25,-10,-1,1,10,25,50,75,150],ax5,True,cblabel='Absolute difference in production [Gg]')
	ax5.set_title('Absolute difference (OLDSOA - OLDSOA-MEGAN2) in production',fontsize=14)
	ax5.annotate('e)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	ax6 = fig4.add_subplot(gs[1, 2])
	#move the axes to middle of figure
	pos = ax6.get_position()
	pos.x0 = 0.2+pos.x0       # for example 0.2, choose your value
	pos.x1 = 0.2+pos.x1       # for example 0.2, choose your value
	#ax6.set_position(pos)
	#plot
	mapit_boundary((np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,:,:,:],axis=(0,2)))-np.squeeze(oldTM5M2)).mean(0)*1e-6*3600*24*365*gb,[-150,-75,-50,-25,-10,-1,1,10,25,50,75,150],ax6,True,cblabel='Absolute difference in production [Gg]')
	ax6.set_title('Absolute difference (NEWSOA - OLDSOA-MEGAN2) in production',fontsize=14)
	ax6.annotate('f)',xy=(0,0.9),xycoords='axes fraction',fontsize=18)
	fig4.savefig(output_png_path+'article/fig4_revised_6panel-'+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+'-diff_annual_production_SOA_SUM.png',dpi=600)
	fig4.savefig(output_pdf_path+'article/fig4_revised_6panel-'+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+'-diff_annual_production_SOA_SUM.pdf',dpi=600)
	
	# soa from isoprene
	exp='newsoa-ri'
	pdata=prod_data_source(basepathraw+'/'+exp+'/general_TM5_'+exp+'_2010.mm.nc','soa')
	isoppi=pdata['el_o3isop']+pdata['el_ohisop']+pdata['sv_o3isop']+pdata['sv_ohisop']
	terppi=pdata['el_o3terp']+pdata['el_ohterp']+pdata['sv_o3terp']+pdata['sv_ohterp']
	print np.shape(isoppi)
	print np.shape(terppi)
	idata=np.zeros((90,120))
	tdata=np.zeros((90,120))
	for m in range(12):
		for j in range(34):
			idata+=isoppi[m,j,:,:]*3600*24*days[m]
			tdata+=terppi[m,j,:,:]*3600*24*days[m]
	#sdfasf
	#hdata=np.squeeze(np.sum(data[EXPS[0]][3][1,:,:,0,:,:],axis=0))*1e9*3600
	f,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,8))
	print ax

	mapit_boundary(np.squeeze(np.sum(isoppi[:,:,:,:],axis=(1))).mean(0)*1e-6*3600*24*365*gb,[0.01,0.1,1,10,20,50,75,100,150,300],ax[0],False,cblabel='Absolute SOA production from isoprene [Gg]')
	mapit_boundary(np.squeeze(np.sum(isoppi[:,:,:,:],axis=(1)).mean(0)*1e-6*3600*24*365*gb/(np.sum(isoppi[:,:,:,:],axis=(1)).mean(0)*1e-6*3600*24*365*gb+np.sum(terppi[:,:,:,:],axis=(1)).mean(0)*1e-6*3600*24*365*gb))*100,[10,20,30,40,50,60,70,80,90],ax[1],False,cblabel='Fraction of SOA from isopreen [%]')
	
	plt.show()

