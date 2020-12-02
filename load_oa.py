import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import sys
#sys.path.append("/Users/bergmant/Documents/Project/ifs+tm5-validation/scripts")
#from colocate_aeronet import do_colocate
#from lonlat import lonlat
#from read_mass import read_mass
#from emep_read import read_mass
from settings import *
#from tm5_tools import get_gridboxarea
from general_toolbox import read_var,get_gridboxarea,lonlat,read_mass
#from plot_m7 import read_var

# def read_soa(infile,comp):
# 	#comp='SOA'
# 	modes=['NUS','AIS','ACS','COS','AII']
# 	data={}
# 	for i in modes:

# 		var='M_'+comp+i
# 		if var == 'M_POMNUS':
# 			continue 
# 		vardata=read_var(infile,var)
# 		#print vardata[0]
# 		data[i]=vardata[0]
# 	outdata=np.zeros_like(data['AIS'][:])
# 	print np.shape(outdata)
# 	for i in modes:
# 		if i in data:
# 			print i
# 			outdata+=data[i]		
# 			print outdata[0,0,0,0]
# 		else:		
# 			continue
# 	return outdata

def seasonal_plot(load,newname,oldname,fractional=True,bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0.10,0.25,0.5,1,1.5,2,2.5,3],oldtitle='old',newtitle='new',axit=None):
	if axit==None:
		f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
	else:
		ax2=axit
	seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
	seas=['DJF','MAM','JJA','SON']
	jj=-1
	# if not fractional:
	# 	newmax=np.max(npload[newname])
	# 	newmin=np.min(load[newname])
	# 	oldmin=np.min(load[oldname])
	# 	oldmax=np.max(load[oldname])
	# 	mindata=min(oldmin,newmin)
	# 	maxdata=max(oldmax,newmax)
	# 	limit=max(abs(mindata),maxdata)
	# 	print limit,mindata,maxdata,oldmin,oldmax,newmin,newmax
	# 	bounds_load=np.linspace(-limit,limit,13)
	# 	print bounds_load
	#bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3]
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
	
	for i in range(4):
		ii=i
		kk=0
		if ii>1:
			ii=ii-2
			kk=1
			#jj+=1
		m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
		#data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
		#data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
		if fractional:
			ax2[kk,ii].set_title('{}'.format(seas[i]))
			image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][seasonidx[i],:,:],0)-np.mean(load[oldname][seasonidx[i],:,:],0))/np.mean(load[oldname][seasonidx[i],:,:],0),norm=norm,cmap=mycmap,latlon=True)
			cb = m.colorbar(image,"bottom", size="5%", pad="2%",extend='both')
			cb.set_label('('+newtitle+'-'+oldtitle+')/'+oldtitle+' [%]]')
		else:
			ax2[kk,ii].set_title('{}'.format(seas[i]))
			image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][seasonidx[i],:,:],0)-np.mean(load[oldname][seasonidx[i],:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
			cb = m.colorbar(image,"bottom", size="5%", pad="2%")
			cb.set_label('[('+newtitle+'-'+oldtitle+')]')
		
		#image=m.pcolormesh(lons,lats,np.squeeze((datanew-dataold)/dataold),norm=norm,cmap=mycmap,latlon=True)
		m.drawparallels(np.arange(-90.,90.,30.))
		m.drawmeridians(np.arange(-180.,180.,60.))
		m.drawcoastlines()
	#f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_{}-{}.png'.format(newname,oldname),dpi=600)
	return f2,ax2#,image

def annual_diff_plot(load,newname,oldname,fractional=True,bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3],ax2=None,label_prefix=''):
	from matplotlib.colors import LinearSegmentedColormap 
	if ax2==None:
		f2,ax2=plt.subplots(nrows=1,ncols=1,figsize=(12,8))
	# if not fractional:
	# 	newmax=np.max(npload[newname])
	# 	newmin=np.min(load[newname])
	# 	oldmin=np.min(load[oldname])
	# 	oldmax=np.max(load[oldname])
	# 	mindata=min(oldmin,newmin)
	# 	maxdata=max(oldmax,newmax)
	# 	limit=max(abs(mindata),maxdata)
	# 	print limit,mindata,maxdata,oldmin,oldmax,newmin,newmax
	# 	bounds_load=np.linspace(-limit,limit,13)
	# 	print bounds_load
	#bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3]
	# change to percentage
	bounds_pct=[element * 100 for element in bounds_load]
	#print bounds_pct
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
	#mycmap=LinearSegmentedColormap.from_list(name='mycmap', 
    #                                         colors =['b', '0.9', 'r'],
    #                                         N=len(bounds_load))
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	#print len(bounds_pct)
	normpct = mpl.colors.BoundaryNorm(bounds_pct, len(bounds_pct)-1)
	m=Basemap(projection='robin',lon_0=0,ax=ax2)
	#data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	if fractional:
		#ax2.set_title('Annual mean fractional change of OA burden.')
		image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][:,:,:],0)-np.mean(load[oldname][:,:,:],0))/np.mean(load[oldname][:,:,:],0)*100.0,norm=normpct,cmap=mycmap,latlon=True)
		cb = m.colorbar(image,"bottom", size="5%", pad="2%",extend='both',ticks=bounds_pct)
		cb.set_label(label_prefix+'('+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+')/'+EXP_NAMEs[1]+' [%]')
		#cb.set_ticks()
	else:
		#ax2.set_title('Annual mean absolute change of OA burden')
		image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][:,:,:],0)-np.mean(load[oldname][:,:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
		cb = m.colorbar(image,"bottom", size="5%", pad="2%")
		cb.set_label(label_prefix+'('+EXP_NAMEs[0]+'-'+EXP_NAMEs[1]+') [ugm-2]]')
	#cb.cmap.set_over('g')
	#cb.cmap.set_under('k')

	#image=m.pcolormesh(lons,lats,np.squeeze((datanew-dataold)/dataold),norm=norm,cmap=mycmap,latlon=True)
	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
#f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_{}-{}.png'.format(newname,oldname),dpi=600)
	return #f2,ax2,image
def annual_mean_map(load,newname,bounds_load=[0,0.10,0.25,0.5,1,1.5,2,2.5,3]):
	f2,ax2=plt.subplots(nrows=1,ncols=1,figsize=(12,8))
	# if not fractional:
	# 	newmax=np.max(npload[newname])
	# 	newmin=np.min(load[newname])
	# 	oldmin=np.min(load[oldname])
	# 	oldmax=np.max(load[oldname])
	# 	mindata=min(oldmin,newmin)
	# 	maxdata=max(oldmax,newmax)
	# 	limit=max(abs(mindata),maxdata)
	# 	print limit,mindata,maxdata,oldmin,oldmax,newmin,newmax
	# 	bounds_load=np.linspace(-limit,limit,13)
	# 	print bounds_load
	#bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3]
	mycmap=plt.get_cmap('Greens',len(bounds_load)) 
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
	
	m=Basemap(projection='robin',lon_0=0,ax=ax2)
	#data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	#ax2.set_title('Annual mean absolute change of OA loading.')
	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][:,:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
	cb = m.colorbar(image,"bottom", size="5%", pad="2%")
	cb.set_label('[('+newname+')]')
		
	#image=m.pcolormesh(lons,lats,np.squeeze((datanew-dataold)/dataold),norm=norm,cmap=mycmap,latlon=True)
	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
#f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_{}-{}.png'.format(newname,oldname),dpi=600)
	return f2,ax2,image


#output_png_path='/Users/bergmant/Documents/tm5-soa/figures/draft-v0.9/png/'

#EXPS=['newsoa-ri','oldsoa-bhn','nosoa']
#EXPSname=['NEWSOA','OLDSOA','NOSOA']
#EXPS=['newsoa-ri','oldsoa-bhn']
#EXPSname=['NEWSOA','OLDSOA']

load={}
loadpoa={}
loadsoa={}
inpath='/Users/bergmant/Documents/tm5-soa/output/'
mass={}
surface_mass_concentration={}
mass={}
#mass['soa-riccobono']={}
mass[EXPS[0]]={}
mass[EXPS[1]]={}
surface_mass_concentration[EXPS[0]]={}
surface_mass_concentration[EXPS[1]]={}
mass[EXPS[0]]={}
mass[EXPS[1]]={}
#mass[EXPS[2]]={}
for exp in EXPS:
	#loadoa : Primary OA
	#loadsoa: secondary OA
	data_poa=nc.Dataset(inpath+'general_TM5_'+exp+'_2010.mm.nc','r').variables['loadoa'][:]
	data_soa=nc.Dataset(inpath+'general_TM5_'+exp+'_2010.mm.nc','r').variables['loadsoa'][:]
	load[exp]=np.squeeze(data_poa+data_soa)
	loadsoa[exp]=np.squeeze(data_soa)
	loadpoa[exp]=np.squeeze(data_poa)
	print exp
	data_soa=read_mass('M_SOA',inpath+'general_TM5_'+exp+'_2010.mm.nc')
	mass[exp]['SOA']=data_soa
	data_poa=read_mass('M_POM',inpath+'general_TM5_'+exp+'_2010.mm.nc')
	mass[exp]['POM']=data_poa
	
	# data_soa=read_soa(inpath+'general_TM5_'+exp+'_2010.mm.nc','SOA')
	# mass[exp]['SOA']=data_soa
	# data_poa=read_soa(inpath+'general_TM5_'+exp+'_2010.mm.nc','POM')
	# mass[exp]['POM']=data_poa
	#print mass[exp]['SOA'][0,0,0,0]
	#print mass[exp]['SOA'][0,0,0,0]
	#print mass[exp]['POM'][:,0,0,0]/mass[exp]['POM'][:,0,0,0]

	raw_input()


	surface_soa=read_mass('sconcsoa',inpath+'general_TM5_'+exp+'_2010.mm.nc')
	surface_mass_concentration[exp]['SOA']=surface_soa
	surface_poa=read_mass('sconcoa',inpath+'general_TM5_'+exp+'_2010.mm.nc')
	surface_mass_concentration[exp]['POM']=surface_poa
raw_input()
# f,ax=plt.subplots(ncols=1,figsize=(8,4))
# k=-1
# #print np.shape(load['soa-riccobono'])
lon,lat=lonlat('TM53x2')
lons, lats = np.meshgrid(lon,lat)
gb=get_gridboxarea('TM53x2')
gph=nc.Dataset(inpath+'general_TM5_'+exp+'_2010.mm.nc','r').variables['gph3D'][:]
# #for exp in EXPS:
# k+=1
# bounds_load=[-5,-4,-3,-2,-1,0,1,2,3,4,5]

# ax.set_title('diff')
# m=Basemap(projection='robin',lon_0=0,ax=ax)

# #image=m.contourf(lons,lats,(np.mean(load['soa-riccobono'],0)-np.mean(load[EXPS[1]],0))/np.mean(load[EXPS[1]],0),bounds_load,cmap=plt.cm.RdBu_r,latlon=True)
# image=m.contourf(lons,lats,(np.mean(load[EXPS[0]],0)-np.mean(load[EXPS[1]],0))/np.mean(load[EXPS[1]],0),bounds_load,cmap=plt.cm.RdBu_r,latlon=True)
# m.drawparallels(np.arange(-90.,90.,30.))
# m.drawmeridians(np.arange(-180.,180.,60.))
# m.drawcoastlines()
# cb = m.colorbar(image,"bottom", size="5%", pad="2%")
# print 'so',np.shape(mass[EXPS[0]]['SOA'])

# bounds=[-3,-2,-1,0,1,2,3]
# f2,ax2=plt.subplots(nrows=3,ncols=4,figsize=(8,4))
# print ax2.shape
# print np.shape(ax2)
# jj=-1
# for i in range(3):
# 	for k in range(4):
# 		jj+=1
# 		#for exp in EXPS:
# 		print np.shape(np.squeeze(load[EXPS[0]]-load[EXPS[1]])/np.mean(load[EXPS[1]]))
# 		ax2[i,k].set_title('Fractional change month={:d} '.format(jj+1))
# 		m=Basemap(projection='robin',lon_0=0,ax=ax2[i,k])
# 		data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][:,0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][:,0,:,:],0)
# 		print np.shape(data_new)
# 		data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][:,0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][:,0,:,:],0)
# 		image=m.contourf(lons,lats,data_new-data_old,bounds,cmap=plt.cm.RdBu_r,latlon=True)
# 		image=m.contourf(lons,lats,np.squeeze(load[EXPS[0]][jj,:,:]-load[EXPS[1]][jj,:,:])/np.mean(load[EXPS[1]][jj,:,:]),bounds_load,cmap=plt.cm.RdBu_r,latlon=True)
# 		m.drawparallels(np.arange(-90.,90.,30.))
# 		m.drawmeridians(np.arange(-180.,180.,60.))
# 		m.drawcoastlines()
# 		cb = m.colorbar(image,"bottom", size="5%", pad="2%")
# 		cb.set_label('[(new-old)/old]')


# f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
# print ax2.shape
# print np.shape(ax2)
# seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
# seas=['DJF','MAM','JJA','SON']
# jj=-1
# bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3]

# for i in range(4):
# 	ii=i
# 	kk=0
# 	if ii>1:
# 		ii=ii-2
# 		kk=1
# 		#jj+=1
# 	#for exp in EXPS:
# 	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
# 	#mycmap=plt.cm.RdBu_r
# 	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
# 	ax2[kk,ii].set_title('Fractional change of load OA in {}'.format(seas[i]))
# 	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
# 	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
# 	print np.shape(data_new)
# 	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
# 	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load[EXPS[1]][seasonidx[i],:,:],0))/np.mean(load[EXPS[1]][seasonidx[i],:,:],0),norm=norm,cmap=mycmap,latlon=True)
# 	m.drawparallels(np.arange(-90.,90.,30.))
# 	m.drawmeridians(np.arange(-180.,180.,60.))
# 	m.drawcoastlines()
# 	cb = m.colorbar(image,"bottom", size="5%", pad="2%")
# 	cb.set_label('[(NEWSOA-OLDSOA)/OLDSOA]')
# f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_NEWSOA-OLDSOA.png',dpi=600)

#f3,ax3=plt.subplots(ncols=2,nrows=2)
f3,ax3=seasonal_plot(load,EXPS[0],EXPS[1],True,[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.75,1,2,3],EXP_NAMEs[0],EXP_NAMEs[1])
#print output_png_path+'/ORGANICMASS/seasonal_map_frac_load_total_oa_NEWSOA-OLDSOA.png'
f3.savefig(output_png_path+'/ORGANICMASS/figS8_seasonal_map_frac_load_total_oa_NEWSOA-OLDSOA.png',dpi=600)
plt.show()
#f3,ax3=plt.subplots(1)
f3,ax3=seasonal_plot(load,EXPS[0],EXPS[1],False,[-13,-11,-9,-7,-5,-3,-1,-0.1,0.1,1,3,5,7,9,11,13],EXP_NAMEs[0],EXP_NAMEs[1])
f3.savefig(output_png_path+'/ORGANICMASS/seasonal_map_abs_load_total_oa_NEWSOA-OLDSOA.png',dpi=600)

f3,ax3=plt.subplots(ncols=2, figsize=(12,6))
annual_diff_plot(load,EXPs[0],EXPs[1],True,[-1.5,-1,-0.75,-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5,0.75,1,1.5],ax3[0],'Burden of total OA ')
#ax3.set_title('Annual mean fractional change in burden of total OA.')
m=Basemap(projection='robin',lon_0=0,ax=ax3[1])
bounds_load=[-100,-75,-50,-25,-10,-5,5,10,25,50,75,100]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold)/dataold)*100,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3[1])
cb.set_label('Change in OA mass on the surface (NEWSOA-OLDSOA)/OLDSOA[%]]')
#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
ax3[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)


f3.savefig(output_png_path+'/ORGANICMASS/fig12_annual_2-panel-frac-load-conc-map_total_oa_NEWSOA-OLDSOA.png',dpi=600)

f3,ax3=plt.subplots(ncols=2, figsize=(12,6))
annual_diff_plot(load,EXPs[0],EXPs[1],True,[-1.5,-1,-0.75,-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5,0.75,1,1.5],ax3[0],'Burden of total OA ')
#ax3.set_title('Annual mean fractional change in burden of total OA.')
m=Basemap(projection='robin',lon_0=0,ax=ax3[1])
bounds_load=[-100,-75,-50,-25,-10,-5,5,10,25,50,75,100]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold)/dataold)*100,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3[1])
cb.set_label('Change in OA mass on the surface (NEWSOA-OLDSOA)/OLDSOA[%]]')
#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
ax3[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)


#f3.savefig(output_png_path+'/ORGANICMASS/fig12_annual_2-panel-frac-load-conc-map_total_oa_NEWSOA-OLDSOA.png',dpi=600)


#for i in range(34):
f3,ax3=plt.subplots(ncols=2, figsize=(12,6))
annual_diff_plot(load,EXPs[0],EXPs[1],False,[-11,-9,-7,-5,-3,-1,-0.1,0.1,1,3,5,7,9,11],ax3[0],'Burden of total OA ')
#ax3.set_title('Annual mean absolute change in burden of total OA.')
m=Basemap(projection='robin',lon_0=0,ax=ax3[1])
bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,-0.05,-0.01,-0.001,0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
#print np.shape(mass[EXPS[0]]['SOA'][:,:,:,:])
#print EXPS[0],EXPS[1]
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3[1])
cb.set_label('Change in OA mass on the surface (NEWSOA-OLDSOA)[ugm-3]]')
#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
ax3[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)


f3.savefig(output_png_path+'/ORGANICMASS/annual_2-panel-abs-load-conc-map_total_oa_NEWSOA-OLDSOA.png',dpi=600)
f3,ax3=plt.subplots(1)
m=Basemap(projection='robin',lon_0=0,ax=ax3)
bounds_load=[-1000,-3,-2,-1,-0.5,-0.25,-0.1,-0.05,-0.01,-0.001,0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3,1000]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
#print np.shape(mass[EXPS[0]]['SOA'][:,:,:,:])
#print EXPS[0],EXPS[1]
datasoa=(mass[EXPS[0]]['SOA'][:,:,:,:]*gph)[:,0,:,:].mean(0)
dataold=(mass[EXPS[1]]['SOA'][:,:,:,:]*gph)[:,0,:,:].mean(0)
temp=0
for l in range(34):
	temp+=(mass[EXPS[0]]['SOA'][:,l,:,:]*gph[:,l,:,:]*gb).mean(0).sum()
#print temp
temp=0
for l in range(34):
	temp+=(mass[EXPS[1]]['SOA'][:,l,:,:]*gph[:,l,:,:]*gb).mean(0).sum()
#print temp
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold))*1e6,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3)
cb.set_label('[asdfasdfs(NEWSOA-OLDSOA)/OLDSOA]')

f3,ax3=plt.subplots(1)
m=Basemap(projection='robin',lon_0=0,ax=ax3)
bounds_load=[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.25,0.5,0.75,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa2=(surface_mass_concentration[EXPS[0]]['SOA'][:,:,:]+surface_mass_concentration[EXPS[0]]['POM'][:,:,:]).mean(0)
dataold2=(surface_mass_concentration[EXPS[1]]['SOA'][:,:,:]+surface_mass_concentration[EXPS[1]]['POM'][:,:,:]).mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa2-dataold2))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3)
cb.set_label('[sconc(NEWSOA-OLDSOA)/OLDSOA]')
f3,ax3=plt.subplots(1)
m=Basemap(projection='robin',lon_0=0,ax=ax3)
bounds_load=[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.25,0.5,0.75,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa3=(mass[EXPS[0]]['SOA'][:,0,:,:]+mass[EXPS[0]]['POM'][:,0,:,:]).mean(0)
dataold3=(mass[EXPS[1]]['SOA'][:,0,:,:]+mass[EXPS[1]]['POM'][:,0,:,:]).mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa3-dataold3))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3)
cb.set_label('[newread(NEWSOA-OLDSOA)/OLDSOA]')
f3,ax3=plt.subplots(1)

m=Basemap(projection='robin',lon_0=0,ax=ax3)
bounds_load=[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.25,0.5,0.75,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
image=m.pcolormesh(lons,lats,np.squeeze((datasoa2-dataold2)/dataold2),norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3)
cb.set_label('[sconc(NEWSOA-OLDSOA)/OLDSOA]')
f3,ax3=plt.subplots(1)

m=Basemap(projection='robin',lon_0=0,ax=ax3)
bounds_load=[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.25,0.5,0.75,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
image=m.pcolormesh(lons,lats,np.squeeze((dataold-dataold2))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3,extend='both')
#cb.set_under('k')
#cb.set_under('y')
cb.set_label('[sconc(NEWSOA-OLDSOA)/OLDSOA]')


#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
#ax3[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
f3,ax3=plt.subplots(1)
f3,ax3=seasonal_plot(loadsoa,EXPS[0],EXPS[1],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,1.5,2],EXP_NAMEs[0],EXP_NAMEs[1])
#ax3.set_title('Seasonal mean fractional change in burden of SOA.')
f3.savefig(output_png_path+'/ORGANICMASS/seasonal_map_frac_load_soa_NEWSOA-OLDSOA.png',dpi=600)


f3,ax3=plt.subplots(1)
annual_diff_plot(loadsoa,EXPS[0],EXPS[1],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,1.5,2],ax3,'Burden of SOA ')
#ax3.set_title('Annual mean fractional change in burden of SOA.')
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_frac_load_soa_NEWSOA-OLDSOA.png',dpi=600)

f3,ax3=plt.subplots(1)
annual_diff_plot(loadsoa,EXPS[0],EXPS[1],False,[-13,-11,-9,-7,-5,-3,-1,-0.1,0.1,1,3,5,7,9,11,13],ax3,'Burden of SOA ')
#ax3.set_title('Annual mean absolute change in burden of SOA.')
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_abs_load_soa_NEWSOA-OLDSOA.png',dpi=600)
f3,ax3=plt.subplots(1)

f3,ax3=plt.subplots(1)
annual_diff_plot(loadpoa,EXPS[0],EXPS[1],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,1.5,2],ax3,'Burden of POA ')
ax3.set_title('Annual mean fractional change in burden of POA.')
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_frac_load_poa_NEWSOA-OLDSOA.png',dpi=600)

f3,ax3=plt.subplots(1)
annual_diff_plot(loadpoa,EXPS[0],EXPS[1],False,[-1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1],ax3,'Burden of POA ')
#ax3.set_title('Annual mean absolute change in burden of POA.')
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_abs_load_poa_NEWSOA-OLDSOA.png',dpi=600)


f3,ax3,image3=annual_mean_map(load,EXPS[0],[0,1,5,7.5,10,12.5,15,20])
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_load_total_oa_NEWSOA.png',dpi=600)

f3,ax3,image3=annual_mean_map(load,EXPS[1],[0,1,5,7.5,10,12.5,15,20])
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_load_total_oa_OLDSOA.png',dpi=600)

f3,ax3,image3=annual_mean_map(loadsoa,EXPS[0],[0,1,5,7.5,10,12.5,15,20])
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_load_soa_NEWSOA.png',dpi=600)

f3,ax3,image3=annual_mean_map(loadsoa,EXPS[1],[0,1,5,7.5,10,12.5,15,20])
f3.savefig(output_png_path+'/ORGANICMASS/annual_map_load_soa_OLDSOA.png',dpi=600)

f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
#print ax2.shape
#print np.shape(ax2)
seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
seas=['DJF','MAM','JJA','SON']
jj=-1
bounds_load=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for i in range(4):
	ii=i
	kk=0
	if ii>1:
		ii=ii-2
		kk=1
		#jj+=1
	#for exp in EXPS:
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	#mycmap=plt.cm.RdBu_r
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
	ax2[kk,ii].set_title('Fractional change of burden OA in {}'.format(seas[i]))
	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#print np.shape(data_new)
	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	cb = m.colorbar(image,"bottom", size="5%", pad="2%")
	cb.set_label('[NEWSOA]')
f2.savefig(output_png_path+'/ORGANICMASS/map_load_NEWSOA.png',dpi=600)
f2,ax2=seasonal_plot(load,EXPS[0],EXPS[1],False,[-13,-11,-9,-7,-5,-3,-1,-0.1,0.1,1,3,5,7,9,11,13],EXP_NAMEs[0],EXP_NAMEs[1])
f2.savefig(output_png_path+'/ORGANICMASS/map_load_NEWSOA.png',dpi=600)


# f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
# seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
# seas=['DJF','MAM','JJA','SON']
# jj=-1
# bounds_load=[-10,-5,-3,-2,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,2,3,5,10]

# for i in range(4):
# 	ii=i
# 	kk=0
# 	if ii>1:
# 		ii=ii-2
# 		kk=1
# 		#jj+=1
# 	#for exp in EXPS:
# 	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
# 	#mycmap=plt.cm.RdBu_r
# 	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
# 	ax2[kk,ii].set_title('Fractional change in {}'.format(seas[i]))
# 	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
# 	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
# 	print np.shape(data_new)
# 	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
# 	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load['nosoa'][seasonidx[i],:,:],0))/np.mean(load['nosoa'][seasonidx[i],:,:],0),norm=norm,cmap=mycmap,latlon=True)
# 	m.drawparallels(np.arange(-90.,90.,30.))
# 	m.drawmeridians(np.arange(-180.,180.,60.))
# 	m.drawcoastlines()
# 	cb = m.colorbar(image,"bottom", size="5%", pad="2%")
# 	cb.set_label('[(NEWSOA-NOSOA)/NOSOA]')
# f2.savefig(output_png_path+'/ORGANICMASS/map_frac_load_NEWSOA-NOSOA.png',dpi=600)


f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
#print ax2.shape
#print np.shape(ax2)
seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
seas=['DJF','MAM','JJA','SON']
jj=-1
bounds_load=[-13,-11,-9,-7,-5,-3,-1,-0.5,0.5,1,3,5,7,9,11,13]

for i in range(4):
	ii=i
	kk=0
	if ii>1:
		ii=ii-2
		kk=1
		#jj+=1
	#for exp in EXPS:
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
	#mycmap=plt.cm.RdBu_r
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
	ax2[kk,ii].set_title('{}'.format(seas[i]))
	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#print np.shape(data_new)
	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load[EXPS[1]][seasonidx[i],:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	cb = m.colorbar(image,"bottom", size="5%", pad="2%",ticks=bounds_load)
	cb.set_label('NEWSOA-OLDSOA [mg m-2]]')
	#print np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load[EXPS[1]][seasonidx[i],:,:],0)).max()*1e6
f2.savefig(output_png_path+'/ORGANICMASS/map_absolute_load_NEWSOA-OLDSOA.png',dpi=600)

f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
#print ax2.shape
#print np.shape(ax2)
seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
seas=['DJF','MAM','JJA','SON']
jj=-1
bounds_load=[-13,-11,-9,-7,-5,-3,-1,-0.5,0.5,1,3,5,7,9,11,13]

for i in range(4):
	ii=i
	kk=0
	if ii>1:
		ii=ii-2
		kk=1
		#jj+=1
	#for exp in EXPS:
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
	#mycmap=plt.cm.RdBu_r
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
	#print mycmap
	ax2[kk,ii].set_title('{}'.format(seas[i]))
	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#print np.shape(data_new)
	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	image=m.pcolormesh(lons,lats,np.ma.masked_inside(np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load[EXPS[1]][seasonidx[i],:,:],0))*1e6,-0.5,0.5),norm=norm,cmap=mycmap,latlon=True)
	m.drawparallels(np.arange(-90.,90.,30.))
	m.drawmeridians(np.arange(-180.,180.,60.))
	m.drawcoastlines()
	cb = m.colorbar(image,"bottom", size="5%", pad="2%",ticks=bounds_load)
	cb.set_label('NEWSOA-OLDSOA [mg m-2]]')
	#print np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load[EXPS[1]][seasonidx[i],:,:],0)).max()*1e6
f2.savefig(output_png_path+'/ORGANICMASS/map_absolute_load_NEWSOA-OLDSOA.png',dpi=600)



# f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
# print ax2.shape
# print np.shape(ax2)
# seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
# seas=['DJF','MAM','JJA','SON']
# jj=-1
# bounds_load=[-13,-11,-9,-7,-5,-3,-1,-0.5,0.5,1,3,5,7,9,11,13]

# for i in range(4):
# 	ii=i
# 	kk=0
# 	if ii>1:
# 		ii=ii-2
# 		kk=1
# 		#jj+=1
# 	#for exp in EXPS:
# 	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load))
# 	#mycmap=plt.cm.RdBu_r
# 	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)) 
# 	print np.shape(np.squeeze(load[EXPS[0]]-load[EXPS[1]])/np.mean(load[EXPS[1]]))
# 	ax2[kk,ii].set_title('Absolute change in {}'.format(seas[i]))
# 	m=Basemap(projection='robin',lon_0=0,ax=ax2[kk,ii])
# 	data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
# 	print np.shape(data_new)
# 	data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
# 	image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[EXPS[0]][seasonidx[i],:,:],0)-np.mean(load['nosoa'][seasonidx[i],:,:],0))*1e6,norm=norm,cmap=mycmap,latlon=True)
# 	m.drawparallels(np.arange(-90.,90.,30.))
# 	m.drawmeridians(np.arange(-180.,180.,60.))
# 	m.drawcoastlines()
# 	cb = m.colorbar(image,"bottom", size="5%", pad="2%",ticks=bounds_load	)
# 	cb.set_label('NEWSOA-NOSOA [mg m-2]]')
# f2.savefig(output_png_path+'/ORGANICMASS/map_absolute_load_NEWSOA-NOSOA.png',dpi=600)



# ff2,ax=plt.subplots(1,figsize=(10,6))
# m=Basemap(projection='robin',lon_0=0,ax=ax)
# bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,2,3]
# norm = mpl.colors.BoundaryNorm(bounds_load, 11)
# mycmap=plt.get_cmap('RdBu_r',11) 
# image=m.pcolormesh(lons,lats,np.squeeze(load[EXPS[0]][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),norm=norm,cmap=mycmap,latlon=True)
# #image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
# m.drawparallels(np.arange(-90.,90.,30.))
# m.drawmeridians(np.arange(-180.,180.,60.))
# m.drawcoastlines()
# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
# cb.set_label('[(NEWSOA-OLDSOA)/OLDSOA]')
# ax.set_title('Fractional change in annual mean organic aerosol loading \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
# ff2.savefig(output_png_path+'/ORGANICMASS/fractional_oa_NEWSOA-OLDSOA_OLDOSOA.png')

# ff2,ax=plt.subplots(1,figsize=(10,6))
# m=Basemap(projection='robin',lon_0=0,ax=ax)
# bounds_load=[-1.3,-1.1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1,1.3]
# norm = mpl.colors.BoundaryNorm(bounds_load, 13)
# mycmap=plt.get_cmap('RdBu_r',13) 
# image=m.pcolormesh(lons,lats,np.squeeze(load[EXPS[0]][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),norm=norm,cmap=mycmap,latlon=True)
# #image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))*1e6,cmap=plt.cm.RdBu_r,latlon=True)
# m.drawparallels(np.arange(-90.,90.,30.))
# m.drawmeridians(np.arange(-180.,180.,60.))
# m.drawcoastlines()
# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
# cb.set_label('(NEWSOA-OLDSOA) [mg m-2]')
# ax.set_title('Change in organic aerosol loading \n NEWSOA-OLDSOA',fontsize=18)
# ff2.savefig(output_png_path+'/ORGANICMASS/oa_new-old.png')

ff2,ax=plt.subplots(1,figsize=(10,6))
m=Basemap(projection='robin',lon_0=0,ax=ax)
bounds_load=[-300,-200,-100,-50,-25,-10,-5,5,10,25,50,100,200,300]
norm = mpl.colors.BoundaryNorm(bounds_load, 11)
mycmap=plt.get_cmap('RdBu_r',11) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold)/dataold)*100,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
cb.set_label('[(NEWSOA-OLDSOA)/OLDSOA]')
ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
ff2.savefig(output_png_path+'/ORGANICMASS/fractional_oa_new-old.png')
plt.show()


ff2,ax=plt.subplots(1,figsize=(10,6))
m=Basemap(projection='robin',lon_0=0,ax=ax)
bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,-0.05,-0.01,-0.001,0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
#datasoa=mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
#dataold=mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
#print np.max(datasoa)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
cb.set_label('[(NEWSOA-OLDSOA)]')
#ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
ff2.savefig(output_png_path+'/ORGANICMASS/absolute_oa_new-old.png')

ff2,ax=plt.subplots(1,figsize=(10,6))
m=Basemap(projection='robin',lon_0=0,ax=ax)
bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,-0.05,-0.01,-0.001,0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
mycmap=plt.get_cmap('Greens',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
#print np.max(datasoa)
image=m.pcolormesh(lons,lats,np.squeeze((dataold))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
cb.set_label('[(OLDSOA)]')
#ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
ff2.savefig(output_png_path+'/ORGANICMASS/absolute_oa_old.png')
ff2,ax=plt.subplots(1,figsize=(10,6))
m=Basemap(projection='robin',lon_0=0,ax=ax)
bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,-0.05,-0.01,-0.001,0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
mycmap=plt.get_cmap('Greens',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[1]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[1]]['POM'][:,0,:,:].mean(0)
#print np.max(datasoa)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa))*1e9,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
cb.set_label('[(NEWSOA)]')
#ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
ff2.savefig(output_png_path+'/ORGANICMASS/absolute_oa_new.png')

# ff2,ax=plt.subplots(1,figsize=(10,6))
# m=Basemap(projection='robin',lon_0=0,ax=ax)
# bounds_load=[-3,-2,-1,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,2,3]
# norm = mpl.colors.BoundaryNorm(bounds_load, 11)
# mycmap=plt.get_cmap('RdBu_r',11) 
# datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
# dataold=mass['nosoa']['POM'][:,0,:,:].mean(0)
# image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold)/dataold),norm=norm,cmap=mycmap,latlon=True)
# #image=m.pcolormesh(lons,lats,np.squeeze(load[EXPS[0]][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
# m.drawparallels(np.arange(-90.,90.,30.))
# m.drawmeridians(np.arange(-180.,180.,60.))
# m.drawcoastlines()
# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
# cb.set_label('[(NEWSOA-NOSOA)/NOSOA]')
# ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-NOSOA)/NOSOA',fontsize=18)
# ff2.savefig(output_png_path+'/ORGANICMASS/fractional_oa_new-no.png')



# data_new=np.mean(np.mean(mass[EXPS[0]]['SOA'][:,:,:,:],3),0)+np.mean(np.mean(mass[EXPS[0]]['POM'][:,:,:,:],3),0)
# print np.shape(data_new)
# data_old=np.mean(np.mean(mass[EXPS[1]]['SOA'][:,:,:,:],3),0)+np.mean(np.mean(mass[EXPS[1]]['POM'][:,:,:,:],3),0)
# print data_new.shape
# # data_no=np.mean(np.mean(mass['nosoa']['SOA'][:,:,:,:],3),0)+np.mean(np.mean(mass['nosoa']['POM'][:,:,:,:],3),0)

# f,ax=plt.subplots(1)
# bounds=[-10,-5,-2,-1.5,-1,-0.5,-0.1,0.1,0.5,1,1.5,2,5,10]
# mycmap=plt.get_cmap('RdBu_r',len(bounds))
# normi=mpl.colors.BoundaryNorm(bounds,mycmap.N)
# im=ax.pcolormesh(lat,np.arange(34),(data_new-data_old)/data_old,norm=normi,cmap=mycmap)
# cb=plt.colorbar(im,ticks=bounds)
# f.savefig(output_png_path+'/masszonal/zonal_fractional_new-old.png')



# f,ax=plt.subplots(1)
# bounds=[-0.10,-0.05,-0.02,-0.015,-0.01,-0.005,0.005,0.01,0.015,0.02,0.05,0.10]
# mycmap=plt.get_cmap('RdBu_r',len(bounds))
# normi=mpl.colors.BoundaryNorm(bounds,mycmap.N)
# im=ax.pcolormesh(lat,np.arange(34),(data_new-data_old)*1e9,norm=normi,cmap=mycmap)
# cb=plt.colorbar(im,ticks=bounds)
# f.savefig(output_png_path+'/masszonal/zonal_new-oldsoa.png')

# f,ax=plt.subplots(1)
# bounds=[0,0.01,0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.5,1]
# mycmap=plt.get_cmap('Blues',len(bounds))
# normi=mpl.colors.BoundaryNorm(bounds,mycmap.N)
# im=ax.pcolormesh(lat,np.arange(34),data_new*1e9,norm=normi,cmap=mycmap)
# cb=plt.colorbar(im,ticks=bounds)
# f.savefig(output_png_path+'/masszonal/zonal_oa_new.png')



# f,ax=plt.subplots(1)
# bounds=[0,0.01,0.05,0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.5,1]
# mycmap=plt.get_cmap('Blues',len(bounds))
# normi=mpl.colors.BoundaryNorm(bounds,mycmap.N)
# im=ax.pcolormesh(lat,np.arange(34),data_old*1e9,norm=normi,cmap=mycmap)
# cb=plt.colorbar(im,ticks=bounds)
# # f.savefig(output_png_path+'/masszonal/zonal_oa_old.png')

plt.show()
