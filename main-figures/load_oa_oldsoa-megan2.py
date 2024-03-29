import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import sys

from settings import *
from general_toolbox import read_var,get_gridboxarea,lonlat,read_mass



def seasonal_plot(load,newname,oldname,fractional=True,bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0.10,0.25,0.5,1,1.5,2,2.5,3],oldtitle='old',newtitle='new',axit=None):
	if axit==None:
		f2,ax2=plt.subplots(nrows=2,ncols=2,figsize=(12,8))
	else:
		ax2=axit
	seasonidx=[[11,0,1],[2,3,4],[5,6,7],[8,9,0]]
	seas=['DJF','MAM','JJA','SON']
	jj=-1
	
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

def annual_diff_plot(load,newname,oldname,fractional=True,bounds_load=[-3,-2.5,-2,-1.5,-1,-0.5,-0.25,-0.10,0,0.10,0.25,0.5,1,1.5,2,2.5,3],ax2=None,label_prefix='',indices=[0,1]):
	from matplotlib.colors import LinearSegmentedColormap 
	if ax2==None:
		f2,ax2=plt.subplots(nrows=1,ncols=1,figsize=(12,8))
	
	# change to percentage
	bounds_pct=[element * 100 for element in bounds_load]
	mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
	norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	normpct = mpl.colors.BoundaryNorm(bounds_pct, len(bounds_pct)-1)
	m=Basemap(projection='robin',lon_0=0,ax=ax2)
	oldind=indices[1]
	newind=indices[0]
	#data_new=1e9*np.mean(mass[EXPS[0]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[0]]['POM'][seasonidx[i],0,:,:],0)
	#data_old=1e9*np.mean(mass[EXPS[1]]['SOA'][seasonidx[i],0,:,:],0)+np.mean(mass[EXPS[1]]['POM'][seasonidx[i],0,:,:],0)
	if fractional:
		#ax2.set_title('Annual mean fractional change of OA burden.')
		image=m.pcolormesh(lons,lats,np.squeeze(np.mean(load[newname][:,:,:],0)-np.mean(load[oldname][:,:,:],0))/np.mean(load[oldname][:,:,:],0)*100.0,norm=normpct,cmap=mycmap,latlon=True)
		cb = m.colorbar(image,"bottom", size="5%", pad="2%",extend='both',ticks=bounds_pct)
		cb.set_label(label_prefix+'('+EXP_NAMEs[newind]+'-'+EXP_NAMEs[oldind]+')/'+EXP_NAMEs[oldind]+' [%]',fontsize=12)
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




load={}
loadpoa={}
loadsoa={}
burden={}
mass={}
surface_mass_concentration={}
EXPS=['newsoa-ri','oldsoa-bhn','oldsoa-bhn-megan2']
for experiment in EXPS:
	surface_mass_concentration[experiment]={}
	mass[experiment]={}
	burden[experiment]={}
for exp in EXPS:
	#loadoa : Primary OA
	#loadsoa: secondary OA
	data_poa=nc.Dataset(output+'general_TM5_'+exp+'_2010.mm.nc','r').variables['loadoa'][:]
	data_soa=nc.Dataset(output+'general_TM5_'+exp+'_2010.mm.nc','r').variables['loadsoa'][:]
	load[exp]=np.squeeze(data_poa+data_soa)
	loadsoa[exp]=np.squeeze(data_soa)
	loadpoa[exp]=np.squeeze(data_poa)
	burden[exp]['oa']=np.squeeze(data_poa+data_soa)
	burden[exp]['soa']=np.squeeze(data_soa)
	burden[exp]['poa']=np.squeeze(data_poa)
	print exp
	data_soa=read_mass('M_SOA',output+'general_TM5_'+exp+'_2010.mm.nc')
	mass[exp]['SOA']=data_soa
	data_poa=read_mass('M_POM',output+'general_TM5_'+exp+'_2010.mm.nc')
	mass[exp]['POM']=data_poa
	


	# surface_soa=read_mass('sconcsoa',output+'general_TM5_'+exp+'_2010.mm.nc')
	# surface_mass_concentration[exp]['SOA']=surface_soa
	# surface_poa=read_mass('sconcoa',output+'general_TM5_'+exp+'_2010.mm.nc')
	# surface_mass_concentration[exp]['POM']=surface_poa

lon,lat=lonlat('TM53x2')
lons, lats = np.meshgrid(lon,lat)


#EXP_NAMEs['NEWSOA','OLDSOA','OLDSOA-MEGAN2']
f3x,ax3x=plt.subplots(ncols=3, figsize=(18,6))
#annual_diff_plot(loadsoa,EXPS[0],EXPS[1],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[0],'Burden of SOA\n ',[0,1])
#annual_diff_plot(loadsoa,EXPS[0],EXPS[2],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[1],'Burden of SOA\n ',[0,2])
#annual_diff_plot(loadsoa,EXPS[1],EXPS[2],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[2],'Burden of SOA\n ',[1,2])
annual_diff_plot(loadsoa,EXPS[0],EXPS[1],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[0],'Burden of SOA\n ',[0,1])
annual_diff_plot(loadsoa,EXPS[0],EXPS[2],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[1],'Burden of SOA\n ',[0,2])
annual_diff_plot(loadsoa,EXPS[1],EXPS[2],True,[-2,-1.5,-1,-0.5,-0.25,-0.1,-0.01,0.01,0.1,0.25,0.5,1,1.5,2],ax3x[2],'Burden of SOA\n ',[1,2])
plt.tight_layout()
ax3x[0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3x[1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3x[2].annotate('c)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
#ax3.set_title('Annual mean fractional change in burden of SOA.')
f3x.savefig(output_png_path+'/article/fig5_revised_annual_map_frac_load_soa_NEWSOA-OLDSOA.png',dpi=600)
f3x.savefig(output_pdf_path+'/article/fig5_revised_annual_map_frac_load_soa_NEWSOA-OLDSOA.pdf',dpi=600)
print EXPs

f3,ax3=plt.subplots(ncols=2,nrows=2, figsize=(12,6))
print ax3
annual_diff_plot(load,EXPS[0],EXPS[1],True,[-1.5,-1,-0.75,-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5,0.75,1,1.5],ax3[0,0],'Burden of total OA ')
annual_diff_plot(load,EXPS[0],EXPS[2],True,[-1.5,-1,-0.75,-0.5,-0.25,-0.1,-0.05,0.05,0.1,0.25,0.5,0.75,1,1.5],ax3[1,0],'Burden of total OA ')
#ax3.set_title('Annual mean fractional change in burden of total OA.')
m=Basemap(projection='robin',lon_0=0,ax=ax3[0,1])
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
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3[0,1])
cb.set_label('Change in OA mass on the surface \n (NEWSOA - OLDSOA)/OLDSOA[%]',fontsize=12)
#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
ax3[0,0].annotate('a)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3[0,1].annotate('b)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
m=Basemap(projection='robin',lon_0=0,ax=ax3[1,1])
bounds_load=[-100,-75,-50,-25,-10,-5,5,10,25,50,75,100]
norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
datasoa=mass[EXPS[0]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[0]]['POM'][:,0,:,:].mean(0)
dataold=mass[EXPS[2]]['SOA'][:,0,:,:].mean(0)+mass[EXPS[2]]['POM'][:,0,:,:].mean(0)
image=m.pcolormesh(lons,lats,np.squeeze((datasoa-dataold)/dataold)*100,norm=norm,cmap=mycmap,latlon=True)
#image=m.pcolormesh(lons,lats,np.squeeze(load['soa-riccobono'][:,:,:].mean(0)-load[EXPS[1]][:,:,:].mean(0))/np.mean(load[EXPS[1]][:,:,:].mean(0)),cmap=plt.cm.RdBu_r,latlon=True)
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))
m.drawcoastlines()
#cb = plt.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%",ax=ax3[1])
cb = m.colorbar(image, "bottom",ticks=bounds_load, size="5%", pad="2%",ax=ax3[1,1])
cb.set_label('Change in OA mass on the surface \n (NEWSOA - OLDSOA-MEGAN2)/OLDSOA-MEGAN2[%]',fontsize=12)
#ax3[1].set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA [%]',fontsize=18)
ax3[1,0].annotate('c)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
ax3[1,1].annotate('d)',xy=(0.05,0.95),xycoords='axes fraction',fontsize=18)
plt.tight_layout()
#f3.savefig(output_png_path+'/article/fig12_annual_2-panel-frac-load-conc-map_total_oa_NEWSOA-OLDSOA.png',dpi=600)
#f3.savefig(output_pdf_path+'/article/fig12_annual_2-panel-frac-load-conc-map_total_oa_NEWSOA-OLDSOA.pdf',dpi=600)

f3,ax3=plt.subplots(ncols=2,nrows=2, figsize=(12,6))
f3,ax3=seasonal_plot(load,EXPS[0],EXPS[1],True,[-3,-2,-1,-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.75,1,2,3],EXP_NAMEs[1],EXP_NAMEs[0])
f3.savefig(output_png_path+'/supplement/figS7_seasonal_map_frac_load_total_oa_NEWSOA-OLDSOA.png',dpi=600)


plt.show()
