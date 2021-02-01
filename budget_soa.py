import numpy as np 
import matplotlib.pyplot as plt 
import netCDF4 as nc 
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import sys
from colocate_aeronet import do_colocate
from general_toolbox import lonlat,get_gb_xr,monthlengths
from pylab import *
import xarray as xr
from settings import *
filein='../output/general_TM5_soa-riccobono_2010.mm.nc'

def calc_budget(ds):
	'''
	Calculate the budget from general output of tm5_tools

	Input:
	ds : xarray dataset with monthly mean general output for one year

	output:
	bud : dictionary with emi+component, wet+component, dry+component as keys, 
	      Values are global annual totals in Tg

	'''
	if ds.dims['time']!=12:
		exit('Time dimension is not 12 months!! Exiting!')
	bud={}
	monlengths=monthlengths(2010,ds['time'])
	dsgb=get_gb_xr()
	if 'lon' not in ds:
		dsgb.rename({'lon':'longitude','lat':'latitude'},inplace=True)
	comp=['soa','oa','bc','so4','ss','dust']
	deplist=['emi','wet','dry']
	for i in deplist:
		for j in comp:
			#print [i+j]
			if i+j in ds:
				bud[i+j]=(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
				#print i+j,': \t',(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
		if i == 'emi':
			j='terp'
			bud[i+j]=(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
			#print i+j,': \t',(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
			j='isop'
			bud[i+j]=(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
			#print i+j,': \t',(ds[i+j]*dsgb['area']*monlengths*3600*24).sum().values/1e9
			isop,terp=read_biomass_burning_nmvoc()
			bud[i+'bbterp']=terp
			bud[i+'bbisop']=isop

	i='load'
	for j in comp:
		#print (i+j)
		if i+j in ds:
				bud[i+j]=(ds[i+j]*dsgb['area']).mean(dim='time').sum().values/1e9
				#print i+j,':\t',(ds[i+j]*dsgb['area']).mean(dim='time').sum().values/1e9
	bud['prodelvoc']=(ds['prod_elvoc']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodelvoc_IP']=((ds['p_el_ohisop']+ds['p_el_o3isop'])*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodelvoc_MT']=((ds['p_el_ohterp']+ds['p_el_o3terp'])*dsgb['area']*monlengths).sum().values*3600*24/1e9
	#print 'prodelvoc: \t',(ds['prod_elvoc']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodliqso4']=(ds['prod_liq_so4']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodgasso4']=(ds['prod_gas_so4']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	#print bud['prodgasso4']
	#print bud['prodliqso4']
	bud['prodso4']=bud['prodgasso4']+ bud['prodliqso4']
	bud['prodsvoc']=(ds['prod_svoc']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodsvoc_IP']=((ds['p_sv_ohisop']+ds['p_sv_o3isop'])*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodsvoc_MT']=((ds['p_sv_ohterp']+ds['p_sv_o3terp'])*dsgb['area']*monlengths).sum().values*3600*24/1e9
	bud['prodsoa']=((ds['prod_svoc']+ds['prod_elvoc'])*dsgb['area']*monlengths).sum().values*3600*24/1e9
	#print 'prodsvoc: \t',(ds['prod_svoc']*dsgb['area']*monlengths).sum().values*3600*24/1e9
	#print 'prodsoa: \t',((ds['prod_svoc']*dsgb['area']*monlengths).sum().values+(ds['prod_elvoc']*dsgb['area']*monlengths).sum().values)*3600*24/1e9
	oldemisoa=xr.open_dataset(output+'/SOA.nc')
	bud['emisoa']=(oldemisoa['FIELD'].values).sum()*1e-9*2.4/1.15 # kg->Tg * aging * 1.15 (unclear, but it is in the old code)
	return bud

def soa_budget(filedict):
	
	budget={}
	for ii in filedict:
		print 'processing: ',ii
		ds=xr.open_dataset(filedict[ii])
		budget[ii]=calc_budget(ds)
	return budget
def print_budget(budget):
	comp=['soa','oa','bc','so4','ss','dust','terp','isop','bbterp','bbisop']
	for exp in budget:

		print exp
		for i in comp:
			for j in ['emi','wet','dry','load','prod']:
				if j+i in budget[exp]:
					if exp=='NEWSOA' and j+i=='emisoa':
						continue
					elif  exp=='OLDSOA' and j+i=='prodsoa':
						continue
					else:
						print j+i,': ',budget[exp][j+i]
					
			#print i,': \t',budget[exp]['load'+i]/(budget[exp]['wet'+i]+budget[exp]['dry'+i])*365
			if exp=='NEWSOA':
				if 'emi'+i in budget['NEWSOA'] and i !='so4' and i!='soa' and 'terp' not in i and 'isop' not in i:
					#print budget[exp]['load'+i],budget[exp]['emi'+i]
					print 'lifetime',i,': \t',budget[exp]['load'+i]/(budget[exp]['emi'+i])*365
				elif i=='soa':
					print 'lifetime',i,': \t',budget[exp]['load'+i]/(budget[exp]['prod'+i])*365	
				elif  'terp' not in i and 'isop' not in i:
					print 'lifetime',i,': \t',budget[exp]['load'+i]/(budget[exp]['emi'+i]+budget[exp]['prod'+i])*365
					print budget[exp]['emi'+i],budget[exp]['prod'+i]
					print 'so4 not emitted'	
			else:
				if exp=='OLDSOA' and 'emi'+i in budget['OLDSOA'] and i!='so4' and'terp' not in i and 'isop' not in i:
					print 'lifetime',i,': \t',budget[exp]['load'+i]/(budget[exp]['emi'+i])*365
				elif 'terp' not in i and 'isop' not in i:
					print 'lifetime',i,': \t',budget[exp]['load'+i]/(budget[exp]['emi'+i]+budget[exp]['prod'+i])*365
					print budget[exp]['emi'+i],budget[exp]['prod'+i]
					print 'so4 not emitted'	
def soa_table(budget):
	budget_table_latex=open(paper+"/budget_table.tex","w")
	print 'making table for paper tm5 SOA'
	text={'loadsoa':'Burden Tg','prodsoa':'Total SOA production Tg [y$^-1$]','prodelvoc':'contribution of ELVOC [Tg y$^-1$]','prodelvoc_IP':'from isoprene [Tg y$^-1$]','prodelvoc_MT':'from monoterpene [Tg y$^-1$]','prodsvoc':'contribution of SVOC [Tg y$^-1$]','prodsvoc_IP':'from isoprene [Tg y$^-1$]','prodsvoc_MT':'from monoterpene [Tg y$^-1$]','emisoa':'Emission [Tg y$^-1$]','wetsoa':'Wet Deposition [Tg y$^-1$]','drysoa':'Dry Deposition [Tg y$^-1$]','life':'Lifetime [days]'}
	for ii in ['loadsoa','prodsoa','prodelvoc','prodelvoc_IP','prodelvoc_MT','prodsvoc','prodsvoc_IP','prodsvoc_MT','emisoa','wetsoa','drysoa','life']:
		print text[ii],'&',
		budget_table_latex.write(text[ii]+'&')
		for jj in sorted(budget.keys()):
			if ii=='life':
				if jj=='NEWSOA':
					print (budget[jj]['loadsoa']/budget[jj]['prodsoa'])*365,'&',
					budget_table_latex.write('%5.2f & '%((budget[jj]['loadsoa']/budget[jj]['prodsoa'])*365))
				else:
					print (budget[jj]['loadsoa']/budget[jj]['emisoa'])*365,'&',
					budget_table_latex.write('%5.2f &'%((budget[jj]['loadsoa']/budget[jj]['emisoa'])*365))
			else:
				if jj=='OLDSOA' and 'prod' in ii:
					budget_table_latex.write('-- ')
					print '-- ',
				elif jj=='NEWSOA' and 'emi' in ii:
					budget_table_latex.write('-- &')
					print '-- &',
				else:
					budget_table_latex.write('%5.2f &'%(budget[jj][ii]))
					print budget[jj][ii],'&',

		budget_table_latex.write('\\\\\n')
		print '\\\\'
	budget_table_latex.close()
def read_biomass_burning_nmvoc():
	# select time steps for 2010 from inputfiles for biomass burning emissions of MT and ISOP
	ds=xr.open_dataset(basepath+'/input/NMVOC-C10H16-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_185001-201512.nc')
	terp=ds['C10H16'].isel(time=[1920,1921,1922,1923,1924,1925,1926,1927,1928,1929,1920,1931])
	ds=xr.open_dataset(basepath+'/input/NMVOC-C5H8-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn_185001-201512.nc')
	isop=ds['C5H8'].isel(time=[1920,1921,1922,1923,1924,1925,1926,1927,1928,1929,1920,1931])
	ds=xr.open_dataset(basepath+'/input/gridcellarea-em-biomassburning_input4MIPs_emissions_CMIP_VUA-CMIP-BB4CMIP6-1-2_gn.nc')
	gb=ds['gridcellarea']

	terp=(terp*gb*monthlengths(2010,terp['time'])).sum().values*3600*24*1e-9
	isop=(isop*gb*monthlengths(2010,isop['time'])).sum().values*3600*24*1e-9
	return terp,isop
def land_burden():
	ds=xr.open_dataset(basepath+'/input/landfraction_3x2.nc')
	dsgb=get_gb_xr()
	if 'lon' not in ds:
		dsgb.rename({'lon':'longitude','lat':'latitude'},inplace=True)
	#for exp in EXPS:
	exp='newsoa-ri'
	ds2=xr.open_dataset(output+'/general_TM5_'+exp+'_2010.mm.nc')
	f,ax=plt.subplots(1)
	landburdenmap=ds['LANDFRACTION']*(ds2['loadsoa']).mean(dim='time')

	# lon,lat=lonlat('TM53x2')
	# lons, lats = np.meshgrid(lon,lat)
	# #print lon
	# m=Basemap(projection='robin',lon_0=0,ax=ax)
	# bounds_load=[0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
	# norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	# mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1) 
	# image=m.pcolormesh(lons,lats,np.squeeze(landburdenmap)*1e6,norm=norm,cmap=mycmap,latlon=True)
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
	# #ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
	# f,ax=plt.subplots(1)
	# #landburdenmap2=lsm.roll
	# oceanburdenmap=(ds['LANDFRACTION']-1)*-1*(ds2['loadsoa']).mean(dim='time')
	# lon,lat=lonlat('TM53x2')
	# lons, lats = np.meshgrid(lon,lat)
	# m=Basemap(projection='robin',lon_0=0,ax=ax)
	# bounds_load=[0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
	# norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	# mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1)
	# image=m.pcolormesh(lons,lats,np.squeeze(landburdenmap)*1e6,norm=norm,cmap=mycmap,latlon=True)
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
	# #ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)

	# f,ax=plt.subplots(1)
	# landburdenmap=(ds2['loadsoa']).mean(dim='time')
	# lon,lat=lonlat('TM53x2')
	# lons, lats = np.meshgrid(lon,lat)
	# m=Basemap(projection='robin',lon_0=0,ax=ax)
	# bounds_load=[0.0001,0.01,0.05,0.1,0.25,0.5,1,2,3]
	# norm = mpl.colors.BoundaryNorm(bounds_load, len(bounds_load)-1)
	# mycmap=plt.get_cmap('RdBu_r',len(bounds_load)-1)
	# image=m.pcolormesh(lons,lats,np.squeeze(oceanburdenmap)*1e6,norm=norm,cmap=mycmap,latlon=True)
	# m.drawparallels(np.arange(-90.,90.,30.))
	# m.drawmeridians(np.arange(-180.,180.,60.))
	# m.drawcoastlines()
	# cb = m.colorbar(image,"bottom", ticks=bounds_load,size="5%", pad="2%")
	# #ax.set_title('Fractional change in annual mean organic aerosol concentration at the surface \n (NEWSOA-OLDSOA)/OLDSOA',fontsize=18)
	# plt.show()
	#print np.shape(landburden)
	landburden=((ds2['loadsoa']+ds2['loadoa'])*dsgb['area']*ds['LANDFRACTION']).mean(dim='time')
	oceanburden=((ds2['loadsoa']+ds2['loadoa'])*dsgb['area']*(ds['LANDFRACTION']-1.0)*-1.0).mean(dim='time')
	#print np.shape(landburden)
	print landburden.sum()*1e-9
	print oceanburden.sum()*1e-9
	#(ds[i+j]*dsgb['area']).mean(dim='time').sum().values/1e9
	return landburden,landburdenmap
if __name__ == "__main__": 
	a,b=land_burden()

	#read_biomass_burning_nmvoc()
	filedict={}
	for name,exp in zip(EXP_NAMEs,EXPS):
		filedict[name]=output+'/general_TM5_'+exp+'_2010.mm.nc'#}#,'OLDSOA':'../output/general_TM5_'+exp+'_2010.mm.nc'}
	budget=soa_budget(filedict)
	print_budget(budget)
	soa_table(budget)