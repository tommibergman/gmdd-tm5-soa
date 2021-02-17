import glob	
import subprocess
import os
import datetime
def aggregate_and_collocate_MODIS(exp,var='od550aer'):
	input_MODIS_location = '/Volumes/Utrecht/MODIS_masked/'
	output_aggregated_location = '/Volumes/Utrecht//MODIS_aggregated/'
	output_col_location = '/Volumes/Utrecht/MODIS_collocated/'

	QA_Flag = 'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'
	AOD = 'AOD_550_Dark_Target_Deep_Blue_Combined'
	year=2010
	AOD_tm5=var#'od550aer'

	#tm5data='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev0.'+var+'.nc' 
	tm5data='tm5-soa/output/raw/'+'/general_TM5_'+exp+'_2010.lev0.'+var+'.nc'
	#'/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev0.'+var+'.nc' 
	if not os.path.isfile(tm5data):
		print "TM5 data not found, exiting!"
		exit()
	print 'Aggregating MODIS data from:'+input_MODIS_location
	print 'to: '+output_aggregated_location
	dayofyear=0
	days=[31,28,31,30,31,30,31,31,30,31,30,31]	
	for mon in range(1,13):
	#for day in range(1,366):	
		for day in range(1,32):
			if day >days[mon-1]:
				break
			dayofyear+=1
			#if len(glob.glob(output_aggregated_location+"Masked_QA2_*_L2_2010_*"+str(year)+str(mon).zfill(2)+".fix.nc"))==0:
			print day,dayofyear,datetime.datetime(2010,1,1)+datetime.timedelta(dayofyear-1)
			if len(glob.glob(output_aggregated_location+"MOD04_MYD04_L2_2010_QA2_*"+str(year)+str(mon).zfill(2)+".fix.nc"))==0:
				#for day in range(1,31):
				print "day " +str(day).zfill(3) #prints day DDD
				

				MODIS_Filenames='Masked_QA2_*_L2_2010_'+str(dayofyear).zfill(3)+'.????.nc'
				#MODIS_MYD_Filenames='Masked_QA2_MYD04_L2_2010_'+str(dayofyear).zfill(3)+'.????.nc'
				#MODIS_MOD_Filenames='Masked_QA2_MOD04_L2_2010_'+str(dayofyear).zfill(3)+'.????.nc'
				MODIS_in = input_MODIS_location+'/' +MODIS_Filenames
				#MODIS_in = input_MODIS_location+'/' +MODIS_MYD_Filenames
				aggregated_MODIS_fname_out='MOD04_MYD04_L2_2010_QA2_aggregated.'
				#Masked_data_filename = 'MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'

				Aggregated_filename= output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
				print (Aggregated_filename)
				if day <=days[mon-1] and len(glob.glob(Aggregated_filename))==0:
					if day==days[mon-1]:
						end=str(mon+1).zfill(2)+"-01"
					else:
						end=str(mon).zfill(2)+"-"+str(day+1).zfill(2)
					print Aggregated_filename

					if mon<=12:
						print "cis aggregate " +AOD+":"+MODIS_in+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename
						subprocess.call("cis aggregate " +AOD+":"+MODIS_in+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					else:
						print "cis aggregate " +AOD+":"+MODIS_in+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename
						subprocess.call("cis aggregate " +AOD+":"+MODIS_in+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010"++str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					subprocess.call("cis aggregate " +AOD+":"+CCI_SU+" x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					col_outdata=output_col_location+'col_ESACCI_SU_TM5.2010'+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
				if day <=days[mon-1] and len(glob.glob(Aggregated_filename[:-2]+"fix.nc"))==0:
					# change order of dimensions for cdo
					print "ncpdq -a time,longitude,latitude "+Aggregated_filename+" "+Aggregated_filename[:-2]+"fix.nc"
					subprocess.call("ncpdq -a time,longitude,latitude "+Aggregated_filename+" "+Aggregated_filename[:-2]+"fix.nc", shell=True)
					print "cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+Aggregated_filename+":variable="+AOD+"  -o "+col_outdata
					subprocess.call("cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+Aggregated_filename+":variable="+AOD+"  -o "+col_outdata , shell=True)	
		
		print 'Merging monthly aggregated data to '+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc"
		if len(glob.glob(output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc"))==0:
			print"cdo mergetime "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+"??.fix.nc "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc"
			subprocess.call("cdo mergetime "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+"??.fix.nc "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc",shell=True)
		print 'change order of dimensions to:'+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc"
		print glob.glob(output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc")
		print (output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc")
		# change order of dimensions for cdo
		if len(glob.glob(output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc"))==0:
			print "ncpdq -a time,longitude,latitude "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc"+" "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc"
			subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".nc"+" "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc", shell=True)

		# change order of dimensions for cdo
		subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".fix.nc"+" output/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".newfix.nc", shell=True)
		if len(glob.glob(output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+'.mm.nc'))==0:
			subprocess.call('cdo monmean '+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".mm.nc ",shell=True)

		col_outdata=output_col_location+'col_MODIS_TM5_'+exp+'_2010'+str(mon).zfill(2)+'_'+AOD_tm5+'.nc'
		print 'collocate TM5 data with monthly MODIS data to: '+col_outdata
		if len(glob.glob(col_outdata))==0:
			print "cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc:variable="+AOD+"  -o "+col_outdata
			subprocess.call("cis version",shell=True)
		
			subprocess.call("cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+aggregated_MODIS_fname_out+str(year)+str(mon).zfill(2)+".newfix2.nc:variable="+AOD+",collocator=nn  -o "+col_outdata , shell=True)	
		col_mm_outdata=output_col_location+'col_MODIS_TM5_'+exp+'_2010'+str(mon).zfill(2)+'_'+AOD_tm5+'.mm.nc'
		if len(glob.glob(col_mm_outdata))==0:
			subprocess.call('cdo monmean '+col_outdata+' '+col_mm_outdata,shell=True)
		

def Mask_QA_MODIS():

	
	QA_Flag = 'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'
	AOD = 'AOD_550_Dark_Target_Deep_Blue_Combined'
	Level=2
	input_MODIS_location_MOD = '/Volumes/Utrecth/MODIS/6/MOD04_L2/2010/'
	input_MODIS_location_MYD = '/Volumes/Utrecth/MODIS/6/MYD04_L2/2010/'
	output_masked_location = '/Volumes/Utrecht/MODIS_masked/'
	for day in range(1,366):
		for hours in range(24):
			for minutes in range(0,60,5):
				MODIS_Filename_MYD = "MYD04_L2.A2010"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+"*"
				MODIS_Filename_MOD = "MOD04_L2.A2010"+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+"*"
				MODIS_MOD = input_MODIS_location_MOD+str(day).zfill(3)+'/'+MODIS_Filename_MOD
				MODIS_MYD = input_MODIS_location_MYD+str(day).zfill(3)+'/'+MODIS_Filename_MYD
				Masked_MYD_filename = 'MYD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'
				Masked_MOD_filename = 'MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'

				if len(glob.glob(output_masked_location+'Masked_'+Masked_MYD_filename))==0:
					print "day"+str(day).zfill(3)+"time"+str(hours).zfill(2)+str(minutes).zfill(2) #prints dayDDDtimeHHMM
					if len(glob.glob(MODIS_MYD)) == 1: #checks if the file in MODIS exists and masked data is missing
						print "cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS_MYD+" 'numpy.ma.masked_where(QA <= "+str(Level)+", AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_MYD_filename
						subprocess.call("cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS+" 'numpy.ma.masked_where(QA <= "+str(Level)+", AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_MYD_filename, shell=True)
				if len(glob.glob(output_masked_location+'Masked_'+Masked_MOD_filename))==0:
					if len(glob.glob(MODIS_MOD)) == 1 and len(glob.glob(output_masked_location+'Masked_'+Masked_MOD_filename))==0: #checks if the file in MODIS exists and masked data is missing
						print "cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS_MOD+" 'numpy.ma.masked_where(QA <= "+str(Level)+", AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_MOD_filename
						subprocess.call("cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS_MOD+" 'numpy.ma.masked_where(QA <= "+str(Level)+", AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_MOD_filename, shell=True)
						#Masks AOD variable where QA_Flag smaller or equal

def main():
	from timeit import default_timer as timer
	Mask_QA_MODIS()
	start=timer()
	aggregate_and_collocate_MODIS('oldsoa-bhn','od550aer')
	end=timer()
	print 'Time elapsed for collocation of oldsoa-bhn: ',end-start
	start=timer()
	aggregate_and_collocate_MODIS('newsoa-ri','od550aer')
	end=timer()
	print 'Time elapsed for collocation of newsoa-ri: ',end-start
if __name__ == '__main__':
	main()