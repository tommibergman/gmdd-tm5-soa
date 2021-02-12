import glob
import subprocess

def aggre_cci(exp,var='od550aer'):
	input_CCI_SU_location = '/Volumes/Utrecht/CCI/'
	output_masked_location = '/Volumes/Utrecht/MODIS_masked/'
	output_aggregated_location = '/Volumes/Utrecht/CCI/aggregated_cci/'
	#output_aggregated_location = 'CCI/aggregated_cci/'
	output_col_location = '/Volumes/Utrecht/CCI/cci_tm5_col/'

	QA_Flag = 'AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag'
	AOD = 'AOD550'
	year=2010
	AOD_tm5=var#'od550aer'

	tm5data='/Volumes/Utrecht/'+exp+'/general_TM5_'+exp+'_2010.lev0.'+var+'.nc' 

	days=[31,28,31,30,31,30,31,31,30,31,30,31]
	for mon in range(1,13):
		if len(glob.glob(output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".fix.nc"))==0:
			for day in range(1,31):
		#	    for hours in range(24):
		#	        for minutes in range(0,60,5):
				print "day"+str(mon).zfill(2) +str(day).zfill(2) #prints dayDDDtimeHHMM
				
				CCI_SU_Filenames = str(year)+str(mon).zfill(2)+"*-ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_*.nc"
				CCI_SU_fname_out = 'ESACCI-L2P-AATSR-SU'
				CCI_SU = input_CCI_SU_location+str(year)+"_"+str(mon).zfill(2)+"_"+str(day).zfill(2)+'/' +CCI_SU_Filenames
				#Masked_data_filename = 'MOD04_L2_2010_'+str(day).zfill(3)+"."+str(hours).zfill(2)+str(minutes).zfill(2)+'.nc'

				#if len(glob.glob(MODIS)) == 1 and len(glob.glob(output_masked_location+"Masked_"+Masked_data_filename))==0: #checks if the file in MODIS exists
				#    subprocess.call("cis eval "+QA_Flag+"=QA,"+AOD+"=AOD:"+MODIS+" 'numpy.ma.masked_where(QA <= 1, AOD)' 1 -o "+AOD+":"+output_masked_location+"Masked_"+Masked_data_filename, shell=True)
				    #Masks AOD variable where QA_Flag smaller or equal to 1
				Aggregated_filename= output_aggregated_location+CCI_SU_fname_out+'_agg.'+str(year)+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
				#CCI_SU_data= 'Masked_MOD04_L2_2010_'#+str(day).zfill(3)
				print (Aggregated_filename)
				print glob.glob(Aggregated_filename)
				if day <=days[mon-1] and len(glob.glob(Aggregated_filename))==0:
					if day==days[mon-1]:
						end=str(mon+1).zfill(2)+"-01"
					else:
						end=str(mon).zfill(2)+"-"+str(day+1).zfill(2)
					if mon<=12:
						print "cis aggregate " +AOD+":"+CCI_SU+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename
						subprocess.call("cis aggregate " +AOD+":"+CCI_SU+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					else:
						print "cis aggregate " +AOD+":"+CCI_SU+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename
						subprocess.call("cis aggregate " +AOD+":"+CCI_SU+" t=[2010-"+str(mon).zfill(2)+"-"+str(day).zfill(2)+"T00:00,2010"++str(mon).zfill(2)+"-"+str(day).zfill(2)+"T23:59,PT1H],x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					#subprocess.call("cis aggregate " +AOD+":"+CCI_SU+" x=[-180,180,1],y=[-90,90,1] -o "+Aggregated_filename, shell=True)	
					#col_outdata=output_col_location+'col_ESACCI_SU_TM5.2010'+str(mon).zfill(2)+str(day).zfill(2)+'.nc'
					print "ncpdq -a time,longitude,latitude "+Aggregated_filename+" "+Aggregated_filename[:-2]+"fix.nc"
					subprocess.call("ncpdq -a time,longitude,latitude "+Aggregated_filename+" "+Aggregated_filename[:-2]+"fix.nc", shell=True)
					#print "cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+Aggregated_filename+":variable="+AOD+"  -o "+col_outdata
					#subprocess.call("cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+Aggregated_filename+":variable="+AOD+"  -o "+col_outdata , shell=True)	
		
		#if len(glob.glob(output_aggregated_location+'ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated.'+str(year)+str(mon).zfill(2)+".fix.nc"))==0:
		
		if len(glob.glob(output_aggregated_location+CCI_SU_fname_out+'_agg.'+str(year)+str(mon).zfill(2)+".nc"))==0:
			print"cdo mergetime "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+"??.fix.nc "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".nc"
			subprocess.call("cdo mergetime "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+"??.fix.nc "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".nc",shell=True)
		
		print glob.glob(output_aggregated_location+CCI_SU_fname_out+'_agg.'+str(year)+str(mon).zfill(2)+".newfix.nc")
		print (output_aggregated_location+CCI_SU_fname_out+'_agg.'+str(year)+str(mon).zfill(2)+".newfix.nc")
		if len(glob.glob(output_aggregated_location+CCI_SU_fname_out+'_agg.'+str(year)+str(mon).zfill(2)+".newfix2.nc"))==0:
			print "ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".nc"+" "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".newfix2.nc"
			
			subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".nc"+" "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".newfix2.nc", shell=True)
			#subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".fix.nc test.newfix.2.nc", shell=True)
		#subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".fix.nc"+" output/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".newfix.nc", shell=True)
		print 'after ncpdq'
		col_outdata=output_col_location+'col_ESACCI_SU_TM5_'+exp+'_2010'+str(mon).zfill(2)+'_'+AOD_tm5+'.nc'
		if len(glob.glob(col_outdata))==0:
			print "cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".newfix2.nc:variable="+AOD+"  -o "+col_outdata
			subprocess.call("cis version",shell=True)
		
			subprocess.call("cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+"ESACCI-L2P-AATSR-SU_agg."+str(year)+str(mon).zfill(2)+".newfix2.nc:variable="+AOD+",collocator=nn  -o "+col_outdata , shell=True)	
		#subprocess.call("cis col " +AOD_tm5+":"+tm5data+" output/ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+str(mon).zfill(2)+".newfix.nc:variable="+AOD+"  -o "+col_outdata , shell=True)	
		#subprocess.call("cis col " +AOD_tm5+":"+tm5data+" test.newfix.2.nc:variable="+AOD+"  -o "+col_outdata , shell=True)	
		#if len(glob.glob(output_aggregated_location+'ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated.'+str(year)+".fix.nc"))==0:
		#	print"cdo mergetime "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+"??.fix.nc "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".fix.nc"
		#	subprocess.call("cdo mergetime "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+"??.fix.nc "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".fix.nc",shell=True)
	#if len(glob.glob(output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".newfix.nc"))==0:
	#	print "ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".fix.nc"+" "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".newfix.nc"
	#	subprocess.call("ncpdq -a time,longitude,latitude "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".fix.nc"+" "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".newfix.nc", shell=True)

	#if len(glob.glob(col_outdata))==0:
	#	print "cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".newfix.nc:variable="+AOD+"  -o "+col_outdata
	#	subprocess.call("cis col " +AOD_tm5+":"+tm5data+" "+output_aggregated_location+"ESACCI-L2P_AEROSOL-AER_PRODUCTS-AATSR_ENVISAT-SU_aggregated."+str(year)+".newfix.nc:variable="+AOD+"  -o "+col_outdata , shell=True)	
def main():
	aggre_cci('oldsoa-bhn','od550soa')
	aggre_cci('newsoa-ri','od550soa')
	aggre_cci('oldsoa-bhn','od550aer')
	aggre_cci('newsoa-ri','od550aer')
		
if __name__ == '__main__':
	main()
