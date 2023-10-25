from pdb import run
import strax
import warnings
import numpy as np
import sys
warnings.filterwarnings("ignore")

if len(sys.argv)<2:
    print(sys.argv)
    print('/home/user/env/anaconda3/envs/DAQ/bin/python </home/user/data_process/tutorials_plotTBA.py> <run id> <peak type> ')
######## define a context, add plugins into it
raw_record_site = strax.DataDirectory('/data1')
st   = strax.Context(
                    register=[
                    strax.CAENRecords_3,
                    strax.PulseProcessing,
                    strax.Peaklets,
                    strax.PeakletClassification],

                    storage=[raw_record_site])
ptype  = int(sys.argv[2])
run_id = sys.argv[1]

### find the sources that are ready for analysis
dsets = st.select_runs(available = 'peaklets') ## available: means which type of the data that are available

### lets get all the run3 data
run_ids  = [i for i in dsets['name'] if i[4]==run_id]

peaklets            = st.get_array(run_id=run_ids,targets = 'peaklets')


# records             = st.get_array(run_id=run_id,targets = 'records')
peak_classification = st.get_array(run_id=run_ids,targets = 'peaklet_classification')
peaklets['type']    = peak_classification['type']

## select those peaklet type is 1
peaklets = peaklets[peaklets['type']==ptype]

## get the integral of each PMT
area_per_channel = peaklets['area_per_channel']

import numpy as np
### separate areas into top and bottom
top_area    = area_per_channel[:,:4]
bottom_area = area_per_channel[:,4:7]

### sum over top and bottom PMTs respectively
top_area    = np.sum(top_area,axis = 1)
bottom_area = np.sum(bottom_area,axis = 1)

### drop those zero PEs
PE          = peaklets[bottom_area !=0]['area']
top_area    = top_area[bottom_area !=0]
bottom_area = bottom_area[bottom_area!=0]

bottom_area = bottom_area[top_area!=0]
PE          = PE[top_area !=0]
top_area    = top_area[top_area != 0]

## calculate asymmetric
asy         = (top_area-bottom_area)/(top_area+bottom_area)

## calculate the 2D histogram
(count,bins_x,bins_y) = np.histogram2d(PE,asy,bins = (50,50),range = ((0,100),(-1,1)))

import matplotlib.pyplot as plt
## visualization
fig = plt.figure(dpi = 100)
ax = plt.subplot(111)
X, Y                = np.meshgrid(bins_x[1:], bins_y[1:])
X                   = X.T
Y                   = Y.T

c = ax.pcolor(
    X,
    Y,
    count,
    shading='nearest'
)

fig.colorbar(c,ax = ax,label = 'count/total')
ax.set_xlabel('PE',fontsize = 15)
ax.set_ylabel('(top-bottom)/(top+bottom) PE',fontsize = 10)
ax.set_title('top bottom asymmetric of S'+str(ptype),fontsize = 15)
plt.savefig('/home/user/data_process/figures/'+'run'+run_id+'S'+str(ptype)+'TBA'+'.pdf')