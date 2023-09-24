# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:03:40 2016

@author: Xiaoxuan Jia
"""

# utils for spikes analysis
import numpy as np
from scipy import stats
import platform
if int(platform.python_version()[0])>2:
    import _pickle as pk
else:
    import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew


def nanunique(x):
    """np.unique get rid of nan numbers."""
    temp = np.unique(x.astype(float))
    return temp[~np.isnan(temp)]

def find_range(x,a,b,option='within'):
    """Find data within range [a,b]"""
    if option=='within':
        return np.where(np.logical_and(x>=a, x<=b))
    elif option=='outside':
        return np.where(np.logical_or(x < a, x > b))
    else:
        print('undefined function')

def flatten_list(l):
    """Flatten a list."""
    return [item for sublist in l for item in sublist]

def autocorr(x):
    # autocorrelation
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def crosscorr(x):
    # autocorrelation
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def get_PSTH(unit_binarized, PSTH_bintime, fs=1000):
    """Calculate PSTH averaged across trials based on binarized spike trains.
    unit_binarized: units*rep*time
    PSTH_bintime: 
    """
    PSTH_binsize = int(PSTH_bintime/1000.*fs)
    sizes = np.shape(unit_binarized)
    print(sizes)

    if len(sizes)==4:
        unit_psth=np.zeros([sizes[0],sizes[1],sizes[2], int(sizes[3]/PSTH_binsize)])
        
        for u in range(sizes[0]):
            for o in range(sizes[1]):
                for r in range(sizes[2]):
                    data = unit_binarized[u,o,r,:]
                    psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[3]/PSTH_binsize),PSTH_binsize)),axis=1)
                    unit_psth[u,o,r,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    if len(sizes)==3:
        unit_psth=np.zeros([sizes[0],sizes[1],int(sizes[2]/PSTH_binsize)])
        
        for u in range(sizes[0]):
            for r in range(sizes[1]):
                data = unit_binarized[u,r,:]
                psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[2]/PSTH_binsize),PSTH_binsize)),axis=1)
                unit_psth[u,r,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    if len(sizes)==2:
        unit_psth=np.zeros([sizes[0],sizes[1]/PSTH_binsize])
        
        for u in range(sizes[0]):
            data = unit_binarized[u,:]
            psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[1]/PSTH_binsize),PSTH_binsize)),axis=1)
            unit_psth[u,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    return unit_psth, time

def get_PSTH_alldim(unit_binarized, PSTH_bintime, fs=1000):
    """Calculate PSTH averaged across trials based on binarized spike trains.
    unit_binarized: last dimension is time
    PSTH_bintime: 
    """
    unit_binarized=np.array(unit_binarized)
    unit_psth = unit_binarized.reshape(unit_binarized.shape[:-1] + (unit_binarized.shape[-1]/PSTH_bintime,PSTH_bintime))
    unit_psth = unit_psth.mean(-1)
    time = np.array(range(unit_binarized.shape[-1]/PSTH_bintime))*PSTH_bintime/1000.
    return unit_psth, time

def plot_PSTH(psth, time, unit_list):
    """Plot calculated psth
    psth is unit by time (averaged across repeats)
    """
    plt.figure(figsize=(20,20))
    for i in range(np.shape(psth)[0]):
        ax = plt.subplot(10,round(np.shape(psth)[0]/10)+1,i+1)
        ax.plot(time,psth[i,:])
        ax.set_title(str(unit_list[i]))
        if i < np.shape(psth)[0]-1:
            ax.set_xticks([])
    return 'end'

def threshold(unit_binarized, ISI, unit_list_V1, delay=70, thresh=5):
    """Calculate FR. Apply FR threshold"""
    # Threshold for FR
    ISI_ms = int(round(ISI*1000))
    FR = np.sum(unit_binarized[:,:,delay:((np.shape(unit_binarized)[2]-ISI_ms)+delay)], axis=2)
    FR_ISI = np.sum(unit_binarized[:,:,((np.shape(unit_binarized)[2]-ISI_ms)+delay):], axis=2)
    cell_FR_idx = np.where(np.mean(FR,axis=1)>=thresh)
    cell_FR=FR[cell_FR_idx]
    cell_FR_ISI = FR_ISI[cell_FR_idx]
    good_unit_list=np.array(unit_list_V1)[cell_FR_idx]
    return cell_FR, cell_FR_ISI, cell_FR_idx, good_unit_list

def get_FR(unit_binarized, ISI, interval, delay=70):
    """Calculate FR. ISI is stimulus presentation time."""
    # Threshold for FR
    ISI_ms = int(round(ISI*1000))
    if ISI_ms+delay<=np.shape(unit_binarized)[2]:
        FR = np.sum(unit_binarized[:,:,delay:(ISI_ms+delay)], axis=2)/float(ISI_ms)*1000
    else:
        FR = np.sum(unit_binarized[:,:,delay:], axis=2)/float(np.shape(unit_binarized)[2]-delay)*1000
    # consider information delay and remove rebound effect
    if interval!=0:
        FR_ISI = np.sum(unit_binarized[:,:,(ISI_ms+200):], axis=2)/float(np.shape(unit_binarized)[2]--ISI_ms-200)*1000
    else:
        FR_ISI=0
    return FR, FR_ISI

class NWB_adapter(object):

    def __init__(self, nwb_data, probename, bin_num=15):
        self.nwb_data = nwb_data
        self.probe_list = list(nwb_data['processing'].keys())
        self.stim_list = list(nwb_data['stimulus']['presentation'].keys())

        self.probename = probename
        self.bin_num = bin_num
        #self.wm = self.get_wm(self.probename)

    def get_wm(self, probename, wm=0):
        """Find the depth of first white matter for later use to determine cortex."""
        if wm==0:
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value

            a = plt.hist(channel_ypos,bins=self.bin_num)
            # white matter determination based on histogram
            if len(a[1][np.where(a[0]==0)[0]])==0:
                wm=-4000
                print('no white matter; all in cortex')
            else:
                wm = a[1][np.where(a[0]==0)[0]][-1]
                if sum(a[0][np.where(a[1]>wm)[0][:-1]])<3:
                    wm=-4000
                    print('outlier single neuron in superficial layer, white matter depth: '+str(wm))
                else:
                    print('white matter depth: '+str(wm))
            self.wm = wm
        else:
            self.wm = wm
        return wm

    def get_unit_list(self):
        '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
        #f = h5py.File(NWB_file, 'r')
        unit_list = self.nwb_data['processing'][self.probename]['unit_list'].value 
        if 'noise' in unit_list:
            unit_list.remove('noise')
        self.unit_list = unit_list
        self.channel_list = self.nwb_data['processing'][self.probename]['channel_list'].value
        
    def get_stim_table(self, key):
        """Get stimulus sync pulse for specified (key) stimuli.
        Opto tagging exp only has one column: 'Start'. 
        """
        if key!='spontaneous':
            temp1 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['data'].value, columns = self.nwb_data['stimulus']['presentation'][key]['features'].value)
            if min(np.shape(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value))==2:
                temp2 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start','End'])
            else:
                temp2 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start'])
            stim_table = temp2.join(temp1)
            del temp1, temp2
        else:
            stim_table = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start','End'])     
        return stim_table

    def get_stim_table_full(self):
        """Get the full synch pulse tabel for all stimuli.
        For tuning curves, not including spontaneous."""

        keys = list(self.nwb_data['stimulus']['presentation'].keys())
        stim_table_full=pd.DataFrame()
        for key in keys:
            if key!='spontaneous':
                temp1 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['data'].value, columns = self.nwb_data['stimulus']['presentation'][key]['features'].value)
                temp2 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start','End'])
                temp = temp2.join(temp1)
                stim_table_full=stim_table_full.append(temp)
                del temp1, temp2
        
        if 'Contrast' in list(stim_table_full.keys()) and 'Color' in list(stim_table_full.keys()):     
            print('True')   
            stim_table_full['con']=stim_table_full['Contrast'].values*stim_table_full['Color'].values

        stim_table_full = stim_table_full.sort('Start')

        return stim_table_full

    def get_spike_times(self, probename, unit):
        return np.array(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['times'].value)  

    def get_ISI(self, stim_table):
        """Inter-stimulus interval for consistent size of binary array. Because there is 
        slight variability in ISI. """
        if 'End' in list(stim_table.keys()):
            ISI = round(np.median(stim_table['End'].values - stim_table['Start'].values), 3)
        else:
            ISI = round(np.median(np.diff(stim_table['Start'].values)), 3)     
        return ISI   

    def get_interval(self, stim_table):
        if 'End' in list(stim_table.keys()):
            self.interval = np.mean(stim_table.Start[1:].values-stim_table.End[:-1].values)
        else:
            self.interval = 0
        return self.interval

    def get_binarized(self, probename, unit, stim_table, pre_time=0., post_time=0., fs = 1000.):
        """Binarize spike trains with 1/fs ms window.
        """
        ISI = self.get_ISI(stim_table)
        sync_start = stim_table['Start']-pre_time
        sync_end = stim_table['Start']+ISI+post_time

        for i in range(20):
            time_range = int(round(max(sync_end.values-sync_start.values)*fs)+i)
            if time_range%10 ==0:
                #print(time_range)
                #print(i,temp)
                break

        spike_times = self.get_spike_times(probename, unit)
        time_repeats = []
        spike_times = spike_times*fs # convert to ms
        for i,t in enumerate(sync_start*fs):
            temp = spike_times[find_range(spike_times,t,t+time_range)]-t  
            time_repeats.append(temp/fs)

        time_repeats=np.array(time_repeats)

        binarized = []
        for trial in time_repeats:
            binarized_ = np.zeros(time_range) # variability in lenth
            for spike in trial:
                spike_t = int(np.floor(spike*fs))
                if spike_t<time_range:
                    binarized_[spike_t] = 1
            binarized.append(binarized_)   
        bin_times = 1./fs*np.arange(time_range)
        return time_repeats, binarized, bin_times

    def get_binarized_one(self, probename, unit, stim_table, pre_time=0., post_time=0., fs = 1000.):
        """Binarize spike trains with 1/fs ms window.
        """
        sync_start = stim_table.Start.values[0]-pre_time
        sync_end = stim_table.End.values[-1]+post_time

        time_range = int(round((sync_end-sync_start)*fs))

        spike_times = self.get_spike_times(probename, unit)
        spike_times = spike_times*fs # convert to ms

        time_repeats = spike_times[find_range(spike_times,sync_start*fs,sync_start*fs+time_range)]-sync_start*fs  
        time_repeats=np.array(time_repeats)

        binarized_ = np.zeros(time_range) # variability in lenth
        for spike in time_repeats:
            spike_t = int(np.floor(spike*fs))
            if spike_t<time_range:
                binarized_[spike_t] = 1
        return binarized_

    def get_binarized_spon(self, probename, unit, sync_start, sync_end,fs = 1000.):
        """For spontaneous activity, for given single start and end time. 
        Convert timebase to ms.
        """
        for i in range(20):
            time_range = int(round(sync_end-sync_start)*fs)+i
            if time_range%10 ==0:
                #print(time_range)
                #print(i,temp)
                break

        spike_times = self.get_spike_times(probename, unit)

        t = sync_start*fs
        spike_times = spike_times*fs
        temp = spike_times[find_range(spike_times,t,t+time_range)]-t

        binarized_ = np.zeros(time_range) # variability in lenth
        for spike in temp:
            spike_t = int(np.floor(spike))
            if spike_t<time_range:
                binarized_[spike_t] = 1
        return binarized_

    def get_probe(self, probename, key, pre_time=0., post_time=0., fs = 1000.):
        stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        print(probename)
        unit_binarized = []
        unit_list = self.nwb_data['processing'][probename]['unit_list'].value
        channel_list = self.nwb_data['processing'][probename]['channel_list'].value

        for idx, unit in enumerate(unit_list):
            time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
            unit_binarized.append(binarized)
        unit_binarized = np.array(unit_binarized)
        return unit_binarized, unit_list, channel_list

    def get_probe_V1(self, probename, key, wm, stim_table=[], pre_time=0., post_time=0., fs = 1000.):
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        print(probename)
        unit_binarized = []
        unit_list = self.nwb_data['processing'][probename]['unit_list'].value
        channel_list = self.nwb_data['processing'][probename]['channel_list'].value
        channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
        

        unit_list_V1=[]
        channel_list_V1=[]
        channel_ypos_V1=[]
        cell_type_V1=[]
        snr_v1=[]

        for idx, unit in enumerate(unit_list):
            ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
            cell_type = str(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['type'].value)
            if ypos>wm:
                #print(idx, unit)
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
                unit_list_V1.append(unit_list[idx])
                channel_list_V1.append(channel_list[idx])
                channel_ypos_V1.append(ypos)
                cell_type_V1.append(cell_type)
                if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                    snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                    #print(idx, unit, snr)
                    snr_v1.append(snr)          

        # todo: can be replaced with xarray to label matrix
        unit_binarized = np.array(unit_binarized)
        unit_list_V1=np.array(unit_list_V1)
        channel_list_V1=np.array(channel_list_V1)
        channel_ypos_V1=np.array(channel_ypos_V1)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_list_V1, channel_ypos_V1, cell_type_V1, snr_v1

    def get_matrix_V1(self, key, pre_time=0., post_time=0., wm_given=[], stim_table=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        probenames = list(self.nwb_data['processing'].keys())
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        for probename in probenames:
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            if len(wm_given)==0:
                wm = self.get_wm(probename)
            else:
                wm=wm_given[0]

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
                    if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                        snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                        snr_v1.append(snr) 
                    
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        waveforms = np.array(waveforms)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_V1, waveforms, snr_v1

    def get_matrix_V1_sortch(self, key, pre_time=0., post_time=0., wm_given=[], stim_table=[], probenames=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        for probename in probenames:
            print(probename)
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            # convert to np array and flatten()
            unit_list=np.array(unit_list).flatten()
            channel_ypos=np.array(channel_ypos).flatten()
            channel_list=np.array(channel_list).flatten()

            # sort according to channel/depth
            unit_list = unit_list[np.argsort(channel_list)]
            channel_ypos = channel_ypos[np.argsort(channel_list)]
            channel_list = channel_list[np.argsort(channel_list)]


            if len(wm_given)==0:
                wm = self.get_wm(probename)
            else:
                wm=wm_given[0]
            print(wm)

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
                    if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                        snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                        snr_v1.append(snr) 
                    
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        waveforms = np.array(waveforms)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_V1, waveforms, snr_v1

    def get_matrix_V1_sortch_snr(self, key, pre_time=0., post_time=0., wm_given=[], stim_table=[], probenames=[], snr_threshold=1.5, fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time.
        cancatenate all data in cortex from different probes
        sort by depth
        threshold with snr
        """
        # 
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        for probename in probenames:
            print(probename)
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            # convert to np array and flatten()
            unit_list=np.array(unit_list).flatten()
            channel_ypos=np.array(channel_ypos).flatten()
            channel_list=np.array(channel_list).flatten()

            # sort according to channel/depth
            unit_list = unit_list[np.argsort(channel_list)]
            channel_ypos = channel_ypos[np.argsort(channel_list)]
            channel_list = channel_list[np.argsort(channel_list)]
            print(len(unit_list))


            if len(wm_given)==0:
                wm = self.get_wm(probename)
            else:
                wm=wm_given[0]
            print(wm)

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                        snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                    if snr>=snr_threshold:
                        time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                        unit_binarized.append(binarized)
                        unit_list_V1.append(unit)
                        channel_V1.append(channel_list[idx])
                    
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        return unit_binarized, unit_list_V1, channel_V1

    def get_matrix_V1_sortch_df(self, key, pre_time=0., post_time=0., wm_given=[], stim_table=[], probenames=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        probe_ids=[]
        for probename in probenames:
            print(probename)
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            # convert to np array and flatten()
            unit_list=np.array(unit_list).flatten()
            channel_ypos=np.array(channel_ypos).flatten()
            channel_list=np.array(channel_list).flatten()

            # sort according to channel/depth
            unit_list = unit_list[np.argsort(channel_list)]
            channel_ypos = channel_ypos[np.argsort(channel_list)]
            channel_list = channel_list[np.argsort(channel_list)]


            if len(wm_given)==0:
                wm = self.get_wm(probename)
            else:
                wm=wm_given[0]
            print(wm)

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
                    probe_ids.append(probename)
                    if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                        snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                        snr_v1.append(snr) 
                    
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        waveforms = np.array(waveforms)
        snr_v1=np.array(snr_v1)
        probe_ids=np.array(probe_ids)

        df = pd.DataFrame()
        df['unit_id']=unit_list_V1
        df['channel_id']=channel_V1
        df['probe_id']=probe_ids

        return unit_binarized, df, waveforms

    def get_matrix_with_df(self, key, df, pre_time=0., post_time=0., stim_table=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        probenames = df.probe_id.unique()
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        for probename in probenames:
            df_probe=df[df.probe_id==probename]
            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(df_probe.unit_id.values):
                if unit in self.nwb_data['processing'][probename]['unit_list'].value:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                else:
                    print(probename, unit, idx)
                    
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        return unit_binarized


    def get_ISI_V1_sortch(self, select=False, wm_given=[], key='',  fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if select==True:
            stim_table = self.get_stim_table(key=key)
            stimulus_gap=stim_table.Start.values[1:]-stim_table.End.values[:-1]
            if len(stim_table)==0:
                stim_table = self.get_stim_table(key=key)
                print('recompute stim table for key')

        probenames = list(self.nwb_data['processing'].keys())
        print(probenames)
        keys = list(self.nwb_data['stimulus']['presentation'].keys())

        #auto = []
        ISI = []
        unit_list_V1 = []
        channel_V1 = []
        snr_v1=[]
        for probename in probenames:
            print(probename)
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            # sort according to channel/depth
            unit_list = unit_list[np.argsort(channel_list)]
            channel_ypos = channel_ypos[np.argsort(channel_list)]
            channel_list = channel_list[np.argsort(channel_list)]

            if len(wm_given)==0:
                wm = self.get_wm(probename)
            else:
                wm=wm_given[0]
            print(wm)

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    # spike times for each unit
                    times = self.get_spike_times(probename, unit)
                    if 'Opto' in keys:
                        # remove opto tagging to avoid artifact in spike times
                        stim_table = self.get_stim_table(key='Opto')
                        times = times[np.where(times<stim_table.Start.values[0])[0]]
                    
                    if select==True:
                        # select times according to stim_table
                        times_selected=[]
                        for idx, start in enumerate(stim_table.Start):
                            end = stim_table.End[idx]
                            if end-start>0:
                                times_selected.append(times[np.where((times>=start) & (times<end))[0]])
                        times_selected = np.array([item for sublist in times_selected for item in sublist])
                        #auto.append(autocorr(times_selected))
                        ISI.append(inter_spike_interval(times_selected))
                    else:
                        #auto.append(autocorr(times))
                        ISI.append(inter_spike_interval(times))
                    
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)].keys()):
                        snr=self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['snr'].value
                        snr_v1.append(snr) 

        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        snr_v1=np.array(snr_v1)

        S=[]
        for i in range(len(ISI)):
            # larger range for ISI, bias the skewness value to higher
            hist, bins = np.histogram(ISI[i],bins=np.arange(0,0.06,0.001))
            # normalized sum hist to 1
            plt.step(np.arange(0,0.06,0.001), np.concatenate(([hist[0]],hist),axis=0)/float(sum(hist)))
            S.append(skew(hist/float(sum(hist))))
        S=np.array(S)
                            
        return ISI, S, unit_list_V1, channel_V1, snr_v1

    def get_matrix_V1_depth(self, key, wm_list, pre_time=0., post_time=0., fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        probenames = list(self.nwb_data['processing'].keys())
        stim_table = self.get_stim_table(key=key)

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        for probename in probenames:
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_ypos = self.nwb_data['processing'][probename]['channel_ypos'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            wm = wm_list[probename]

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            unit_list_V1_tmp = []
            channel_V1_tmp = []
            for idx, unit in enumerate(unit_list):
                ypos = self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['ypos_probe'].value
                if ypos>wm:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    
        unit_binarized = np.array(unit_binarized)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        return unit_binarized, unit_list_V1, channel_V1

    
    def get_matrix_all(self, key, pre_time=0., post_time=0., fs = 1000.):
        stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        probenames = list(self.nwb_data['processing'].keys())
        matrix=np.array([])
        matrix_unit=np.array([])
        matrix_channel=np.array([])
        for probename in probenames:
            print(probename)
            unit_binarized = []
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            for idx, unit in enumerate(unit_list):
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
            unit_binarized = np.array(unit_binarized)
            
            if len(matrix)==0:
                matrix=unit_binarized
                matrix_unit=unit_list
                matrix_channel=channel_list
            else:
                matrix = np.concatenate((matrix, unit_binarized),axis=0)
                matrix_unit = np.concatenate((matrix_unit, unit_list),axis=0)
                matrix_channel = np.concatenate((matrix_channel, channel_list),axis=0)
        return matrix, matrix_unit, matrix_channel

    def get_matrix_all_spon(self, key, pre_time=0., post_time=0., fs = 1000.):
        """To do: add field to select time window for spontaneous activity."""
        stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        probenames = list(self.nwb_data['processing'].keys())
        matrix=np.array([])
        matrix_unit=np.array([])
        matrix_channel=np.array([])
        for probename in probenames:
            print(probename)
            unit_binarized = []
            unit_list = self.nwb_data['processing'][probename]['unit_list'].value
            channel_list = self.nwb_data['processing'][probename]['channel_list'].value

            for idx, unit in enumerate(unit_list):
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
            unit_binarized = np.array(unit_binarized)
            
            if len(matrix)==0:
                matrix=unit_binarized
                matrix_unit=unit_list
                matrix_channel=channel_list
            else:
                matrix = np.concatenate((matrix, unit_binarized),axis=0)
                matrix_unit = np.concatenate((matrix_unit, unit_list),axis=0)
                matrix_channel = np.concatenate((matrix_channel, channel_list),axis=0)
        return matrix, matrix_unit, matrix_channel


    def get_trials(self, unit_binarized, stim_table):
        """TO DO"""
        keys = [i for i in list(stim_table.keys()) if i not in ['Start', 'End']]
        dim=[]
        n_unique=[]
        for key in keys:
            dim.append(np.unique(stim_table[key]))
            n_unique.append(len(np.unique(stim_table[key])))

        rep=len(stim_table)/reduce(lambda x, y: x*y, n_unique)   
        
        spikes = np.zeros((np.shape(unit_binarized)[0], 2, 8, rep, np.shape(unit_binarized)[2]))

        oris = np.unique(stim_table['Ori'])
        cons = np.unique(stim_table['Contrast'])
        for i, c in enumerate(cons):
            for j, o in enumerate(oris):
                tmp = unit_binarized[:,np.where((stim_table['Ori']==o) & (stim_table['Contrast']==c))[0],:]
                spikes[:,i,j,:,:] = tmp
        return spikes


    def get_PSTH(self, unit_binarized, PSTH_bintime, fs=1000):
        """Calculate PSTH averaged across trials based on binarized spike trains.
        unit_binarized: units*rep*time
        PSTH_bintime: 
        BUG if function in object! why? tuple indexing error
        """
        PSTH_binsize = PSTH_bintime/1000.*fs
        sizes = np.shape(unit_binarized)
        unit_psth=np.zeros([sizes[0],sizes[1],int(sizes[2]/PSTH_binsize)])
        
        for u in range(sizes[0]):
            for r in range(sizes[1]):
                data = unit_binarized[u,r,:]
                psth = np.sum(np.reshape(data,(int(sizes[2]/PSTH_binsize),int(PSTH_binsize))),axis=1)
                unit_psth[u,r]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.
        return unit_psth, time

    def get_PSTH_select(self, binarized, rep_select, time_select, step=2.):
        """unit_binarized: units*rep*time
        rep_select: select trials for specific conditions
        time_select: select time bins based on sampling rate (fs) for unit_binarized.
        step: unit ms.
        psth_select: units*
        """
        psth = binarized[:,rep_select,time_select]
        t = np.arange(-100,np.shape(psth)[2]-100,step)+step/2
        psth_mean = []
        psth_std = []
        for i in range(np.shape(psth)[0]):
            psth_temp = np.squeeze(psth[i,:,:])
            tmp = np.sum(psth_temp.reshape(np.shape(psth_temp)[0],
                                               np.shape(psth)[2]/step,step)/(step/np.shape(psth)[2]),axis=2)
            psth_mean.append(np.nanmean(tmp, axis=0))
            psth_std.append(np.nanstd(tmp, axis=0))
        psth_mean=np.array(psth_mean)
        psth_std=np.array(psth_std)
        return psth_mean, psth_std


    def plot_raster(self, probename, unit, sync_start, sync_end, fs=1000., plot='all_trials'):
        """Chunck spike time with sync pulse and plot raster.
        Time base is in seconds. plot determine whether plot all trials for a single unit or plot one trial"""
        spike_times = self.get_spike_times(probename, unit)
        for i in range(20):
            time_range = (round(max(sync_end.values-sync_start.values)*fs)+i)
            if time_range%10 ==0:
                break
                
        spike_times = spike_times*fs # convert to ms
        for i,t in enumerate(sync_start*fs):
            temp = spike_times[find_range(spike_times,t,t+time_range)]-t  

            # plot raster for desired trial for a given unit
            if plot=='all_trials':
                plt.plot(temp, np.ones(len(temp))*(i+1),'.',linewidth=3,c='k')
                plt.ylim([-1,len(sync_start)])
                plt.tick_params(direction='out')
                plt.xlabel('Time (sec)')
                plt.ylabel('Trials')       

    





