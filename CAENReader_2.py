

from numpy import nan, fromfile, dtype
from os import path
import numpy as np
import sys
import os
# import strax


time_dt_fields = [
    (('Start time since unix epoch [ns]',
      'time'), np.int64),
    # Don't try to make O(second) long intervals!
    (('Length of the interval in samples',
      'length'), np.int32),
    (('Width of one sample [ns]',
      'dt'), np.int16)]

# Base dtype for interval-like objects (pulse, hit)
interval_dtype = time_dt_fields + [
    (('Channel/PMT number',
        'channel'), np.int16)]

def CAEN_raw_record_dtype(sample_per_record = 100):
    """Data type for a waveform raw_record from CAEN Digitizer V1925.
    """
    return interval_dtype+[
            (('Baseline given by the digitizer','baseline'),np.int16),
            (('Digital to analog conversion coefficient','DACC'),np.float16),
            (('end of the trigger','end'),np.int64),
            (('Waveform data in samples','data'),np.int16,int(sample_per_record))]


# =============================================
# ============= Data File Class ===============
# =============================================

class DataFile:
    def __init__(self, fileName, DAQ='WaveDump'):
        """
        Initializes the dataFile instance to include the fileName, access time,
        and the number of boards in the file. Also opens the file for reading.
        """
        self.fileName        = path.abspath(fileName)
        self.file            = open(self.fileName, 'rb')
        self.recordLen       = 800
        self.DAQ = DAQ
        self.num_of_pre_samples = 100
        self.dt              = 4 # in ns

        ##### initial the trigger list by zeros
        self.numChannels     = 8
        self.is_the_first    = True
        self.if_cut          = False
        self.total_event     = 0
        self.pre_trigger     =  np.zeros(self.numChannels,
                                        dtype=CAEN_raw_record_dtype(self.recordLen))
        self.after_trigger   =  np.zeros(self.numChannels,
                                        dtype=CAEN_raw_record_dtype(self.recordLen))  
        self.current_trigger =  np.zeros(self.numChannels,
                                        dtype=CAEN_raw_record_dtype(self.recordLen))  
        
        self.cut_reserve     =  np.zeros(self.numChannels,
                                        dtype=CAEN_raw_record_dtype(self.recordLen)) 
        
        ##### reserve the time in case the char in digitizer roll over
        self.timeTagRollover = 0
        self.oldTimeTag      = 0 
        self.filepos  = 0
        self.iterator = iter(self.UnpackAll())
        self.is_finish = False
        return
    def _splittrigger(self,trigger):
        return 
    def _getNextTrigger(self):
        """
        This function returns  the next trigger from the dataFile. It reads the control words into h[0-3], unpacks them,
        and then reads the next event. It returns a RawTrigger object, which includes the fileName, location in the
        file, and a dictionary of the traces
        :raise:IOError if the header does not pass a sanity check: (sanity = 1 if (i0 & 0xa0000000 == 0xa0000000) else 0
        """
        # Read the 4 long-words of the event header
        try:
            i0, i1, i2, i3 = fromfile(self.file, dtype='I', count=4)
        except ValueError:
            return None

        # Check to make sure the event starts with the key value (0xa0000000), otherwise it's ill-formed
        sanity = 1 if (i0 & 0xa0000000 == 0xa0000000) else 0
        if sanity == 0:
            raise IOError('Read did not pass sanity check')

        # For 16ch boards, the channelMask is split over two header words, ref: Tab. 8.2 in V1730 manual
        # To get the second half of the channelMask to line up properly with the first, we only shift it by
        # 16bits instead of 24.
        channelUse = (i1 & 0x000000ff) + ((i2 & 0xff000000) >> 16)
        # The trigger time-tag (timestamp) is the entire fourth long-word
        triggerTimeTag   = i3

        # Since the trigger time tag is only 32 bits, it rolls over frequently. This checks for the roll-over

        if i3 < self.oldTimeTag:
            self.timeTagRollover += 1
            self.oldTimeTag = float(i3)
        else:
            self.oldTimeTag = float(i3)

        # correcting triggerTimeTag for rollover
        triggerTimeTag += self.timeTagRollover*(2**31)

        # convert from ticks to us since the beginning of the file
        triggerTime    = triggerTimeTag * 8e-3
        # convert channel map into an array of 0's or 1's indicating which channels are in use
        whichChan = [1 if (channelUse & 1 << k) else 0 for k in range(16)]
        # looping over the entries in the whichChan list, only reading data if the entry is 1
        fixed_length = 0
        # initialize the trigger array
        triggers     = np.zeros(self.numChannels,dtype=CAEN_raw_record_dtype(sample_per_record=self.recordLen))
        for ind, k in enumerate(whichChan):
            if k == 1: # whether the channel have data

                # create a data-type of unsigned 16bit integers with the correct ordering
                dt = dtype('<H')
                # read the size of this channel
                chsize  = fromfile(self.file,dtype=dt,count = 2)[0]
                if fixed_length<int(2*(chsize-3)):
                    ### find the biggest record length among channels
                    fixed_length = 2*(chsize-3)
                # read the trigger time stamp
                trtime  = fromfile(self.file,dtype='I',count = 1)[0]

                # convert from ticks to us since the beginning of the file
                trtime      = trtime * 8e-3
                this_trigger                 = np.zeros(1,dtype=CAEN_raw_record_dtype(sample_per_record=self.recordLen))
                this_trigger['time']         = triggerTime*1e3 - self.num_of_pre_samples*self.dt # in ns
                this_trigger['dt']           = self.dt
                this_trigger['length']       = int(2*(chsize-3))
                this_trigger['channel']      = ind-1
                this_trigger['baseline']     = fromfile(self.file, dtype=dt, count=2)[1]
                if self.recordLen >=  int(2*(chsize-3)):
                    ## in this case, zero padding 
                    this_trigger['data'][0,:int(2*(chsize-3))] = np.expand_dims(fromfile(self.file, dtype=dt, count=int(2*(chsize-3))),
                                                                              axis = 0)
                    this_trigger['data'][0,int(2*(chsize-3)):] = this_trigger['baseline'][0]
                    this_trigger['end']     = this_trigger['time']+this_trigger['dt']*self.recordLen 
                else:
                    ## in thit case cut off the trigger 
                    data                 = np.expand_dims(fromfile(self.file, dtype=dt, count=int(2*(chsize-3))),
                                                        axis = 0)
                    length                 = int(2*(chsize-3))
                    this_trigger['length'] = self.recordLen
                    this_trigger['data']   = data[0,:self.recordLen]
                    this_trigger['end']  = this_trigger['time']+this_trigger['dt']*self.recordLen 
              
                    ## then reserve what have been cut 
                    # self.cut_reserve['channel'][ind-1] = ind
                    # self.cut_reserve['data'][ind-1][:int(length-self.recordLen)] = data[0,self.recordLen:]
                    # self.cut_reserve['length'][ind-1]  = int(length-self.recordLen) 
                    # self.if_cut   = True
                triggers[ind-1]   = this_trigger
                triggers['time']  = this_trigger['time'][0] ### set all the channels are triggered at the same time
                triggers['dt']    = this_trigger['dt'][0]
                triggers['end']   = this_trigger['end'][0]
                self.total_event += 1
        return triggers,self.file.tell()
    def UnpackAll(self):
        totalsize       = os.path.getsize(self.fileName)
        pos             = 0
        triggers         = []
        while pos < 0.989999*totalsize:
            i = int(pos*100/totalsize)    
            print("\r", end="")
            i = int(pos*100/totalsize)  
            print("Unpacking raw record file: {}%: ".format(i), "▓" * (i // 2), end="") 
            sys.stdout.flush() 
            trigger,pos = self._getNextTrigger()
            triggers.append(trigger)
        return np.array(triggers)
    def getNextTrigger(self,mode):
        # if self.filepos<0.989999*totalsize
        if  mode == 'first':
            self.pre_trigger              = next(self.iterator)
            self.current_trigger           = next(self.iterator)
            self.after_trigger  = next(self.iterator)
            self.is_the_first    = False
            return self.pre_trigger,self.current_trigger,self.after_trigger
        elif mode == 'next':
            ###### move to next trigger
            self.pre_trigger     = self.current_trigger
            self.current_trigger = self.after_trigger
            self.after_trigger   = next(self.iterator,-1)
            if self.after_trigger == -1:
                self.is_finish = True

            return self.pre_trigger,self.current_trigger,self.after_trigger
        elif mode == 'frozen':
            return self.pre_trigger,self.current_trigger,self.after_trigger
    



# dataset = DataFile(fileName=r"C:\Users\lbg\Desktop\PMT_top_raw_b0_seg5.bin")
# time1 = 0
# for i in range(10):
#     _,trigger,_ = dataset.getNextTrigger(mode = 'frozen')
#     print(trigger['time']-time1)
#     time1 = trigger['time']


 