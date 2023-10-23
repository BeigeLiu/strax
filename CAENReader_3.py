

from numpy import nan, fromfile, dtype
from os import path
import numpy as np
import sys
from sys import getsizeof
import os
import strax
import gc
import time
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




def raw_record_dtype(samples_per_record=100):
    """Data type for a waveform raw_record.

    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
    """
    return interval_dtype + [
        # np.int16 is not enough for some PMT flashes...
        (('Length of pulse to which the record belongs (without zero-padding)',
            'pulse_length'), np.int32),
        (('Fragment number in the pulse',
            'record_i'), np.int64),
        (('Baseline determined by the digitizer (if this is supported)',
            'baseline'), np.int16),
        # Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        (('Waveform data in raw ADC counts',
            'data'), np.int16, samples_per_record)]
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
        self.recordLen       = 400
        self.DAQ = DAQ
        self.num_of_pre_samples = 100
        self.dt              = 4 # in ns
        self.time_log        = []

        ##### initial the trigger list by zeros
        self.numChannels     = 8
        self.is_the_first    = True
        self.total_event     = 0
        
        ##### reserve the time in case the char in digitizer roll over
        self.timeTagRollover = 0
        self.oldTimeTag      = 0 
        self.filepos         = 0
        self.triggers        = self.UnpackAll()
        self.is_finish       = False
        self.begin_time      = 0
        return
    def _splittrigger(self,triggers):
        splited_triggers = []
        for trigger in triggers:
            split_into       = int(trigger['pulse_length']//self.recordLen + 1)
            for i in range(split_into):
                this_trigger             = np.zeros(1,dtype=raw_record_dtype(samples_per_record=self.recordLen))
                this_trigger['data']    += trigger['baseline']
                this_trigger['time']     = trigger['time'] + self.recordLen *trigger['dt']*i
                this_trigger['channel']  = trigger['channel']
                this_trigger['baseline'] = trigger['baseline']
                this_trigger['record_i'] = self.total_event + i
                this_trigger['dt']       = trigger['dt']
                this_trigger['length']   = self.recordLen
                this_data                = trigger['data'][0,self.recordLen*i:self.recordLen*(i+1)]
                this_trigger['pulse_length']                = len(this_data)
                this_trigger['data'][0,:len(this_data)]     = this_data
                splited_triggers.append(this_trigger)
        self.total_event        += split_into
        del triggers
        return splited_triggers
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
        numChannels  = np.sum(whichChan)
        # initialize the trigger array
        triggers     = []
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
                this_trigger                 = np.zeros(1,dtype=raw_record_dtype(samples_per_record=10*self.recordLen))
                this_trigger['time']         = triggerTime*1e3# - self.num_of_pre_samples*self.dt # in ns
                this_trigger['dt']           = self.dt
                this_trigger['pulse_length'] = int(2*(chsize-3))
                this_trigger['length']       = self.recordLen
                this_trigger['channel']      = ind-1
                this_trigger['baseline']     = fromfile(self.file, dtype=dt, count=2)[1]
                this_trigger['data']        += this_trigger['baseline']
                #  firstly, zero padding 
                this_trigger['data'][0,:int(2*(chsize-3))] = np.expand_dims(fromfile(self.file, dtype=dt, count=int(2*(chsize-3))),
                                                                            axis = 0)
                triggers.append(this_trigger)
                
        triggers = self._splittrigger(triggers=triggers)
        return triggers,self.file.tell()
    def UnpackAll(self):
        totalsize       = os.path.getsize(self.fileName)
        pos             = 0
        triggers        = []
        while pos < 0.989*totalsize:
            i = int(pos*100/totalsize)    
            print("\r", end="")
            i = int(pos*100/totalsize)  
            print("Unpacking raw record file: {}%: ".format(i), "▓" * (i // 2), end="") 
            sys.stdout.flush() 
            # t1 = time.time()
            trigger,pos = self._getNextTrigger()
            for t in trigger:
                triggers.append(t)
            # self.time_log.append(time.time()-t1)
            del trigger
        triggers = np.array(triggers).reshape(-1)
        self.begin_time = min(triggers['time'])
        self.end_time   = max(strax.endtime(triggers))
        # import pickle as pkl
        # pkl.dump(self.time_log,open(r'C:\Users\lbg\Desktop\CAENUnpack\straxen-master\time_log.pkl','wb+'))
        return triggers
    def get(self,time_range,chunk_i):
        result = self.triggers[(self.triggers['time']<=int(time_range*(chunk_i+1)+self.begin_time))&\
                               (self.triggers['time']>int(time_range*(chunk_i)+self.begin_time))]
        result = strax.sort_by_time(result)
        return result
    


