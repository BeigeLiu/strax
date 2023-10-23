

from numpy import nan, zeros, fromfile, dtype
from os import path
import numpy as np
import matplotlib.pylab as plt
import blosc
import strax
import time

# =============================================
# ============= Data File Class ===============
# =============================================


class DataFile:
    def __init__(self, config, DAQ='WaveDump'):
        """
        Initializes the dataFile instance to include the fileName, access time,
        and the number of boards in the file. Also opens the file for reading.
        """
        self.config   = config
        self.fileName = path.abspath(config['filename'])
        self.file     = open(self.fileName, 'rb')
        self.recordLen = 0
        self.oldTimeTag = 0.
        self.timeTagRollover = 0
        self.DAQ = DAQ
        self.lastendtime = 0
        self.dt          = 1
        self.pos         = 0
        self.num_of_activate_channel = 4
        if self.config['n_chunks'] != 0:
            self.data_list,self.baseline_list = self.getcontinuedata(num_of_triggers=self.config['n_chunks'],pos = 0)
    def getNextTrigger(self):

        """
        This function returns  the next trigger from the dataFile. It reads the control words into h[0-3], unpacks them,
        and then reads the next event. It returns a RawTrigger object, which includes the fileName, location in the
        file, and a dictionary of the traces
        :raise:IOError if the header does not pass a sanity check: (sanity = 1 if (i0 & 0xa0000000 == 0xa0000000) else 0
        """

        # Instantize a RawTrigger object
        #trigger = RawTrigger()
        # Fill the file position
        filePos = self.file.tell()

        # Read the 4 long-words of the event header

        try:
            i0, i1, i2, i3 = fromfile(self.file, dtype='I', count=4)
        except ValueError:
            return None

        # Check to make sure the event starts with the key value (0xa0000000), otherwise it's ill-formed
        sanity = 1 if (i0 & 0xa0000000 == 0xa0000000) else 0
        if sanity == 0:
            raise IOError('Read did not pass sanity check')

        # extract the event size from the first header long-word
        eventSize = i0 - 0xa0000000

        # extract the board ID and channel map from the second header long-word
        boardId = (i1 & 0xf8000000) >> 27

        # For 16ch boards, the channelMask is split over two header words, ref: Tab. 8.2 in V1730 manual
        # To get the second half of the channelMask to line up properly with the first, we only shift it by
        # 16bits instead of 24.
        channelUse = (i1 & 0x000000ff) + ((i2 & 0xff000000) >> 16)

        # convert channel map into an array of 0's or 1's indicating which channels are in use
        whichChan = [1 if (channelUse & 1 << k) else 0 for k in range(16)]

        # determine the number of channels that are in the event by summing whichChan
        numChannels = int(sum(whichChan))

        # Test for zero-length encoding by looking at one bit in the second header long-word
        zLE = True if i1 & 0x01000000 != 0 else False

        # Create an event counter mask and then extract the counter value from the third header long-word
        eventCounterMask = 0x00ffffff
        eventCounter     = i2 & eventCounterMask

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

        # Calculate length of each trace, using eventSize (in long words) and removing the 4 long words from the header
        size = int(4 * eventSize - 16)
        
        # initialize traces
        traces = []

        # looping over the entries in the whichChan list, only reading data if the entry is 1
        for ind, k in enumerate(whichChan):
            if k == 1:
                # create a name for each channel according to the board and channel numbers
                traceName = "b" + str(boardId) + "tr" + str(ind)
                # If the data are not zero-length encoded (default)
                if not zLE:
                    # create a data-type of unsigned 16bit integers with the correct ordering
                    dt = dtype('<H')

                    # Use numpy's fromfile to read binary data and convert into a numpy array all at once
                    trace = fromfile(self.file, dtype=dt, count=size//(2*numChannels))
                else:
                    # initialize an array of length self.recordLen, then set all values to nan
                    trace = zeros(self.recordLen)
                    trace[:] = nan

                    # The ZLE encoding uses a keyword to indicate if data to follow, otherwise number of samples to skip
                    (trSize,) = fromfile(self.file, dtype='I', count=1)

                    # create two counting indices, m and trInd, for keeping track of our position in the trace and
                    m = 1
                    trInd = 0

                    # create a data-type for reading the binary data
                    dt = dtype('<H')

                    # loop while the m counter is less than the total size of the trace
                    while m < trSize:
                        # extract the control word from the data
                        (controlWord,) = fromfile(self.file, dtype='I', count=1)

                        # determine the number of bytes to read, and convert into samples (x2)
                        length = (controlWord & 0x001FFFFF) * 2

                        # determine whether that which follows are data or number of samples to skip
                        good = controlWord & 0x80000000

                        # If they are data...
                        if good:

                            # Read and convert the data (length is
                            tmp = fromfile(self.file, dtype=dt, count=length)
                            # insert the read data into the empty trace otherwise full of NaNs
                            trace[trInd:trInd + length] = tmp

                        # Increment both the trInd and m indexes by their appropriate amounts
                        trInd += length
                        m += 1 + (length/2 if good else 0)

                # create a dictionary entry for the trace using traceName as the key
                traces.append(trace)
                traces.append(trace)
                traces.append(trace)
                traces.append(trace) ####!!!!!!!!!!!!!!!!!!!!!#### just for test !!!!!!!!!!
        traces            = np.asarray(traces,dtype=int)
        return traces,triggerTime
    def pack_to_chunk(self,chunk_i):
        trace             = self.data_list[chunk_i]
        ###### reset last end time
        
        data_type         = strax.raw_record_dtype(samples_per_record=self.config['recs_per_chunk'])
        r                 = np.zeros(self.num_of_activate_channel, data_type)
        r['time']         = chunk_i*self.config['recs_per_chunk']
        r['dt']           = self.dt
        r['length']       = self.config['recs_per_chunk']
        r['baseline']     = self.baseline_list[:,chunk_i]
        r['channel']      = np.linspace(0,self.num_of_activate_channel-1,self.num_of_activate_channel,dtype=int)
        r['data']         = np.copy(trace)
        r['record_i']     = chunk_i

        #print('chunk',chunk_i)
        #print(r)
        return r

    def getNextChunk(self,chunk_i):
        r              = self.pack_to_chunk(chunk_i)
        return r
    
    def getcontinuedata(self,num_of_triggers,pos):
        ###
        # input : 
        #         num_of_triggers: number of triggers
        #         pos            : file pointer position, default 0  
        # output: time-continue list of wave shape 
        ###
        self.file.seek(pos)
        pre_end_time    = 4000 
        trigger_list = np.zeros((self.num_of_activate_channel,0),dtype=int)
        baseline_list = np.zeros((self.num_of_activate_channel,0))
        t1 = time.time()
        for i in range(num_of_triggers):
            traces,begin_time  = self.getNextTrigger()
            baseline      = traces[:,5].reshape(self.num_of_activate_channel,-1)
            begin_time = begin_time + 4000
            if begin_time>pre_end_time or begin_time == 0:
                trace         = np.zeros((self.num_of_activate_channel,
                                         int((begin_time-pre_end_time)/self.dt)),
                                         dtype=int)+baseline
                trigger_list  = np.concatenate([trigger_list,trace],axis = 1)
                baseline_list = np.concatenate([baseline_list,baseline],axis = 1)
                pre_end_time = begin_time
            trace              = traces[:self.num_of_activate_channel,6:]
            end_time           = begin_time+self.dt*(trace.shape[0]-5) ## in ns
            trigger_list       = np.concatenate([trigger_list,trace],axis = 1)
            pre_end_time       = end_time
            baseline_list = np.concatenate([baseline_list,baseline],axis = 1)
        while trigger_list.shape[1]%self.config['recs_per_chunk'] != 0:
            shape = self.config['recs_per_chunk']-trigger_list.shape[1]%self.config['recs_per_chunk']
            zeros_pad          = np.zeros((self.num_of_activate_channel,shape),dtype=int)
            trigger_list       = np.concatenate([trigger_list,zeros_pad],axis = 1)
        trigger_list = trigger_list.reshape((-1,self.num_of_activate_channel,self.config['recs_per_chunk']))
        t2 = time.time()
        return trigger_list,baseline_list
    
    def _length(self):
        return len(self.data_list)
    def MovefilePos(self,pos):
        self.file.seek(pos)
        return self.file.tell()


    def close(self):
        """
        Close the open data file. Helpful when doing on-the-fly testing
        """
        self.file.close()
