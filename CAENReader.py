
from numpy import nan, zeros, fromfile, dtype
from os import path
import numpy as np
import matplotlib.pylab as plt
import blosc
import strax

# =============================================
# ============= Data File Class ===============
# =============================================


class DataFile:
    def __init__(self, fileName, DAQ='WaveDump'):
        """
        Initializes the dataFile instance to include the fileName, access time,
        and the number of boards in the file. Also opens the file for reading.
        """
        self.fileName = path.abspath(fileName)
        self.file = open(self.fileName, 'rb')
        self.recordLen = 0
        self.oldTimeTag = 0.
        self.timeTagRollover = 0
        self.DAQ = DAQ
        self.lastendtime = 0
        self.dt          = 4
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
        traces            = np.asarray(traces)
        return traces,triggerTime
    def pack_to_chunk(self,traces,triggerTime,config):
        trace             = traces[0]
        length            = traces.shape[1]-6
        ###### reset last end time
        
        data_type         = strax.raw_record_dtype(samples_per_record=config['recs_per_chunk'])
        r                 = np.zeros(config['recs_per_chunk'], data_type)
        r['time']         = triggerTime*1e3
        r['dt']           = self.dt
        r['length']       = config['recs_per_chunk']
        r['baseline']     = trace[5]
        r['channel']      = 0
        if length < config['recs_per_chunk']:
            r['data'][0,:length]     = trace[6:] - trace[5]
        else:
            r['data']                = trace[6:config['recs_per_chunk']+6]
        #r['amplitude_bit_shift'] = 8
        filePos           = self.file.tell()
        return r,filePos

    def getNextChunk(self,config):
        traces,triggerTimeTag  = self.getNextTrigger()
        r,filePos              = self.pack_to_chunk(traces,triggerTimeTag,config)
        return r,filePos
            
    def MovefilePos(self,pos):
        self.file.seek(pos)
        return self.file.tell()


    def close(self):
        """
        Close the open data file. Helpful when doing on-the-fly testing
        """
        self.file.close()
