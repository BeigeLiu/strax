import strax
import strax.CAENReader_2 as CAENReader
import numpy as np
from .plugin import Plugin, SaveWhen
import os
from tqdm import tqdm
import sys
##### One Channel Only 
##### TODO： multi channel reader plugin
import time
@strax.takes_config(
    strax.Option('crash', type=bool, default=False),
    strax.Option('n_triggers', type=int, default=500, track=False),
    strax.Option('n_chunks',type=int,default=50000,track=False),
    strax.Option('read_all_samples',type=bool,default=False),
    strax.Option('recs_per_chunk', type=int, default=800, track=False),
    strax.Option('filename',type=str,default=r"C:\Users\lbg\Desktop\run00001_raw_b0_seg0.bin",track=False)
)


class CAENRecords_2(Plugin):
    provides = 'raw_records'
    parallel = 'process'
    depends_on      = ()
    dtype           = strax.raw_record_dtype()
    if_the_first    = True
    run_data        = CAENReader.DataFile
    pre_end_time    = 0
    if_empty_time   = False
    empty_samples   = 0
    abs_time        = 0
    def setup(self):
        self.run_data    = CAENReader.DataFile(fileName=self.config['filename'])
        pre,current,next = self.run_data.getNextTrigger(mode='first')
        self.abs_time    = pre['time']
        self.pre_end_time= 0
        pre['time']     -= self.abs_time
        current['time'] -= self.abs_time
        next['time']    -= self.abs_time
        #pre,current,next = self.run_data.getNextTrigger()
        self.if_the_first = False
        return super().setup()

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        if not self.config['read_all_samples']:
            return chunk_i < self.config['n_chunks']
        else:
            return not self.run_data.is_finish#chunk_i < self.config['n_chunks']

    def check_filename(self):
        if self.config['filename']=='':
            raise ValueError(' Path of DAQ data file is not given')
        if not os.path.exists(self.config['filename']):
            raise FileExistsError(self.config['filename']+'is not found')
    def compute(self,chunk_i):
        # if self.if_the_first:
        #     self.run_data    = CAENReader.DataFile(fileName=self.config['filename'])
        #     pre,current,next = self.run_data.getNextTrigger(mode='first')
        #     self.abs_time    = pre['time']
        #     self.pre_end_time= 0
        #     pre['time']     -= self.abs_time
        #     current['time'] -= self.abs_time
        #     next['time']    -= self.abs_time
        #     #pre,current,next = self.run_data.getNextTrigger()
        #     self.if_the_first = False
        # else:
        if self.if_empty_time:
            pre,current,next     = self.run_data.getNextTrigger(mode='frozen')
            pre['time']     -= self.abs_time
            current['time'] -= self.abs_time
            next['time']    -= self.abs_time
        else:
            pre,current,next     = self.run_data.getNextTrigger(mode='next')
            pre['time']     -= self.abs_time
            current['time'] -= self.abs_time
            next['time']    -= self.abs_time

        start = self.pre_end_time
        end   = start + self.config['recs_per_chunk']*current['dt'][0]
        # print('===================')
        # print('start',start)
        # print('end',end)
        # print('current',current['time'][0],current['end'][0])
        # print('pre',pre['time'][0],pre['end'][0])
        # print('next',next['time'][0],next['end'][0])
        # print('===================')
        if current['time'][0] > end:
            # print('=======case 1==========')
            r = np.zeros(pre['end'].shape[0],dtype=strax.raw_record_dtype(samples_per_record=self.config['recs_per_chunk']))
            r['time']   = start
            r['dt']     = int((current['time'][0] - end)/self.config['recs_per_chunk'])+1
            r['length'] = self.config['recs_per_chunk']
            r['pulse_length'] = self.config['recs_per_chunk']
            r['record_i'] = chunk_i
            #r['data']    += np.expand_dims(np.transpose(current['baseline']),axis = 1)
            #print(r['data'])
            r['channel']  = np.linspace(0,r.shape[0]-1,r.shape[0])
            end = current['time'][0]
            self.if_empty_time  = True
            self.empty_samples += 0
        elif start <= current['time'][0] and current['end'][0] <= end:
            ##### the chunk fully contains the trigger
            if start <= pre['end'][0] or next['time'][0] <= end:
                ##### the chunk have overlaps with pre and next
                pre_overlap  = int((pre['end'][0]-start)/current['dt'][0])
                next_overlap = int((end['time'][0] - end)/current['dt'][0])
                r = np.zeros(next.shape[0],dtype=strax.raw_record_dtype(samples_per_record=self.config['recs_per_chunk']))
                r['time'] = pre['end']
                r['dt']   = current['dt']
                r['length'] = self.config['recs_per_chunk']
                r['channel']  = np.linspace(0,r.shape[0]-1,r.shape[0])
                r['data']    += np.expand_dims(np.transpose(current['baseline']),axis = 1)
                r['data']     = np.concatenate(
                    [pre['data'][:,pre_overlap:],current['data'],next['data'][:,:next_overlap]]
                )

            elif start > pre['end'][0] and next['time'][0] > end:
                ##### chunk do not overlap with pre and next
                samples_before  = int((start - current['time'][0])/current['dt'][0])
                samples_after   = int(self.config['recs_per_chunk']-(end-current['end'][0])/current['dt'][0])
                r = np.zeros(next.shape[0],dtype=strax.raw_record_dtype(samples_per_record=self.config['recs_per_chunk']))
                r['time']     = start
                r['dt']       = current['dt']
                r['length']   = self.config['recs_per_chunk']
                r['channel']  = np.linspace(0,r.shape[0]-1,r.shape[0])
                r['data']    += np.expand_dims(np.transpose(current['baseline']),axis = 1)
                r['data'][:,samples_before:samples_after]   = current['data']
                r['record_i'] = chunk_i
                if samples_after>self.config['recs_per_chunk']:
                    r['pulse_length'] = self.config['recs_per_chunk']
                else:
                    r['pulse_length'] = current['data'].shape[1]
                r['baseline']     = current['baseline']
                self.if_empty_time = False
                del pre
                del current
                del next
        # print('==========')
        # print('chunk start:',r['time'],' end:',end)
        # print('pulse start:',r['time'],' end:',strax.endtime(r))
        self.pre_end_time = end
        # print('pre_end:',self.pre_end_time)
        chunk = self.chunk(start=start,end = end,data = r)
        # print(chunk_i)
        # print(chunk)
        # print(r['data'].shape)
        # print(r['data'])
        # print('==========')
        if not self.config['read_all_samples']:
            i = int(chunk_i/self.config['n_chunks']*100)
        else: 
            i = int(chunk_i/self.run_data.total_event*100)
        print("\r", end="")
        print("Creating raw record source: {}%: ".format(i), "▓" * (i // 2), end="") 
        sys.stdout.flush() 
        return chunk