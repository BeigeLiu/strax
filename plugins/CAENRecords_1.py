import strax
import strax.CAENReader_1 as CAENReader
import numpy as np
from .plugin import Plugin, SaveWhen
import os
from tqdm import tqdm
##### One Channel Only 
##### TODO： multi channel reader plugin
@strax.takes_config(
    strax.Option('crash', type=bool, default=False),
    strax.Option('n_triggers', type=int, default=500, track=False),
    strax.Option('n_chunks',type=int,default=2000,track=False),
    strax.Option('read_all_samples',type=bool,default=False),
    strax.Option('recs_per_chunk', type=int, default=1200, track=False),
    strax.Option('filename',type=str,default=r"C:\Users\lbg\Desktop\CAENUnpack\LEDS1S2\40us_1.25V40ns_2.40V20kHz_raw_b0_seg0.bin",track=False)
)


class CAENRecords_1(Plugin):
    provides = 'raw_records'
    parallel = 'process'
    depends_on      = ()
    dtype           = strax.raw_record_dtype()
    rechunk_on_save = False
    start           = True
    run_data        = CAENReader.DataFile
    end             = 10000
    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.end

    def check_filename(self):
        if self.config['filename']=='':
            raise ValueError(' Path of DAQ data file is not given')
        if not os.path.exists(self.config['filename']):
            raise FileExistsError(self.config['filename']+'is not found')
    def compute(self,chunk_i):
        if self.start:
            self.run_data = CAENReader.DataFile(self.config)
            self.start    = False
            self.end      = self.run_data._length()
            print(self.end)
        print(chunk_i)
        r                 = self.run_data.getNextChunk(chunk_i)
        start             = r['time']
        end               = r['time']+r['length']*r['dt']
        chunk = self.chunk(start=start[0],end = end[0],data = r)
        return chunk
