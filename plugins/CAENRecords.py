import strax
import strax.CAENReader as CAENReader
import numpy as np
from .plugin import Plugin, SaveWhen
import os
##### One Channel Only 
##### TODO： multi channel reader plugin
@strax.takes_config(
    strax.Option('crash', type=bool, default=False),
    strax.Option('n_chunks', type=int, default=500, track=False),
    strax.Option('read_all_samples',type=bool,default=False),
    strax.Option('recs_per_chunk', type=int, default=1000, track=False),
    strax.Option('filename',type=str,default=r'C:\Users\lbg\Desktop\CAENUnpack\LEDS1S2\50us_1.27V36ns_2.40V15kHz_raw_b0_seg0.bin',track=False)
)


class CAENRecords(Plugin):
    provides = 'raw_records'
    parallel = 'process'
    depends_on      = ()
    dtype           = strax.record_dtype()
    last_end        = 6000
    rechunk_on_save = False
    last_filePos    = 0
    run_data        = CAENReader.DataFile
    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        print(chunk_i)
        if self.config['read_all_samples']:
            return os.path.getsize(self.config['filename']) > self.last_filePos+1
        else:
            return chunk_i < self.config['n_chunks']

    def check_filename(self):
        if self.config['filename']=='':
            raise ValueError(' Path of DAQ data file is not given')
        if not os.path.exists(self.config['filename']):
            raise FileExistsError(self.config['filename']+'is not found')
    def compute(self):
        if self.last_filePos == 0:
            self.run_data = CAENReader.DataFile(fileName=self.config['filename'])
        r,filePos         = self.run_data.getNextChunk(self.config)
        self.last_filePos = filePos
        start             = r['time']
        end               = r['time']+r['length']*r['dt']
        self.last_end     = end[0]
        chunk = self.chunk(start=start[0],end = end[0],data = r)
        return chunk
