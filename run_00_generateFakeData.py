import logging

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

tmllc.data.generateFakeData(ntransits=16384, nnontransit=16384, sigmaPhoton=200e-6, saveDir="data/fakeWideParamsL/", maxDataPerFile=1024)
#tmllc.data.generateFakeData(ntransits=400, nnontransit=500, sigmaPhoton=200e-6, saveDir="data/fakeWideParams/", maxDataPerFile=200)