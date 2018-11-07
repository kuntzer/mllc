import logging

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

tmllc.data.generateFakeData(ntransits=9000, nnontransit=9000, sigmaPhoton=200e-6, saveDir="data/fakeLarge/")