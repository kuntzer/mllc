import logging

import tmllc

logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.INFO)

# Transforming the TESS files into big tables that are saved in pickle form
# TODO allow for a function to be applied to the data first!
tmllc.utils.fits2pickle(["planet", "none", "eb", "backeb"], timeColumn="TIME")