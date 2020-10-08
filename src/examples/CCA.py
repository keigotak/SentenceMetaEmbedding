from gcca.cca import CCA
import logging
import numpy as np

# set log level
logging.root.setLevel(level=logging.INFO)

# create data in advance
a = np.random.rand(50, 50)
b = np.random.rand(50, 60)

# create instance of CCA
cca = CCA()
# calculate CCA
cca.fit(a, b)
# transform
cca.transform(a, b)
# transform by PCCA
cca.ptransform(a, b)
# save
cca.save_params("../../models/example_cca.h5")
# load
cca.load_params("../../models/example_cca.h5")
# plot
cca.plot_result()