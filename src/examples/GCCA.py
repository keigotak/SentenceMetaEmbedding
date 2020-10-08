from gcca.gcca import GCCA
import logging
import numpy as np

# set log level
logging.root.setLevel(level=logging.INFO)

# create data in advance
a = np.random.rand(50, 50)
b = np.random.rand(50, 60)
c = np.random.rand(50, 70)
d = np.random.rand(50, 80)
e = np.random.rand(50, 90)
f = np.random.rand(50, 100)
g = np.random.rand(50, 110)
h = np.random.rand(50, 120)
i = np.random.rand(50, 130)
j = np.random.rand(50, 140)
k = np.random.rand(50, 150)

# create instance of GCCA
gcca = GCCA()
# calculate GCCA
gcca.fit(a, b, c, d, e, f, g, h, i, j, k)
# transform
gcca.transform(a, b, c, d, e, f, g, h, i, j, k)
# save
gcca.save_params("../../models/example_cca.h5")
# load
gcca.load_params("../../models/example_cca.h5")
# plot
gcca.plot_result()