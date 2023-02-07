import os
import numpy as np
from pyarrow.parquet import ParquetFile


PATH = "/global/cfs/cdirs/m3443/usr/ameyat/diffusion/raw-data/"
SAVE_PATH = "/global/cfs/cdirs/m3443/usr/ameyat/diffusion/pointclouds/"
FILE = "Boosted_Jets_Sample-0.snappy.parquet"
NUM_EVENTS = 32_000

f = ParquetFile(os.path.join(PATH, 'Boosted_Jets_Sample-0.snappy.parquet'))
for i in range(0, NUM_EVENTS):
    jet = np.float32((f.read_row_group(i, columns=["X_jets"]).to_pydict())["X_jets"][0])
    jet[jet < 5e-3] = 0. # Ignore all hits < 5 MeV, assumption

    # ECAL
    ecal_idxs = np.nonzero(jet[1])
    ecal_cloud = np.stack((ecal_idxs[0], ecal_idxs[1], jet[2][ecal_idxs])).transpose()

    # HCAL
    hcal_idxs = np.nonzero(jet[2])
    hcal_cloud = np.stack((hcal_idxs[0], hcal_idxs[1], jet[2][hcal_idxs])).transpose()
    
    filename = f"event_" + str(i).zfill(8) + ".npz"
    np.savez(os.path.join(SAVE_PATH, filename), ecal=ecal_cloud, hcal=hcal_cloud)
    