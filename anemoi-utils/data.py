from glob import glob

import numpy as np
from tqdm import trange, tqdm
from anemoi.datasets import open_dataset

from utils import *

def read_era5(filename, fields):
    """Read ERA5 data, given filename and
    fields subset to read. Return anemoi-
    datasets object."""
    select = []
    for field in fields:
        select.extend(map_keys[field]['era5'])
    ds = open_dataset(filename, frequency="6h", start="2022-01-01", end="2022-12-31", select=select)
    return ds

def get_era5_data(path, time, fields, lead_times, resolution):
    """Fetch data from dataset."""
    filename = f"{path}/aifs-ea-an-oper-0001-mars-{resolution}-1979-2022-6h-v6.zarr"
    ds = read_era5(filename, fields)
    lead_time_stamps = following_steps(time, max(lead_times))
    era5 = {}
    for field in fields:
        era5[field] = []
        for lead_time_stamp in tqdm(lead_time_stamps):
            era5[field].append(map_keys[field]['transform'](ds, lead_time_stamp))
        era5[field] = np.asarray(era5[field])
    return era5

def read_npy(filename):
    """Read npy file and return dict."""
    data = np.load(filename, allow_pickle=True)
    data = data.item()
    return data

def process_file(i, path, time):
    """For parallel processing."""
    file_pattern = path + f"{i}/*{time}.npy"
    filenames = glob(file_pattern)
    if len(filenames) < 1:
        raise ValueError(f"No file matches file pattern '{file_pattern}'!")
    filename = filenames[0]  # pick first match if several matches exist
    data_dict = read_npy(filename)
    return data_dict
    
def parallel_process(ens_size, path, time, pool):
    arguments = [(i, path, time) for i in range(ens_size)]
    data_dict = pool.starmap(process_file, arguments)
    return data_dict

def get_data(path, time, fields, lead_times, ens_size, pool):
    """
    Args:
        path: str
            Path to directory with npy file.
            Expects subdirs if ens_size is not None

    Outputs:
        data_dict: dict
            Dict in the form of
            data_dict[field][member,lead_time,coords]
    """
    data_dict_ = {}
    if ens_size is None:
        # Flat folder structure
        filename = glob(path + f"*{time}.npy")[0]
        data_dict = read_npy(filename)
        for key, value in data_dict.items():
            data_dict_[key] = value[lead_times,0][np.newaxis]
    else:
        # npy files in subdirs
        data_list = parallel_process(ens_size, path, time, pool)
        for data_dict in data_list:
            for key, value in data_dict.items():
                if not key in fields:
                    continue
                try:
                    data_dict_[key] = np.concatenate((data_dict_[key], value[lead_times,0][np.newaxis]))
                except KeyError:
                    data_dict_[key] = value[lead_times,0][np.newaxis]
    if 'air_temperature_2m' in data_dict_.keys():
        data_dict_['air_temperature_2m'] -= 273.15
    if 'precipitation_amount_acc6h' in data_dict_.keys():
        data_dict_['precipitation_amount_acc6h'] *= 1000
    return data_dict_

