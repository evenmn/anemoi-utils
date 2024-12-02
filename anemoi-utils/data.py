from glob import glob

import numpy as np
from tqdm import trange, tqdm
from anemoi.datasets import open_dataset
import xarray as xr
import dask.array as da
import pandas as pd

from utils import *

def read_era5(fields, path, resolution, times, lead_time, freq='6h'):
    """Read ERA5 data, given filename and fields subset to 
    read. Return anemoi-datasets object."""
    filename = f"{path}/aifs-ea-an-oper-0001-mars-{resolution}-1979-2022-6h-v6.zarr"
    select = []
    for field in fields:
        select.extend(map_keys[field]['era5'])
    start = times[0]
    end = times[-1]+pd.Timedelta(hours=6 * lead_time)
    ds = open_dataset(filename, frequency=freq, start=start, end=end, select=select)
    return ds

def get_era5_data(ds, time_idx, fields, lead_time):
    """Fetch data from dataset."""
    slc = slice(time_idx, time_idx + lead_time)
    era5 = {}
    for field in fields:
        era5[field] = np.array(map_keys[field]['transform'](ds, slc))
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
    data_list = pool.starmap(process_file, arguments)
    return data_list

def serial_process(ens_size, path, time):
    data_list = []
    for i in trange(ens_size):
        data_dict = process_file(i, path, time)
        data_list.append(data_dict)
    return data_list

def get_data(path, time, fields, lead_times, ens_size, pool=None):
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
        if pool is None:
            data_list = serial_process(ens_size, path, time)
        else:
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


def get_data_nc(path, time, ens_size):
    """
    Args:
        path: str
            Path to directory with nc file.
            Expects subdirs if ens_size is not None

    Outputs:
        data_dict: dict
            Dict in the form of
            data_dict[field][member,lead_time,coords]
    """
    time = time.strftime('%Y-%m-%dT%H')
    if ens_size is None:
        filename = glob(path + f"*{time}.nc")[0]
        ds = xr.open_dataset(filename)
        ds = ds.expand_dims('members').assign_coords(members=1)
    else:
        # load datasets
        datasets = []
        for i in range(ens_size):
            try:
                filename = glob(path + f"{i}/*{time}.nc")[0]
            except IndexError:
                print(f"No inference file found for time stamp {time}, member {i}")
                exit(1)
            ds = xr.open_dataset(filename, chunks={'lead_times': 'auto', 'points': 'auto'})
            datasets.append(ds)

        datasets_with_members = [ds.expand_dims('members').assign_coords(members=[i]) for i, ds in enumerate(datasets)]
        # concatinate datasets
        ds = xr.concat(datasets_with_members, dim='members')

    if 'air_temperature_2m' in ds.variables:
        ds['air_temperature_2m'] -= 273.15
    if 'precipitation_amount_acc6h' in ds.variables:
        ds['precipitation_amount_acc6h'] *= 1000
    return ds

