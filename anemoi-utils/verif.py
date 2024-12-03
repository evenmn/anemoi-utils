import os
import datetime

import numpy as np
from tqdm import tqdm
import xarray as xr

from data import get_data_nc, get_era5_data, read_era5
from utils import *


def verif(
        times: str,
        fields: str,
        path: str,
        path_era: str, 
        lead_time: int = slice(None),
        ens_size: int = None,
        qs: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        every: int = 10000,
        path_out: str = "/users/nordhage/",
        label: str = None,
        **kwargs,
    ) -> None:

    if label is None:
        now = datetime.datetime.now()
        label = now.strftime("%Y%m%d%H%M%S")

    fields = np.atleast_1d(fields)

    times_idx = (times - times[0]).total_seconds() / 3600 // 6

    ds = get_data_nc(path, times[0], ens_size)
    flat, resolution = resolution_auto(ds[fields[0]])
    assert flat, "Interpolated fields should not be used for verification"

    ens_size, lead_time, points = ds[fields[0]].shape
    ds_era5 = read_era5(fields, path_era, resolution, times, lead_time)

    # for cumulative density function (CDF)
    p_ = np.linspace(0, 1, ens_size, endpoint=False)

    dss = []
    pbar = tqdm(zip(times, times_idx), total=len(times))
    for time, time_idx in pbar:
        pbar.set_description(time.strftime('%Y-%m-%dT%H'))
        if time != times[0]:
            ds = get_data_nc(path, time, ens_size)
        data_era5 = get_era5_data(ds_era5, int(time_idx), fields, lead_time)
        for j, field in enumerate(fields):
            units = map_keys[field]['units']
            thresholds = map_keys[field]['thresholds']
            long_name = map_keys[field]['long_name']

            data_ = np.array(ds[field][...,::every])
            nmember, nlead_time, nlocation = data_.shape

            obs_ = data_era5[field][...,::every]
            fcst_ = np.mean(data_, axis=0)
            mae_ = np.abs(fcst_-obs_)
            ens_var_ = 0
            for i in range(ens_size):
                ens_var_ += np.sum(np.abs(data_[i][None] - data_[i+1:]), axis=0)
            crps_ = mae_ - ens_var_ / (ens_size * (ens_size - 1))

            x_ = np.quantile(data_, qs, axis=0).transpose(1,2,0)

            data_sorted = np.sort(data_, axis=0)
            cdf_ = np.empty((len(thresholds), nlead_time, nlocation))
            pit_ = np.empty((nlead_time, nlocation))
            for i in range(data_.shape[1]):
                for j in range(data_.shape[2]):
                    cdf_[:,i,j] = np.interp(thresholds, data_[:,i,j], p_)
                    pit_[i,j] = np.interp(fcst_[i,j], data_[:,i,j], p_)

            lat = np.array(ds.latitude[::every])
            lon = np.array(ds.longitude[::every])

            attrs = {
                'long_name': long_name,
                'standard_name': field,
                'units': units,
                'verif_version': "1.0.0",
            }
            coords = {
                'time': [int(time.timestamp())],
                'leadtime': 6 * np.arange(nlead_time, dtype=float),
                'location': np.arange(nlocation),
                'threshold': thresholds,
                'quantile': qs,
                'lat':      ('location', lat),
                'lon':      ('location', lon),
                'altitude': ('location', np.zeros(nlocation)),
            }
            data_vars = {
                'obs':      (('time', 'leadtime', 'location'), obs_[None]),
                'fcst':     (('time', 'leadtime', 'location'), fcst_[None]),
                'cdf':      (('time', 'leadtime', 'location', 'threshold'), cdf_.transpose(1,2,0)[None]),
                'x':        (('time', 'leadtime', 'location', 'quantile'), x_[None]),
                'pit':      (('time', 'leadtime', 'location'), pit_[None]),
                'ensemble': (('time', 'leadtime', 'location', 'ensemble_member'), data_.transpose(1,2,0)[None]),
                'crps':     (('time', 'leadtime', 'location'), crps_[None])
            }
            ds_ = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

            if len(dss) > j:
                dss[j] = xr.concat([dss[j], ds_], dim='time')
            else:
                dss.append(ds_)

    for j, field in enumerate(fields):
        subfolder = map_keys[field]['standard']
        filename = path_out + f'/{subfolder}/{label}.nc'
        dss[j].to_netcdf(filename, mode='w', unlimited_dims=['time'])
        dss[j].close()

if __name__ == "__main__":
    import pandas as pd

    fields = ['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level']

    times = pd.date_range(start='2022-01-02T00', end='2022-03-24T12', freq='1W')

    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010_incomplete/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    path_out = '/users/nordhage/'
    verif(times, fields, path, path_era, ens_size=4, every=500, path_out=path_out, label='spatial_noise')
