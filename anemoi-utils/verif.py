import os
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from data import get_data, get_era5_data, read_era5
from map_keys import map_keys


def verif(
        times: list[str] or str or pd.Timestamp or pd.DatetimeIndex,
        fields: list[str] or str,
        path: str,
        file_era: str,
        lead_time: int = slice(None),
        ens_size: int = None,
        qs: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        every: int = 10000,
        path_out: str = "~/",
        label: str = None,
    ) -> None:
    """Generating probabilistic verification files to be used with verif. Files
    are verified against ERA5 analysis. Returning NetCDF format.
    see: https://github.com/WFRT/verif

    Args:
        times: list[str] or str or pd.Timestamp or pd.DatetimeIndex
            Specify one or multiple time stamps to be verified
        fields: list[str] or str
            Specify one or multiple fields to be verified. Currently supports
            air_temperature_2m, wind_speed_10m, precipitation_amount_acc6, air_sea_level_pressure
        path: str
            Path to directory where files to be analysed are found. 
            (maybe add information about NetCDF format and folder structure?)
        file_era: str
            ERA5 analysis file to be compared to
        lead_time: int
            Number of lead times to include. All lead times by default
        ens_size: int
            Number of ensemble members to include
        qs: list[float]
            Quantiles to calculate, numbers between 0 and 1
        every: int
            Include every this grid point in analysis. A reasonable number is ~1000 total points
        path_out: str
            Output path, home directory by default.  Will write each field to respective sub folders.
        label: str
            Label associated with outputs. Will make random time-dependent label if not given.
    """

    times = np.atleast_1d(times)
    fields = np.atleast_1d(fields)

    if label is None:
        now = datetime.datetime.now()
        label = now.strftime("%Y%m%d%H%M%S")

    if isinstance(times, (list, tuple, np.ndarray)):
        times = pd.to_datetime(times)

    # convert times to indices to be called from anemoi datasets
    times_idx = (times - times[0]).total_seconds() / 3600 // 6

    ds = get_data(path, times[0], ens_size)

    assert ds[fields[0]].ndim==3, "Interpolated fields should not be used for verification"

    ens_size, lead_time, points = ds[fields[0]].shape
    ds_era5 = read_era5(fields, file_era, times, lead_time)

    # for cumulative density function (CDF)
    p_ = np.linspace(0, 1, ens_size)

    dss = []
    pbar = tqdm(zip(times, times_idx), total=len(times))
    for time, time_idx in pbar:
        pbar.set_description(time.strftime('%Y-%m-%dT%H'))
        if time != times[0]:
            ds = get_data(path, time, ens_size)
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
            for i in range(nlead_time):
                for k in range(nlocation):
                    cdf_[:,i,k] = np.interp(thresholds, data_sorted[:,i,k], p_, left=0, right=1)
                    pit_[i,k] = np.interp(obs_[i,k], data_sorted[:,i,k], p_, left=0, right=1)

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
                'altitude': ('location', np.zeros(nlocation)),  # altitude = z/9.81
            }
            data_vars = {
                'obs':      (('time', 'leadtime', 'location'), obs_[None]),
                'fcst':     (('time', 'leadtime', 'location'), fcst_[None]),
                'cdf':      (('time', 'leadtime', 'location', 'threshold'), cdf_.transpose(1,2,0)[None]),
                'x':        (('time', 'leadtime', 'location', 'quantile'), x_[None]),
                'pit':      (('time', 'leadtime', 'location'), pit_[None]),
                #'ensemble': (('time', 'leadtime', 'location', 'ensemble_member'), data_.transpose(1,2,0)[None]),
                'crps':     (('time', 'leadtime', 'location'), crps_[None])
            }
            ds_ = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

            try:
                dss[j] = xr.concat([dss[j], ds_], dim='time')
            except IndexError:
                dss.append(ds_)

    for j, field in enumerate(fields):
        subfolder = map_keys[field]['standard']
        filename = path_out + f'/{subfolder}/{label}.nc'
        dss[j].to_netcdf(filename, mode='w', unlimited_dims=['time'])
        dss[j].close()

if __name__ == "__main__":
    verif(
        times=pd.date_range(start='2022-01-01T00', end='2022-12-31T18', freq='4W'),
        fields=['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level'], 
        path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_a_fix/inference/epoch_099/predictions/",
        file_era="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr", 
        ens_size=20,
        every=10, 
        path_out='/pfs/lustrep3/scratch/project_465000454/nordhage/verification/',
        label='spatial_noise_o96',
    )
