import os
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from data import get_data, get_era5_data, read_era5
from map_keys import map_keys
from utils import flatten, inter

from yrlib_utils import get_station_metadata, get_available_timeseries, get


def verif(
        times: list[str] or str or pd.Timestamp or pd.DatetimeIndex,
        fields: list[str] or str,
        path: str,
        file_ref: str,
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
        file_ref: str
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
    times = times.sort_values()

    # convert times to indices to be called from anemoi datasets
    times_idx = (times - times[0]).total_seconds() / 3600 // 6

    ds = get_data(path, times[0], ens_size)

    if ds.latitude.ndim == 2:
        ds = flatten(ds, fields)
    ens_size, lead_time, points = ds[fields[0]].shape

    # find reference
    if '.' not in file_ref:
        # assuming frost reference
        ref = 'frost'
        station_ids = get_available_timeseries(frost_client_id=file_ref, variable="air_temperature")
    else:
        ref = 'zarr'
        ds_ref = read_era5(fields, file_ref, times, lead_time)

    # for cumulative density function (CDF)
    p_ = np.linspace(0, 1, ens_size)

    dss = []
    pbar = tqdm(zip(times, times_idx), total=len(times))
    for time, time_idx in pbar:
        pbar.set_description(time.strftime('%Y-%m-%dT%H'))
        unix = int(time.timestamp())
        if time != times[0]:
            ds = get_data(path, time, ens_size)
            if ds.latitude.ndim == 2:
                ds = flatten(ds, fields)
        if ref == "zarr":
            data_ref = get_era5_data(ds_ref, int(time_idx), fields, lead_time)
        for j, field in enumerate(fields):
            units = map_keys[field]['units']
            thresholds = map_keys[field]['thresholds']
            long_name = map_keys[field]['long_name']

            if ref == "frost":
                frost_name = map_keys[field]['frost']
                obs_times, obs_locations, obs_ = get(unix, unix + 6*lead_time*3600, frost_name, file_ref, station_ids=station_ids)
                obs_ = obs_[::6][:-1]  # Frost returns hourly data
                alt = [loc.elev for loc in obs_locations] 
                lat = [loc.lat for loc in obs_locations] 
                lon = [loc.lon for loc in obs_locations] 
                loc = [loc.id for loc in obs_locations] 
                eval_ = np.asarray([lat, lon], dtype=np.float32).T
                data_ = np.array(ds[field])
                data__ = []
                for m in range(ens_size):
                    data__.append([])
                    for t in range(lead_time):
                        data__[-1].append(inter(data_[m,t], ds.latitude, ds.longitude, eval_))
                data_ = np.asarray(data__)

            elif ref == "zarr":
                data_ = np.array(ds[field][...,::every])
                obs_ = data_ref[field][...,::every]
                alt = np.array(ds.altitude[::every])
                lat = np.array(ds.latitude[::every])
                lon = np.array(ds.longitude[::every])
                loc = np.arange(data_.shape[2])
            nmember, nlead_time, nlocation = data_.shape

            fcst_ = data_[0] #np.mean(data_, axis=0)
            ens_mean_ = np.mean(data_, axis=0)
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


            attrs = {
                'long_name': long_name,
                'standard_name': field,
                'units': units,
                'verif_version': "1.0.0",
            }
            coords = {
                'time': [unix],
                'leadtime': 6 * np.arange(nlead_time, dtype=float),
                'location': loc,
                'threshold': thresholds,
                'quantile': qs,
                'altitude': ('location', alt),
                'lat':      ('location', lat),
                'lon':      ('location', lon),
            }
            data_vars = {
                'obs':      (('time', 'leadtime', 'location'), obs_[None]),
                'fcst':     (('time', 'leadtime', 'location'), fcst_[None]),
                'ens_mean':     (('time', 'leadtime', 'location'), ens_mean_[None]),
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
        times=[
            '2022-01-01T12', 
            '2022-01-22T18', 
            '2022-02-27T12',
            '2022-03-04T00',
            '2022-04-08T12',
            '2022-04-16T00',
            '2022-05-18T12',
            '2022-05-26T00',
            '2022-06-27T12',
            '2022-07-05T00',
            '2022-08-06T12',
            '2022-08-14T00',
            '2022-09-26T00',
            '2022-10-02T12',
            '2022-11-05T00',
            '2022-11-12T12',
        ], #pd.date_range(start='2022-01-03T00', end='2022-12-31T18', freq='4W'),
        fields=['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level'], 
        path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_077/predictions/",
        #file_ref="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", 
        file_ref="295a464e-e25d-43b5-a560-40d07296c8ea",
        lead_time=40,
        ens_size=10,
        every=500,
        path_out='/pfs/lustrep3/scratch/project_465000454/nordhage/verification/',
        label='spatial_noise_n320_2p5km',
    )
