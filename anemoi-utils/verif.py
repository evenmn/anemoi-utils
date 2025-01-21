import os
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from data import get_data, get_era5_data, read_era5, read_verif
from map_keys import map_keys
from utils import flatten, inter

from yrlib_utils import get, get_station_metadata, get_available_timeseries, get_common_indices


def verif(
        fields: list[str] or str,
        path: str,
        file_ref: str,
        times: list[str] or str or pd.Timestamp or pd.DatetimeIndex = None,
        lead_time: int = slice(None),
        ens_size: int = None,
        every: int = 10000,
        qs: list[float] = [],
        thresholds_apply: bool = False,
        write_members: bool = True,
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

    fields = np.atleast_1d(fields)
    times = np.atleast_1d(times)

    # find reference
    if '.' not in file_ref:
        # assuming frost reference
        ref = 'frost'
        metadata = get_station_metadata(frost_client_id=file_ref, wmo=True, country='Norge')
        station_ids = [id for id in metadata]
        obs_lats = [metadata[id]["lat"] for id in station_ids]
        obs_lons = [metadata[id]["lon"] for id in station_ids]
        obs_elevs = [metadata[id]["elev"] for id in station_ids]
        # Frost uses SN18700, whereas in Verif we want just 18700
        obs_ids = [int(id.replace("SN","")) for id in metadata]
        #self.points = gridpp.Points(obs_lats, obs_lons, obs_elevs)
        """
        if None in times:
            raise Exception
        if isinstance(times, (list, tuple, np.ndarray)):
            times = pd.to_datetime(times)
        times = times.sort_values()
        ds_ref = read_era5(fields, "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", times, lead_time)
        """

        #station_ids = get_available_timeseries(frost_client_id=file_ref, variable="sum(precipitation_amount PT6H)")
    elif file_ref.endswith('.nc') or file_ref.endswith('.txt'):
        # using existing verif file as reference, using the same times and locations
        ref = 'verif'
        ds_ref = read_verif(file_ref)
        times = pd.to_datetime(ds_ref.time, unit='s', origin='unix')
    else:
        ref = 'zarr'
        if None in times:
            raise Exception
        if isinstance(times, (list, tuple, np.ndarray)):
            times = pd.to_datetime(times)
        times = times.sort_values()
        ds_ref = read_era5(fields, file_ref, times, lead_time)

    # times must be given, unless reference is another verif file with all times
    if None in times:
        raise Exception
    if isinstance(times, (list, tuple, np.ndarray)):
        times = pd.to_datetime(times)
    times = times.sort_values()

    # convert times to indices to be called from anemoi datasets
    times_idx = (times - times[0]).total_seconds() / 3600 // 6

    # create label based on date if not explicitly given
    if label is None:
        now = datetime.datetime.now()
        label = now.strftime("%Y%m%d%H%M%S")

    # get forecast data
    ds = get_data(path, times[0], ens_size)

    # represent all locations as a 1d sequence
    if ds.latitude.ndim == 2:
        ds = flatten(ds, fields)
    ens_size, lead_time, points = ds[fields[0]].shape

    if thresholds_apply:
        # for cumulative density function (CDF)
        p_ = np.linspace(0, 1, ens_size)

    # loop over times, append times to field files
    dss = []
    pbar = tqdm(zip(times, times_idx), total=len(times))
    for t_, (time, time_idx) in enumerate(pbar):
        pbar.set_description(time.strftime('%Y-%m-%dT%H'))
        unix = int(time.timestamp())
        lead_time_unix = np.arange(unix, unix + 6*3600*lead_time, 6*3600)
        if time != times[0]:
            ds = get_data(path, time, ens_size)
            if ds.latitude.ndim == 2:
                ds = flatten(ds, fields)
        if ref == "zarr": # or ref == "frost":
            data_ref = get_era5_data(ds_ref, int(time_idx), fields, lead_time)
        """
        else:
            # do interpolation once
            data_ = np.array(ds[field])
            data__ = []
            for m in range(ens_size):
                data__.append([])
                for t in range(lead_time):
                    data__[-1].append(inter(data_[m,t], ds.latitude, ds.longitude))
            data_inter = np.asarray(data__)
        """

        # loop over fields
        for j, field in enumerate(fields):
            units = map_keys[field]['units']
            long_name = map_keys[field]['long_name']

            if thresholds_apply:
                thresholds = map_keys[field]['thresholds']

            if ref == "frost":
                frost_name = map_keys[field]['frost']
                obs_times, obs_locations, obs_values = get(unix, unix + 6*lead_time*3600, frost_name, file_ref, station_ids=station_ids)
                obs_ = np.nan * np.zeros([lead_time, len(obs_locations)], np.float32)
                for t, call_time in enumerate(lead_time_unix):
                    idx = np.argwhere(obs_times == call_time)
                    obs_[t] = obs_values[idx]
                alt = [loc.elev for loc in obs_locations] 
                lat = [loc.lat for loc in obs_locations] 
                lon = [loc.lon for loc in obs_locations] 
                loc = [loc.id for loc in obs_locations] 
                eval_ = np.asarray([lon, lat], dtype=np.float32).T
                data_ = np.array(ds[field])
                data__ = []
                for m in range(ens_size):
                    data__.append([])
                    for t in range(lead_time):
                        #data__[-1].append(data_interi[m,t](eval_))
                        data__[-1].append(inter(data_[m,t], ds.latitude, ds.longitude, eval_))
                data_ = np.asarray(data__)
                if field == "air_temperature_2m":
                    # elevation correction
                    alt_inter = inter(ds.altitude, ds.latitude, ds.longitude, eval_)
                    dh = alt - alt_inter
                    dT = dh * 6.5 / 1000.
                    data_ += dT[None, None, :]

            elif ref == "verif":
                obs_ = np.array(ds_ref.obs)[t_]
                alt = np.array(ds_ref.altitude)
                lat = np.array(ds_ref.lat)
                lon = np.array(ds_ref.lon)
                loc = np.array(ds_ref.location)
                eval_ = np.asarray([lon, lat], dtype=np.float32).T
                data_ = np.array(ds[field])
                data__ = []
                for m in range(ens_size):
                    data__.append([])
                    for t in range(lead_time):
                        #data__[-1].append(data_inter[m,t](eval_))
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

            fcst_ = np.mean(data_, axis=0)

            # compute CRPS
            mae_ = np.abs(fcst_-obs_)
            ens_var_ = 0
            for i in range(ens_size):
                ens_var_ += np.sum(np.abs(data_[i][None] - data_[i+1:]), axis=0)
            crps_ = mae_ - ens_var_ / (ens_size * (ens_size - 1))

            if len(qs) > 0:
                x_ = np.quantile(data_, qs, axis=0).transpose(1,2,0)

            if thresholds_apply:
                data_sorted = np.sort(data_, axis=0)
                cdf_ = np.empty((len(thresholds), nlead_time, nlocation))
                pit_ = np.empty((nlead_time, nlocation))
                for i in range(nlead_time):
                    for k in range(nlocation):
                        cdf_[:,i,k] = np.interp(thresholds, data_sorted[:,i,k], p_, left=0, right=1)
                        #pit_[i,k] = np.interp(obs_[i,k], data_sorted[:,i,k], p_, left=0, right=1)

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
                'altitude': ('location', alt),
                'lat':      ('location', lat),
                'lon':      ('location', lon),
            }
            data_vars = {
                'obs':      (('time', 'leadtime', 'location'), obs_[None]),
                'fcst':     (('time', 'leadtime', 'location'), fcst_[None]),
                'crps':     (('time', 'leadtime', 'location'), crps_[None])
            }
            if thresholds_apply:
                coords['threshold'] = thresholds
                data_vars['cdf'] = (('time', 'leadtime', 'location', 'threshold'), cdf_.transpose(1,2,0)[None])
                data_vars['pit'] = (('time', 'leadtime', 'location'), pit_[None])
            if len(qs) > 0:
                coords['quantile'] = qs
                data_vars['x'] = (('time', 'leadtime', 'location', 'quantile'), x_[None])
            if write_members:
                data_vars['ensemble'] = (('time', 'leadtime', 'location', 'ensemble_member'), data_.transpose(1,2,0)[None])
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
            '2022-07-04T00',
            '2022-08-06T12',
            '2022-08-13T00',
            '2022-09-26T00',
            '2022-10-02T12',
            '2022-11-05T00',
            '2022-11-12T12',
        ], #pd.date_range(start='2022-01-03T00', end='2022-12-31T18', freq='4W'),
        fields=['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level'], 
        path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_077/predictions/",
        #file_ref="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", 
        file_ref="295a464e-e25d-43b5-a560-40d07296c8ea",
        #file_ref = '/pfs/lustrep3/scratch/project_465000454/nordhage/verification/mslp/spatial_noise_n320_2p5km.nc',
        lead_time=40,
        ens_size=10,
        every=500,
        path_out='/pfs/lustrep3/scratch/project_465000454/nordhage/verification/',
        label='spatial_noise_n320_2p5km_frost',
        #qs=[0.1, 0.25, 0.5, 0.75, 0.9],
        #thresholds_apply=True,
        #write_members=False,
    )
