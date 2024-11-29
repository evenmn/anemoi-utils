#from multiprocessing import Pool
import os

import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#import verif
import xarray as xr
import pandas as pd
import scipy as sp
from scipy import interpolate

from data import get_data, get_data_nc, get_era5_data, read_era5
from utils import *

plt.rcParams["font.family"] = "serif"


def verif(
        times: str,
        fields: str,
        path: str,
        path_era: str, 
        lead_time: int = slice(None),
        ens_size: int = None,
        qs: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        every: int = 10000,
        outfile_prefix: str = "/users/nordhage/",
        **kwargs,
    ) -> None:
    #times = np.atleast_1d(times)
    fields = np.atleast_1d(fields)

    times_idx = (times - times[0]).total_seconds() / 3600 // 6

    """
    if isinstance(lead_time, int):
        lead_time = np.arange(lead_time)
    """


    # get (ensemble) data
    #pool =  Pool(ens_size)
    #ds = get_data(path, times[0], fields, lead_time, ens_size, pool)
    ds = get_data_nc(path, times[0], ens_size)
    flat, resolution = resolution_auto(ds[fields[0]])
    assert flat, "Interpolated fields should not be used for verification"

    ens_size, lead_time, points = ds[fields[0]].shape  # in case lead_time is not defined
    ds_era5 = read_era5(fields, path_era, resolution, times, lead_time)

    """
    # get lats and lons
    try:
        lats, lons = ds.latitude, ds.longitude
    except:
        #lats, lons = np.load(f"/pfs/lustrep3/scratch/project_465000454/nordhage/anemoi-utils/files/coords_{resolution}.npy")
        lats, lons = np.load(__file__ + f"../files/coords_{resolution}.npy")
    """

    p_ = np.linspace(0, 1, ens_size, endpoint=False)

    dss = []
    for time, time_idx in tqdm(zip(times, times_idx)):
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

            q_vals_ = np.quantile(data_, qs, axis=0).transpose(1,2,0)

            data_sorted = np.sort(data_, axis=0)
            f = sp.interpolate.interp1d(p_, data_sorted, axis=0, bounds_error=False, fill_value=(0,1))
            x_ = f(thresholds)

            #pit_ = f(fcst_)

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
                'cdf':      (('time', 'leadtime', 'location', 'threshold'), x_.transpose(1,2,0)[None]),
                #'pdf':      (('time', 'leadtime', 'location', 'threshold'), np.zeros((nlead_time, nlocation, len(thresholds)))[None]),
                'x':        (('time', 'leadtime', 'location', 'quantile'), q_vals_[None]),
                #'pit':      (('time', 'leadtime', 'location'), pit_[None]),
                'ensemble': (('time', 'leadtime', 'location', 'ensemble_member'), data_.transpose(1,2,0)[None]),
                'crps':     (('time', 'leadtime', 'location'), crps_[None])
            }
            ds_ = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

            if len(dss) > j:
                dss[j] = xr.concat([dss[j], ds_], dim='time')
            else:
                dss.append(ds_)

    for j, field in enumerate(fields):
        filename = outfile_prefix + f'_{field}.nc'
        dss[j].to_netcdf(filename, mode='w', unlimited_dims=['time'])
        dss[j].close()

    """
    ds_era5 = get_era5_data(path_era, times[0], fields, lead_times, resolution)
    # write new files
    for field in fields:
        units = map_keys[field]['units']
        thresholds = map_keys[field]['thresholds']
        with open(outfile_prefix + f'_{field}.txt', 'w') as f:
            f.write(f"# variable: {field}\n")
            f.write(f"# units: {units}\n")
            f.write("date hour leadtime location lat lon obs fcst")
            for p in thresholds:
                f.write(f" p{p}")
            for q in qs:
                f.write(f" q{q}")
            for i in range(ens_size):
                f.write(f" e{i}")
            f.write(" pit crps \n")

    # append data to files
    for time in times:
        if time != times[0]:
            ds = get_data_nc(path, time, ens_size)
        ds_era5 = get_era5_data(path_era, time, fields, lead_times, resolution)
        date, hour = time.split('T')
        date = date.replace('-', '')
        hour = int(hour)
        for field in fields:
            print(time, field)
            data_field = ds[field]
            thresholds = map_keys[field]['thresholds']
            with open(outfile_prefix + f'_{field}.txt', 'a') as f:
                # vectorized
                data_ = np.array(ds[field][...,::every])
                num_points = data_.shape[-1]
                data_ = data_.reshape(ens_size,-1)
                num_rows = data_.shape[1]
                date_ = np.full(num_rows, date)[None]
                print('date.shape', date_.shape)
                hour_ = np.full(num_rows, hour)[None]
                print('hour.shape', hour_.shape)
                lt_idx_ = np.arange(lead_time).repeat(num_points)[None]
                print('lt_idx.shape', lt_idx_.shape)

                print('data.shape:', data_.shape)
                fcst_ = np.mean(data_, axis=0)[None]
                print('fcst.shape:', fcst_.shape)
                obs_ = ds_era5[field][...,::every].flatten()[None]
                print('obs.shape:', obs_.shape)
                #data_sorted_ = np.sort(data, axis=0)
                cdf_ = np.zeros(num_rows)[None]
                q_vals_ = np.quantile(data_, qs, axis=0)
                print('q_vals.shape:', q_vals_.shape)
                pit_ = np.zeros(num_rows)[None]
                mae_ = np.abs(fcst_-obs_)
                print('mae.shape:', mae_.shape)
                ens_var_ = 0
                for i in range(ens_size):
                    ens_var_ += np.sum(np.abs(data_[i][None] - data_[i+1:]), axis=0)
                crps_ = mae_ - ens_var_ / (ens_size * (ens_size - 1))
                print('crps.shape:', crps_.shape)

                out_dump_ = np.concatenate([date_, hour_, lt_idx_, fcst_, obs_, cdf_, q_vals_, pit_, crps_, data_], axis=0)
                print('out_dump.shape:', out_dump_.shape)
                fmt_list = out_dump_.shape[0] * ['%.3f']
                fmt_list[0] = '%s'; fmt_list[1] = '%s'; fmt_list[2] = '%d'
                fmt = ' '.join(fmt_list)
                print(fmt)
                np.savetxt(f, out_dump_.T, fmt=fmt) #, fmt='%.3f')

                for point_idx in trange(0, points, every):
                    for lt_idx in range(lead_time):
                        data = np.array(data_field[:,lt_idx,point_idx])

                        lat = lats[point_idx]
                        lon = lons[point_idx]
                        fcst = np.mean(data)
                        obs = ds_era5[field][lt_idx,point_idx]

                        f.write(f"{date} {hour} {lt_idx} {point_idx//every} {lat:.3f} {lon:.3f} {obs:.3f} {fcst:.3f}")

                        # Cumulative distribution function (CDF)
                        data_sorted = np.sort(data)
                        func = lambda t: np.interp(t, p_, data_sorted)
                        for t in thresholds:
                            f.write(f" {func(t):.3f}")

                        # quantiles
                        for q in qs:
                            q_val = np.quantile(data, q)
                            f.write(f" {q_val:.3f}")

                        # ensemble members 
                        for i in range(ens_size):
                            e = data[i]
                            f.write(f" {e:.3f}")

                        # probability integral transform (pit)
                        pit = func(fcst)
                        f.write(f" {pit:.3f}")

                        # crps
                        crps = np.abs(obs-fcst)
                        ens_var = 0
                        for i in range(ens_size):
                            ens_var += np.sum(np.abs(data[i][None] - data[i+1:]))
                        crps -= ens_var / (ens_size * (ens_size - 1))
                        f.write(f" {crps:.3f}\n")
                """

if __name__ == "__main__":
    #fields = ['air_temperature_2m']
    fields = ['precipitation_amount_acc6h']
    fields = ['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level']

    times = pd.date_range(start='2022-01-02T00', end='2022-03-24T12', freq='1W')

    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_new/inference/epoch_010/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    #path_era = "/leonardo_work/DestE_330_24/anemoi/datasets/ERA5/"
    outfile_prefix = '/users/nordhage/ni1_b_new'
    verif(times, fields, path, path_era, ens_size=4, every=500, outfile_prefix=outfile_prefix)

    """
    for field in fields:
        filename = outfile_prefix + f"_{field}.txt"
        data = verif.Data.Data([filename])
        data.getFilenames()
        taylor.setFigsize((10,8))
        taylor.plot(data)

        error = verif.Output.Error()
        error.setFigsize((10,8))
        error.plot(data)
    """
