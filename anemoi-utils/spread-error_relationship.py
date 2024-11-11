#from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import trange, tqdm
#from anemoi.datasets import open_dataset
#import scipy
import cartopy.crs as ccrs

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"


def plot_error_spread(times, field, ens_size, path, resolution='n320'):
    ds = get_era5_ds(resolution)

    fig, axs = plt.subplots(1,3, figsize=(8,3))
    for time in times:
        ens_arr, ens_mean, ens_var = get_ens_data(time, field, ens_size, path)

        spread_mean = ens_var.mean(axis=1)
        n_lead_times = len(spread_mean)
        lead_times_days = np.arange(n_lead_times) / STEP_PER_DAY

        era5_arr = get_era5_data(time, field, n_lead_times, ds)

        mse = (era5_arr-ens_mean)**2
        mse_mean = mse.mean(axis=1)

    # plot
    axs[0].plot(lead_times_days, spread_mean, label=time)
    axs[1].plot(lead_times_days, mse_mean)
    axs[0].legend(loc='best')
    axs[0].set_xlabel("Lead times (days)")
    axs[0].set_ylabel(r"Ensemble variance ($m^2/s^2$)") #K^2$)")
    axs[0].set_xticks(lead_times_days.astype(int))
    axs[1].set_xlabel("Lead times (days)")
    axs[1].set_ylabel(r"Mean-squared error ($m^2/s^2$)") #K^2$)")
    axs[1].set_xticks(lead_times_days.astype(int))
    axs[2].set_xlabel(r"Ensemble variance ($m^2/s^2$)") #K^2$)")
    axs[2].set_ylabel(r"Mean-squared error ($m^2/s^2$)") #K^2$)")
    fig.suptitle(field)
    plt.tight_layout()
    plt.show()

def plot_error_spread_accu(times, field, ens_size, path, resolution='n320'):
    ds = get_era5_ds(resolution)

    #spread_means = []
    #mse_means = []
    for time in times:
        ens_arr, ens_mean, ens_var = get_ens_data(time, field, ens_size, path)

        spread_mean = np.sqrt(ens_var).mean(axis=1)
        n_lead_times = len(spread_mean)
        lead_times_days = np.arange(n_lead_times) / STEP_PER_DAY

        era5_arr = get_era5_data(time, field, n_lead_times, ds)

        mse = (era5_arr-ens_mean)**2
        mse_mean = mse.mean(axis=1)
        np.savetxt( f"files/mse_mean_{time}.npy",mse_mean)
        np.savetxt( f"files/spread_mean_{time}.npy",spread_mean)

        #spread_means.append(spread_mean)
        #mse_means.append(mse_mean)

    """
    spread_mean = np.mean(spread_means, axis=0)
    mse_mean = np.mean(mse_means, axis=0)

    # plot
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(lead_times_days, spread_mean, label='STD')
    ax.plot(lead_times_days, mse_mean, label='RMSE')
    ax.legend(loc='best')
    ax.set_xlabel("Lead times (days)")
    ax.set_ylabel(r"Ensemble RMSE/STD ($K^2$)")
    ax.set_xticks(lead_times_days.astype(int))
    fig.suptitle(field)
    plt.tight_layout()
    plt.savefig('/leonardo_work/DestE_330_24/enordhag/ens-score-anemoi/spread_error.png')
    plt.show()
    """



### PLOT

def plot(ax, data, lat_grid, lon_grid, **kwargs):
    """Plot data using pcolormesh on redefined ax"""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black')
    im = ax.pcolormesh(lon_grid, lat_grid, data, **kwargs)
    #im = ax.contourf(lon_grid, lat_grid, data)
    return im

def imshow_error_spread(time, field, ens_size, path, lead_time=0, resolution='n320'):
    ds = get_era5_ds(resolution)
    fig, axs = plt.subplots(1,2, figsize=(6,4), subplot_kw={'projection': ccrs.PlateCarree()})
    ens_arr, ens_mean, ens_var = get_ens_data(time, field, ens_size, path)
    n_lead_times = len(ens_mean)

    era5_arr = get_era5_data(time, field, n_lead_times, ds)

    mse = (era5_arr-ens_mean)**2

    # interpolate
    var_field = interpolate(ens_var[lead_time], ds.latitudes, ds.longitudes)
    mse_field = interpolate(mse[lead_time], ds.latitudes, ds.longitudes)

    # plot
    lat_grid, lon_grid = mesh(resolution)
    plot(axs[0], var_field, lat_grid, lon_grid, cmap='viridis', shading='auto')
    plot(axs[0], mse_field, lat_grid, lon_grid, cmap='viridis', shading='auto')
    axs[0].set_title("Ensemble variance")
    axs[1].set_title("Mean-squared error")
    fig.suptitle(field)
    plt.tight_layout()
    plt.show()


def field_plotter(
        time: str, 
        fields: str,
        path: str,
        path_era: str = None, 
        lead_times: str = [0],
        ens_size: int = None,
        plot_ens_mean: bool = False,
        **kwargs,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    """
    fields = np.atleast_1d(fields)
    lead_times = np.atleast_1d(lead_times)

    # get (ensemble) data
    data_dict = get_data(path, time, ens_size)
    flat, resolution = resolution_auto(data_dict)

    # ERA5
    include_era = False if path_era is None else True
    if include_era:
        data_era5 = get_era5_data(path_era, time, resolution, fields, lead_times)
        #ds = get_era5_ds(path_era, resolution, fields)
        #data_era5 = get_era5_data(time, fields, lead_times, ds)

    n, ens_size = panel_config_auto(ens_size, include_era + plot_ens_mean)
    
    lat_grid, lon_grid = mesh(resolution)
    lats, lons = np.load(f"/pfs/lustrep3/scratch/project_465000454/nordhage/anemoi-utils/files/coords_{resolution}.npy")


    for field in fields:
        units = map_keys[field]['units']
        for lead_time in lead_times:
            # find vmin and vmax
            vmin = data_dict[field][:,lead_time].min()
            vmax = data_dict[field][:,lead_time].max()
            if include_era:
                vmin = min(vmin, data_era5[field][lead_time].min())
                vmax = max(vmax, data_era5[field][lead_time].max())
            cen = (vmax-vmin)/10.
            kwargs['vmin'] = vmin + cen
            kwargs['vmax'] = vmax - cen
            kwargs['shading'] = 'auto'

            fig, axs = plt.subplots(*n, figsize=(6,4), squeeze=False, subplot_kw={'projection': ccrs.PlateCarree()})
            
            # member panel(s)
            k = 0
            for i in range(n[0]):
                for j in range(n[1]):
                    data = data_dict[field][k, lead_time]
                    if flat:
                        data = interpolate(data, lats, lons, resolution)

                    # plot
                    im = plot(axs[i,j], data, lat_grid, lon_grid, **kwargs)
                    axs[i,j].set_title(f"Member {k}")
                    k += 1
                    if k >= ens_size:
                        break
                else:
                    continue
                break

            # extra panels
            if plot_ens_mean:
                data = data_dict[field][:,lead_time].mean(axis=0)
                if flat:
                    data = interpolate(data, lats, lons, resolution)
                sec_last_ax = axs[n[0]-1, n[1]-2]
                im = plot(sec_last_ax, data, lat_grid, lon_grid, **kwargs)
                sec_last_ax.set_title("Ensemble mean")

            if include_era:
                data = data_era5[field][lead_time]
                data = interpolate(data, lats, lons, resolution)
                last_ax = axs[n[0]-1, n[1]-1]
                im = plot(last_ax, data, lat_grid, lon_grid, **kwargs)
                last_ax.set_title("ERA5")

            #TODO: Add custom field for comparison
            #data_ = read_npy('era5_168hfc_2024-01-02T00.npy', field)

            # show plot
            fig.suptitle(field + f" + {6*lead_time}h")
            plt.tight_layout()
            cbax = fig.colorbar(im, ax=axs.ravel().tolist())
            cbax.set_label(f"{field} ({units})")
            plt.show()


if __name__ == "__main__":
    fields = ['wind_speed_10m', 'air_temperature_2m']

    # cmap/norm
    #bounds = [0, 0.1, 0.5, 1, 2, 4, 8, 16, 32]
    #bounds = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
    cmap = 'turbo' #matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])
    norm = None #matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend = 'both')

    times = [
        #"2022-01-02T00",
        #"2022-02-02T00",
        #"2022-03-02T00",
        #"2022-04-02T00",
        #"2022-05-02T00",
        #"2022-06-02T00",
        #"2022-07-02T00",
        #"2022-08-02T00",
        "2022-09-02T00",
        #"2022-10-02T00",
        #"2022-11-02T00",
        ]

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/silly_variable_10/inference/epoch_010/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    #plot_error_spread_accu(times, field, ens_size, path)
    #imshow_error_spread(times[0], field, ens_size, path)
    field_plotter(times[0], fields, path, path_era, lead_times=[5,12], ens_size=4, plot_ens_mean=True, cmap='turbo')
