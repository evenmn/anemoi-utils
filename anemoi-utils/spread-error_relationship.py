import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
from anemoi.datasets import open_dataset
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import *

plt.rcParams["font.family"] = "serif"

# MACROS
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
FREQ = 6
STEP_PER_DAY = 24 // FREQ

### ERA5

def read_era5(filename, field):
    # Read ERA5 data (hard-coded for now)
    if field == 'wind_speed_10m':
        select = ['10u', '10v']
    elif field == 'air_temperature_2m':
        select = ['2t']
    else:
        select = None
    ds = open_dataset(filename, frequency="6h", start="2022-01-01", end="2022-12-31", select=select)
    return ds

def get_era5_ds(path, resolution, field):
    filename = f"{path}/aifs-ea-an-oper-0001-mars-{resolution}-1979-2022-6h-v6.zarr" 
    return read_era5(filename, field)

def get_era5_data(time, field, n_lead_times, ds):
    # ERA5
    lead_time_stamps = following_steps(time, n_lead_times-1)
    era5 = []
    for lead_time_stamp in tqdm(lead_time_stamps):
        if field == 'wind_speed_10m':
            era5.append(wind_magnitude(ds, lead_time_stamp))
        elif field == 'air_temperature_2m':
            era5.append(ds[date_to_index(lead_time_stamp)][0,0])
        else:
            raise ValueError("")
    era5_arr = np.asarray(era5)
    return era5_arr


### IO

def read_npy(filename, field):
    data = np.load(filename, allow_pickle=True)
    data = data.item()
    data = data[field][:,0]
    return data

### INFERENCE FIELDS

def get_ens_data(time, field, ens_size, path):
    """Get data for all ensemble members at a given time"""
    ens_data = []
    template = path + "{}/era5_72hfc_{}.npy" #flat_era5_960hfc_{}.npy"
    for i in trange(ens_size):
        filename = template.format(i, time)
        data = read_npy(filename, field)
        ens_data.append(data)
    ens_arr = np.asarray(ens_data)
    ens_mean = ens_arr.mean(axis=0)
    if ens_size > 1:
        ens_var = (ens_size+1)/(ens_size-1) * ens_arr.var(axis=0)
    else:
        ens_var = None
    return ens_arr, ens_mean, ens_var

## PLOT

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

def resolution_auto(ens_mean):
    flat = len(ens_mean.shape) == 2

    match ens_mean.shape[1]:
        case 40320:
            resolution = 'o96'
            assert flat
        case 180:
            resolution = 'o96'
        case 500000:
            resolution = 'n320'
            assert flat
        case 720:
            resolution = 'n320'
        case _:
            raise ValueError

    return flat, resolution

def panel_config_auto(ens_size, extra_panels):
    """Configure panel orientation, given
    number of ensemble members."""
    n_panels = ens_size
    n_panels += extra_panels

    conf_map = [None,
                (1,1), (1,2), (2,2), (2,2),
                (2,3), (2,3), (2,4), (2,4),
                (3,3), (3,4), (3,4), (3,4),
                (4,4), (4,4), (4,4), (4,4),
               ]
    panel_limit = len(conf_map) - 1

    if n_panels > panel_limit:
        print(f"Panel limit reached, continuing with {panel_limit} panels")
        n_panels = panel_limit
        ens_size = panel_limit - extra_panels

    """
    match n_panels:
        case 1:
            n = (1,1)
        case 2:
            n = (1,2)
        case 3 | 4:
            n = (2,2)
        case 5 | 6:
            n = (2,3)
        case 7 | 8:
            n = (2,4)
        case 9:
            n = (3,3)
        case 10 | 11 | 12:
            n = (3,4)
        case 13 | 14 | 15 | 16:
            n = (4,4)
        case _:
            print("Continuing with 16 panels")
            ens_size = 16 - extra_panels
    """
    n = conf_map[n_panels]
    return n, ens_size


def imshow_field(time: str, 
                 field: str, 
                 path: str,
                 path_era: str = None, 
                 lead_time: str = 0,
                 ens_size: str = 1,
                 plot_ens_mean: bool = False,
                 **kwargs,
                 ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    TODO: Collect lat and lon independently of ERA5
    """
    include_era = False if path_era is None else True

    # get (ensemble) data
    n, ens_size = panel_config_auto(ens_size, include_era + plot_ens_mean)
    ens_arr, ens_mean, ens_var = get_ens_data(time, field, ens_size, path)
    n_lead_times = len(ens_mean)
    
    flat, resolution = resolution_auto(ens_mean)
    lat_grid, lon_grid = mesh(resolution)

    # temporal solution to get correct lat and lon, need a better solution
    ds = get_era5_ds(path_era, resolution, field)

    # find vmin and vmax
    vmin = ens_arr.min()
    vmax = ens_arr.max()
    if include_era:
        data_era5 = get_era5_data(time, field, lead_time+1, ds)
        vmin = min(vmin, data_era5.min())
        vmax = max(vmax, data_era5.max())

    fig, axs = plt.subplots(*n, figsize=(6,4), squeeze=False, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # member panel(s)
    k = 0
    for i in range(n[0]):
        for j in range(n[1]):
            data = ens_arr[k, lead_time]
            if flat:
                data = interpolate(data, ds.latitudes, ds.longitudes, resolution)

            # plot
            im = plot(axs[i,j], data, lat_grid, lon_grid, shading='auto', vmin=vmin, vmax=vmax, **kwargs)
            axs[i,j].set_title(f"Member {k}")
            k += 1
            if k >= ens_size:
                break
        else:
            continue
        break

    # extra panels
    if plot_ens_mean:
        data = ens_mean[lead_time]
        if flat:
            data = interpolate(data, ds.latitudes, ds.longitudes, resolution)
        sec_last_ax = axs[n[0]-1, n[1]-2]
        im = plot(sec_last_ax, data, lat_grid, lon_grid, shading='auto', vmin=vmin, vmax=vmax, **kwargs)
        sec_last_ax.set_title("Ensemble mean")

    if include_era:
        data = data_era5[lead_time]
        data = interpolate(data, ds.latitudes, ds.longitudes, resolution)
        last_ax = axs[n[0]-1, n[1]-1]
        im = plot(last_ax, data, lat_grid, lon_grid, shading='auto', vmin=vmin, vmax=vmax, **kwargs)
        last_ax.set_title("ERA5")

    #TODO: Add custom field for comparison
    #data_ = read_npy('era5_168hfc_2024-01-02T00.npy', field)
    

    # show plot
    fig.suptitle(field + f" + {6*lead_time}h")
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

if __name__ == "__main__":
    field = 'wind_speed_10m' #'precipitation_amount_acc6h'
    #field = 'air_temperature_2m'

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
        #"2022-09-02T00",
        "2022-10-02T00",
        #"2022-11-02T00",
        ]

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_a/inference/epoch_080/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    #plot_error_spread_accu(times, field, ens_size, path)
    #imshow_error_spread(times[0], field, ens_size, path)
    imshow_field(times[0], field, path, path_era, lead_time=10, ens_size=4, plot_ens_mean=True, cmap='turbo')
