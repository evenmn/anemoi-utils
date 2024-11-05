import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
from anemoi.datasets import open_dataset
import scipy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


plt.rcParams["font.family"] = "serif"

# MACROS
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
FREQ = 6
STEP_PER_DAY = 24 // FREQ

def wind_magnitude(ds, date):
    """Convert wind in x- and y-dirs to
    wind magnitude"""
    u10, v10 = ds[date_to_index(date)][:,0]
    w10 = np.sqrt(u10**2+v10**2)
    return w10

def str_to_idx(date: str) -> (int, int, int, int):
    """Given a time str, return the year,
    month, day and time as zero-indexed indices"""
    y, m, dt = date.split('-')
    d, t = dt.split('T')
    y = int(y)
    m = int(m)-1  # zero-indexed
    d = int(d)-1  # zero-indexed
    t = int(t) // FREQ
    return y, m, d, t

def idx_to_str(y: int, m: int, d: int, t: int) -> str:
    """Inverse of above function"""
    m += 1
    d += 1
    t *= FREQ
    date = f"{y}-{m:>02}-{d:>02}T{t:>02}"
    return date

def following_steps(date: str, n: int) -> list[str]:
    """Given a time str, return the following
    n time strs"""
    dates = [date]
    y, m, d, t = str_to_idx(date)
    for i in range(n):
        add_t = 1  # always change t
        proposed_t = t + add_t
        add_d = proposed_t // STEP_PER_DAY # 1 if add to day
        proposed_d = d + add_d
        add_m = proposed_d // MONTH_LENGTH[m]
        proposed_m = m + add_m
        add_y = proposed_m // len(MONTH_LENGTH)
        proposed_y = y + add_y
        y = proposed_y  # year can increase infinite
        m = proposed_m % len(MONTH_LENGTH)
        d = proposed_d % MONTH_LENGTH[m]
        t = proposed_t % STEP_PER_DAY
        dates.append(idx_to_str(y,m,d,t))
    return dates


def date_to_index(date: str) -> int:
    """Convert date to index.
    Assuming no missing dates.
    Date on the form 'YYYY-MM-DDTTT'"""
    y, m, d, t = str_to_idx(date)
    days_passed = sum(MONTH_LENGTH[:m]) + d
    time_steps_passed = 4 * days_passed + t
    return time_steps_passed

def test_date_to_index():
    """ """
    assert date_to_index('2022-01-01T00') == 0
    assert date_to_index('2022-02-01T00') == 124

test_date_to_index()


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


def read_npy(filename, field):
    data = np.load(filename, allow_pickle=True)
    data = data.item()
    data = data[field][:,0]
    return data


def get_ens_data(time, field, ens_size, path):
    """Get data for all ensemble members at a given time"""
    ens_data = []
    template = path + "{}/flat_era5_960hfc_{}.npy"
    for i in trange(ens_size):
        filename = template.format(i, time)
        data = read_npy(filename, field)
        ens_data.append(data)
    ens_arr = np.asarray(ens_data)
    ens_mean = ens_arr.mean(axis=0)
    ens_var = (ens_size+1)/(ens_size-1) * ens_arr.var(axis=0)
    return ens_arr, ens_mean, ens_var

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


def mesh(resolution):
    if resolution == 'o96':
        era_lat_gridded_axis = np.arange(-90, 90, 1)
        era_lon_gridded_axis = np.arange(0, 360, 1)
    elif resolution == 'n320':
        era_lat_gridded_axis = np.arange(-90, 90, 0.25)
        era_lon_gridded_axis = np.arange(0, 360, 0.25)
    era_lat_gridded, era_lon_gridded = np.meshgrid(era_lat_gridded_axis, era_lon_gridded_axis)
    era_lat_gridded = era_lat_gridded.transpose()
    era_lon_gridded = era_lon_gridded.transpose()
    return era_lat_gridded, era_lon_gridded


def interpolate(data, lat, lon, resolution):
    """ """
    era_lat_gridded, era_lon_gridded = mesh(resolution)

    # Interpolate irregular ERA grid to regular lat/lon grid
    icoords = np.asarray([lon, lat], dtype=np.float32).T
    ocoords = np.asarray([era_lon_gridded.flatten(), era_lat_gridded.flatten()], dtype=np.float32).T

    interpolator = scipy.interpolate.NearestNDInterpolator(icoords, data) # input coordinated
    q = interpolator(ocoords)  # output coordinates
    q = q.reshape(era_lat_gridded.shape)
    return q

def plot(ax, data, lat_grid, lon_grid, **kwargs):
    """Plot data using pcolormesh on redefined ax"""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black')
    ax.pcolormesh(lon_grid, lat_grid, data, **kwargs)
    #ax.contourf(lon_grid, lat_grid, data)

def get_era5_ds(resolution):
    if resolution.lower() == 'o96':
        ds = read_era5("/leonardo_work/DestE_330_24/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6.zarr", field)
    elif resolution.lower() == 'n320':
        ds = read_era5("/leonardo_work/DestE_330_24/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr", field)
    else:
        raise ValueError
    return ds

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

def imshow_field(time, field, path, lead_time=0, resolution='n320'):
    ds = get_era5_ds(resolution)
    lat_grid, lon_grid = mesh(resolution)

    # cmap/norm
    bounds = [0, 0.1, 0.5, 1, 2, 4, 8, 16, 32]
    bounds = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
    cmap = 'turbo' #matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])
    norm = None #matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend = 'both')

    nx = ny = 2
    ens_size = nx*ny
    fig, axs = plt.subplots(nx,3,figsize=(6,4), subplot_kw={'projection': ccrs.PlateCarree()})
    data_ = get_era5_data(time, field, lead_time+1, ds)
    print(data_.shape)
    """
    data_ = read_npy('era5_168hfc_2024-01-02T00.npy', field)
    """
    data_ = data_[lead_time]
    data_ = interpolate(data_, ds.latitudes, ds.longitudes, resolution)
    plot(axs[1,2], data_, lat_grid, lon_grid, cmap=cmap, norm=norm, shading='auto')
    axs[1,2].set_title("ERA5")
    
    ens_arr, ens_mean, ens_var = get_ens_data(time, field, ens_size, path)
    n_lead_times = len(ens_mean)

    for i in range(ens_size):
        j = i//nx
        k = i%nx
        # interpolate
        data = interpolate(ens_arr[i, lead_time], ds.latitudes, ds.longitudes, resolution)

        # plot
        plot(axs[j,k], data, lat_grid, lon_grid, cmap=cmap, norm=norm, shading='auto')
        axs[j,k].set_title(f"Member {i}")
    data = interpolate(ens_mean[lead_time], ds.latitudes, ds.longitudes, resolution)
    plot(axs[0,2], data, lat_grid, lon_grid, cmap=cmap, norm=norm, shading='auto')
    axs[0,2].set_title("Ensemble mean")
    
    fig.suptitle(field + f" + {6*lead_time}h")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ens_size = 4
    field = 'wind_speed_10m' #'precipitation_amount_acc6h'
    #field = 'air_temperature_2m'


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

    path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    #plot_error_spread_accu(times, field, ens_size, path)
    #imshow_error_spread(times[0], field, ens_size, path)
    imshow_field(times[0], field, path, lead_time=27, resolution='o96')
