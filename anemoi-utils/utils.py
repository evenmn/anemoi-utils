import numpy as np
import scipy.interpolate
import cartopy.feature as cfeature

# MACROS
MONTH_LENGTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
FREQ = 6
STEP_PER_DAY = 24 // FREQ

def kelvin2celsius(ds, date):
    """Convert Kelvin to Celsius."""
    t_idx = ds.variables.index('2t')
    return ds[date_to_index(date)][t_idx,0] - 273.15

def wind_magnitude(ds, date):
    """Convert wind in x- and y-dirs to
    wind magnitude"""
    u_idx = ds.variables.index('10u')
    v_idx = ds.variables.index('10v')
    u10 = ds[date_to_index(date)][u_idx,0]
    v10 = ds[date_to_index(date)][v_idx,0]
    w10 = np.sqrt(u10**2+v10**2)
    return w10

def precip_accu6(ds, date):
    """Precip m to mm."""
    p_idx = ds.variables.index('tp')
    return ds[date_to_index(date)][p_idx,0] * 1000

map_keys = {
    'air_temperature_2m': {
        'era5': ['2t'],
        'units': 'C',
        'transform': kelvin2celsius,
    },
    'wind_speed_10m': {
        'era5': ['10u', '10v'], 
        'units': 'm/s', 
        'transform': wind_magnitude,
    },
    'precipitation_amount_acc6h': {
        'era5': ['tp'],
        'units': 'mm',
        'transform': precip_accu6,
    },
}

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

# TODO: Make separate test script for unit tests
#test_date_to_index()


def mesh(resolution):
    if resolution == 'o96':
        lat = np.arange(-90, 90, 1)
        lon = np.arange(0, 360, 1)
    elif resolution == 'n320':
        lat = np.arange(-90, 90, 0.25)
        lon = np.arange(0, 360, 0.25)
    lat_grid, lon_grid = np.meshgrid(lat, lon)
    #lat_grid = lat_grid.transpose()
    #lon_grid = lon_grid.transpose()
    return lat_grid.T, lon_grid.T

def interpolate(data, lat, lon, resolution):
    """ """
    era_lat_gridded, era_lon_gridded = mesh(resolution)

    # Interpolate irregular ERA grid to regular lat/lon grid
    icoords = np.asarray([lon, lat], dtype=np.float32).T
    ocoords = np.asarray([era_lon_gridded.flatten(), era_lat_gridded.flatten()], dtype=np.float32).T

    interpolator = scipy.interpolate.NearestNDInterpolator(icoords, data) # input coordinates
    q = interpolator(ocoords)  # output coordinates
    q = q.reshape(era_lat_gridded.shape)
    return q

def resolution_auto(data_dict):
    shape = data_dict[list(data_dict.keys())[0]].shape
    flat = len(shape) == 3
    match shape[2]:
        case 40320:
            resolution = 'o96'
            assert flat
        case 180:
            resolution = 'o96'
        case 542080:
            resolution = 'n320'
            assert flat
        case 720:
            resolution = 'n320'
        case _:
            print(shape)
            raise ValueError
    return flat, resolution

def panel_config_auto(ens_size, extra_panels):
    """Configure panel orientation, given
    number of ensemble members."""
    ens_size = 1 if ens_size is None else ens_size
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

    n = conf_map[n_panels]
    return n, ens_size

def plot(ax, data, lat_grid, lon_grid, **kwargs):
    """Plot data using pcolormesh on redefined ax"""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black')
    im = ax.pcolormesh(lon_grid, lat_grid, data, **kwargs)
    #im = ax.contourf(lon_grid, lat_grid, data)
    return im

