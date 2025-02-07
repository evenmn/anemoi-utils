import numpy as np
import scipy.interpolate
from scipy.ndimage.filters import gaussian_filter
import cartopy.feature as cfeature
import xarray as xr

def mesh(lat, lon, increment):
    lat = np.arange(lat.min(), lat.max(), increment)
    lon = np.arange(lon.min(), lon.max(), increment)
    lat_grid, lon_grid = np.meshgrid(lat, lon)
    return lat_grid.T, lon_grid.T

def inter_(data, lat, lon):
    """ """
    icoords = np.asarray([lon, lat], dtype=np.float32).T
    return scipy.interpolate.NearestNDInterpolator(icoords, data)

def inter(data, lat, lon, eval_):
    """ """
    interpolator = inter_(data, lat, lon)
    return interpolator(eval_)


def interpolate(data, lat, lon, increment):
    """ """
    lat_grid, lon_grid = mesh(lat, lon, increment)
    ocoords = np.asarray([lon_grid.flatten(), lat_grid.flatten()], dtype=np.float32).T
    q = inter(data, lat, lon, ocoords)
    q = q.reshape(lat_grid.shape)
    return q

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

def plot(ax, data, x, y, contour=None, **kwargs):
    """Plot data using pcolormesh on redefined ax"""
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, edgecolor='black')
    #ax.grid()
    im = ax.pcolormesh(x, y, data, **kwargs)
    if contour is not None:
        contour = gaussian_filter(contour, 2)
        im_cntr = ax.contour(x, y, contour, linewidths=1, colors='magenta')
        ax.clabel(im_cntr, inline=True, fontsize=8, fmt='%1.0f')
    #ax.set_aspect(2)
    return im

def flatten(ds, fields):
    """Flatten xarray dataset."""
    for field in fields:
        ens_size, lead_time, ylen, xlen = ds[field].shape
        ds[field] = xr.DataArray(np.array(ds[field]).reshape(ens_size, lead_time, -1), dims=('members', 'leadtimes', 'points'))
    ds['altitude'] = np.ravel(ds.altitude)
    ds['latitude'] = np.ravel(ds.latitude)
    ds['longitude'] = np.ravel(ds.longitude)
    return ds
