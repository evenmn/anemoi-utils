import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr

from data import get_data, get_era5_data, read_era5
from utils import mesh, panel_config_auto, interpolate, plot
from map_keys import map_keys

plt.rcParams["font.family"] = "serif"


def field_plotter(
        time: str or pd.Timestamp,
        fields: list[str] or str,
        path: str,
        file_truth: str = None, 
        lead_times: list[int] or int = 0,
        ens_size: int = None,
        plot_ens_mean: bool = False,
        norm: bool = False,
        xlim: tuple[float] = None,
        ylim: tuple[float] = None,
        resolution: float = None,
        freq: str = '6h',
        path_out: str = None,
        pressure_contour: bool = False,
        file_prefix: str = "",
        **kwargs,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.

    Args:
        time: str or pd.Timestamp
            Specify a time stamps to be plotted
        fields: list[str] or str
            Specify one or multiple fields to be verified. Currently supports
            air_temperature_2m, wind_speed_10m, precipitation_amount_acc6, air_sea_level_pressure
        path: str
            Path to directory where files to be analysed are found. 
            (maybe add information about NetCDF format and folder structure?)
        file_era: str
            ERA5 analysis file to be compared to. Not included by default.
        lead_times: list[int] or int
            One or multiple lead times to be plotted
        ens_size: int
            Number of ensemble members to include
        plot_ens_mean: bool
            Whether or not to plot ensemble mean.
        norm: bool
            Whether or not to normalize plots. In particular used with precipitation.
        xlim: tuple[float]
            xlim used in panels. No limit by default
        ylim: tuple[float]
            ylim used in panels. No limit by default
        resolution: float
            Resolution in degrees used in interpolation. Using 1 for o96 and 0.25 for n320 by default.
        freq: str
            Frequency of lead times. Supports pandas offset alias: 
            https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        path_out: str
            Path to where to save the image(s). If not given, images will not be saved
        pressure_contour: bool
            Whether or not to add pressure contour lines to field. Not added by default
        file_prefix: str
            filenames prefix, to use if there are multiple files at same date
    """
    fields = np.atleast_1d(fields)
    lead_times = np.atleast_1d(lead_times)

    if isinstance(time, str):
        time = pd.Timestamp(time)

    # get (ensemble) data
    ds = get_data(path, time, ens_size, file_prefix)

    if resolution is None:
        resolution = 0.25 if ds[fields[0]].shape[-1] == 542080 else 1

    regular = False
    if ds.latitude.ndim==1:
        if xlim is not None and ylim is not None:
            ds = ds.where(
                (ds.latitude >= xlim[0]) &
                (ds.latitude <= xlim[1]) &
                (ds.longitude >= ylim[0]) &
                (ds.longitude <= ylim[1]),
                drop=True
            )
        lon = np.array(ds.latitude)
        lat = np.array(ds.longitude)
        lat_grid, lon_grid = mesh(lat, lon, resolution) 
    elif ds.latitude.ndim==2:
        regular = True
        lat_grid, lon_grid = ds.y, ds.x #ds.latitude, ds.longitude
        lat_center = float((ds.latitude.max()-ds.latitude.min()) / 2.)
        lon_center = float((ds.longitude.max()-ds.longitude.min()) / 2.)
    else:
        raise ValueError

    # Reference
    include_truth = False if file_truth is None else True
    if include_truth:
        fields_ = fields
        if pressure_contour:
            fields_ = np.append(fields, 'air_pressure_at_sea_level')
        ds_truth = read_era5(fields_, file_truth, [time], max(lead_times)+1, freq=freq)
        data_truth = get_era5_data(ds_truth, 0, fields_, max(lead_times)+1)
        if regular:
            x = len(ds.x)
            y = len(ds.y)
            for key, value in data_truth.items():
                data_truth[key] = value.reshape(-1, y, x)
        if ds.latitude.ndim == 1:
            lat_grid_truth, lon_grid_truth = mesh(lat_truth, lon_truth, resolution)
        elif ds.latitude.ndim == 2:
            ylen, xlen = ds.latitude.shape
            for field in fields_:
                data_truth[field] = data_truth[field].reshape(max(lead_times)+1, ylen, xlen)

            lat_grid_truth = ds.y #ds_truth.latitudes.reshape(ylen, xlen)
            lon_grid_truth = ds.x #ds_truth.longitudes.reshape(ylen, xlen)
        else:
            raise ValueError

        """
        ds_ = xr.Dataset()
        coords = {
            "leadtime": lead_times,
            "y": ds.y,
            "x": ds.x,
        }
        dims = ["leadtime", "y", "x"]
        ds_.coords["latitude"] = (["y", "x"], lat_grid_truth)
        ds_.coords["longitude"] = (["y", "x"], lon_grid_truth)
        for key, values in data_truth.items():
            ds_[key] = xr.DataArray(
                values,
                coords=coords,
                dims=dims,
                name=key,
            )
        ds_.attrs['time'] = time.strftime('%Y-%m-%dT%H')
        filename = f"~/truth_240hfc_{time.strftime('%Y-%m-%dT%H')}.nc"
        ds_.to_netcdf(filename)
        ds_.close()
        """

    n, ens_size = panel_config_auto(ens_size, include_truth + plot_ens_mean)
    
    for field in fields:
        units = map_keys[field]['units']
        for lead_idx in tqdm(lead_times):
            # find vmin and vmax
            vmin = float(ds[field][:,lead_idx].min())
            vmax = float(ds[field][:,lead_idx].max())
            if include_truth:
                vmin = min(vmin, data_truth[field][lead_idx].min())
                vmax = max(vmax, data_truth[field][lead_idx].max())
            cen = (vmax-vmin)/10.
            vmin += cen
            vmax -= cen
            if norm:
                boundaries = np.logspace(0.001, np.log10(vmax), cmap.N-1)
                boundaries = [0.0, 0.5, 1, 2, 4, 8, 16, 32]
                norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, extend='both')
                kwargs['norm'] = norm
            else:
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            kwargs['shading'] = 'auto'

            if regular:
                #fig, axs = plt.subplots(*n, figsize=(8,6), squeeze=False, subplot_kw={'projection': ccrs.LambertConformal(central_latitude=lat_center, central_longitude=lon_center)})
                fig, axs = plt.subplots(*n, figsize=(8,6), squeeze=False, subplot_kw={'projection': ccrs.LambertConformal(central_longitude = 15, central_latitude = 63.3, standard_parallels = (63.3, 63.3))})
            else:
                fig, axs = plt.subplots(*n, figsize=(8,6), squeeze=False, subplot_kw={'projection': ccrs.PlateCarree()})
            
            # member panel(s)
            data_contour = None
            k = 0
            for i in range(n[0]):
                for j in range(n[1]):
                    data = ds[field][k, lead_idx]
                    if data.ndim == 1:
                        data = interpolate(data, lat, lon, resolution)
                    
                    if pressure_contour:
                        data_contour = ds['air_pressure_at_sea_level'][k, lead_idx]
                        if data_contour.ndim == 1:
                            data_contour = interpolate(data_contour, lat, lon, resolution)

                    # plot
                    im = plot(axs[i,j], data, lon_grid, lat_grid, contour=data_contour, **kwargs)
                    axs[i,j].set_title(f"Member {k}")
                    axs[i,j].set_xlim(xlim)
                    axs[i,j].set_ylim(ylim)
                    k += 1
                    if k >= ens_size:
                        break
                else:
                    continue
                break

            # extra panels
            if plot_ens_mean:
                data = ds[field][:,lead_idx].mean(axis=0)
                if data.ndim == 1:
                    data = interpolate(data, lat, lon, resolution)

                if pressure_contour:
                    data_contour = ds['air_pressure_at_sea_level'][:, lead_idx].mean(axis=0)
                    if data_contour.ndim == 1:
                        data_contour = interpolate(data_contour, lat, lon, resolution)

                sec_last_ax = axs[n[0]-1, n[1]-2]
                im = plot(sec_last_ax, data, lon_grid, lat_grid, contour=data_contour, **kwargs)
                sec_last_ax.set_title("Ensemble mean")
                sec_last_ax.set_xlim(xlim)
                sec_last_ax.set_ylim(ylim)

            if include_truth:
                data = data_truth[field][lead_idx]
                if data.ndim == 1:
                    data = interpolate(data, ds_truth.latitudes, ds_truth.longitudes, resolution)
                if pressure_contour:
                    data_contour = data_truth['air_pressure_at_sea_level'][lead_idx]
                    if data_contour.ndim == 1:
                        data_contour = interpolate(data_contour, ds_truth.latitudes, ds_truth.longitudes, resolution)

                last_ax = axs[n[0]-1, n[1]-1]
                im = plot(last_ax, data, lon_grid_truth, lat_grid_truth, contour=data_contour, **kwargs)
                last_ax.set_title("Truth")
                last_ax.set_xlim(xlim)
                last_ax.set_ylim(ylim)

            # show plot
            lead_time_hours = int(freq[:-1]) * lead_idx
            fig.suptitle(field + f" ({time.strftime('%Y-%m-%dT%H')}) + {lead_time_hours}h")
            plt.tight_layout()
            cbax = fig.colorbar(im, ax=axs.ravel().tolist())
            cbax.set_label(f"{field} ({units})")
            if path_out is not None:
                standard = map_keys[field]['standard']
                extent = 'global' if (xlim is None or  ylim is None) else f'lim{xlim[0]}-{xlim[1]}-{ylim[0]}-{ylim[1]}'
                filename = f"{path_out}/{standard}_{time.strftime('%Y%m%d%H')}_+{lead_time_hours:03d}_{extent}.png"
                #plt.savefig(filename)
            plt.show()
            #plt.close()

if __name__ == "__main__":
    import matplotlib

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])

    field_plotter(
        time="2022-01-02T12", 
        fields=['precipitation_amount_acc6h'], #['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level'], 
        #path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_roll2_warmup10/inference/epoch_009/predictions/", 
        path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_009/predictions/", 
        #file_truth="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr", 
        file_truth="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", 
        lead_times=[40],
        ens_size=5,
        plot_ens_mean=False,
        cmap=cmap,
        norm=True,
        #xlim=(-10,32),
        #ylim=(25,73),
        resolution=0.25,
        path_out = '/pfs/lustrep3/scratch/project_465000454/nordhage/verification/ni3_c',
        pressure_contour=False,
        file_prefix='meps',
    )
