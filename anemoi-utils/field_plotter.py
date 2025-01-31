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


def panel_daemon(num_models, num_lead_times, num_members, plot_ens_mean, include_ref, swap_axes=False):
    """Panel daemon that controls panel configuration.
    Striving for a square-horizontal layout, but a square-vertical
    layout can be invokes by setting swap_axes=True.

    Policy:
    - Can only plot two dimensions at the same time, meaning that
      either num_models, num_lead_times or num_members + plot_ens_mean
      has to be one.
    - If two of them are one, the dimensions is rearranged into 2d
      as close to square as possible, with a horizontal preference.
      Note that panels can be left empty if perfect rearrangement
      if not possible (e.g. in case of a prime number of panels)
    - If two dimensions, the shortest dimension will always be in
      vertical direction (usually in order model - lead time - ens).
    - Reference will never be appended to lead times direction,
      and then appended to the longest dimension of model and ens

    Args:
    - num_models: int
        number of models/paths
    - num_lead_times: int
        number of lead times to plot
    - num_members: int
        number of ensemble members to include
    - plot_ens_mean: bool
        plot ens mean true/false
    - include ref: bool
        include reference true/false
    - swap_axes: bool
        swap x- and y axis, making plot vertical

    Returns:
    - n: tuple[int]
        number of panels in x (horizontal) and y (vertical) directions
    - ens_size: int
        potentially updated ensemble size
    - idx: tuple[int]
        variable indices of dimensions.
        0: model, 1: lead time, 2: ens
    - ref_dim: int
        dimension to append reference along
    """
    return n, ens_size, idx, ref_dim
    if num_members is None:
        # num_members can not be set (None) for two reasons:
        # 1. deterministic forecast, set num_members to 1
        # 2. plotting ensemble mean only, set num_members to 0
        if plot_ens_mean:
            num_members = 0
        else:
            num_members = 1

    # always append ens mean panel in ensemble dim
    num_members += plot_ens_mean

    # assert correct dimensionalities
    dim_len = [num_models, num_lead_times, num_members]
    assert not 0 in dim_len, "Cannot deal with zero dimension"
    assert 1 in in dim_len, "At least one dimension needs length 1"

    # check if 1d or 2d
    active_dim = np.where(dim_len > 1)
    num_dims = active_dim.sum()
    dim_len_argsort = np.argsort(dim_len)

    if num_dims == 1:
        n, ens_size = panel_config_auto()
        return n, ens_size, None, None

    # horizontal direction, idx: 0-mod 1-lt 2-ens
    idx = dim_len_argsort[:-2]

    n = (dim_len[idx[0]], dim_len[idx[1]])

    ref_dim = None
    if include_ref:
        # add ref along longest dim, but not lead times
        if idx[0] != 1:
            ref_dim = 0
        else:
            ref_dim = 1
        n[ref_dim] += 1

    if swap_axes:
        n = tuple(reversed(n))
        idx = tuple(reversed(idx))
    
    return n, ens_size, idx, ref_dim


def _get_path_features(self, resolution):
    """extract path-specific features.
    Returns a list with a dictionary for each path.

    TODO: more variables can be made path specific,
    for instance resolution could be given as a tuple.
    """
    path_features = []
    for path in self.paths:
        features = {}
        # get (ensemble) data
        ds = get_data(path, self.time, self.ens_size, self.file_prefix)
        features['ds'] = ds

        if resolution is None:
            resolution = 0.25 if ds[fields[0]].shape[-1] == 542080 else 1
        features['resolution'] = resolution

        regular = False
        if ds.latitude.ndim==1:
            if self.xlim is not None and self.ylim is not None:
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
            features['lon'] = lon
            features['lat'] = lat
        elif ds.latitude.ndim==2:
            regular = True
            lat_grid, lon_grid = ds.y, ds.x #ds.latitude, ds.longitude
            lat_center = float((ds.latitude.max()-ds.latitude.min()) / 2.)
            lon_center = float((ds.longitude.max()-ds.longitude.min()) / 2.)
        else:
            raise ValueError
        features['regular'] = regular
        features['lat_grid'] = lat_grid
        features['lon_grid'] = lon_grid
        path_features.append(features)
    return path_features


def _get_ref_features(self, file_truth, pressure_contour, time, lead_times, freq, ds, fields, lat_truth, lon_truth, resolution)
    """Reference features."""
    include_truth = False if file_truth is None else True
    if include_truth:
        fields_ = fields  # fields to plot vs. fields to fetch
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
        return data_truth, lat_grid_truth, lon_grid_truth

def _plot_panel(ax, data, lat_grid, lon_grid, data_pressure=None, lat=None, lon=None, xlim=None, ylim=None, title="", **kwargs):
    """Plot a single panel. Interpolate if necessary."""
    if data.ndim == 1:
        data = interpolate(data, lat, lon, resolution)

    if data_pressure is not None:
        if data_contour.ndim == 1:
            data_contour = interpolate(data_contour, lat, lon, resolution)

    im = plot(ax, data, lon_grid, lat_grid, contour=data_pressure, **kwargs)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def _process_fig(freq, lead_idx, fig, axs, field, units, path_out):
    # show plot
    # def _process_fig(self, freq, lead_idx, fig, axs, field, units, xlim, ylim, path_out)
    lead_time_hours = int(freq[:-1]) * lead_idx
    fig.suptitle(field + f" ({time.strftime('%Y-%m-%dT%H')}) + {lead_time_hours}h")
    plt.tight_layout()
    cbax = fig.colorbar(im, ax=axs.ravel().tolist())
    cbax.set_label(f"{field} ({units})")
    """
    if path_out is not None:
        standard = map_keys[field]['standard']
        extent = 'global' if (xlim is None or  ylim is None) else f'lim{xlim[0]}-{xlim[1]}-{ylim[0]}-{ylim[1]}'
        filename = f"{path_out}/{standard}_{time.strftime('%Y%m%d%H')}_+{lead_time_hours:03d}_{extent}.png"
        #plt.savefig(filename)
    """
    plt.savefig(path_out)
    plt.show()
    #plt.close()

class FieldPlotter:
    """Plot ensemble field and potentially compare to ERA5."""
    def __init__(
            field: str,
            time: str or pd.Timestamp,
            path: str or list[str],
            ens_size: int = None,
            resolution: float = None,
            freq: str = '6h',
            file_prefix: str = "",
            swap_axes: bool = False,
            **kwargs,
        ) -> None:
        """
        Args:
            field: str
                Specify a field to be verified. Currently supports
                air_temperature_2m, wind_speed_10m, precipitation_amount_acc6, air_sea_level_pressure
            time: str or pd.Timestamp
                Specify a time stamps to be plotted
            path: str or list[str]
                Path to directory where files to be analysed are found.
                If several paths are provided, comparison mode is invoked
                (maybe add information about NetCDF format and folder structure?)
            file_era: str
                ERA5 analysis file to be compared to. Not included by default.
            lead_times: list[int] or int
                One or multiple lead times to be plotted
            ens_size: int
                Number of ensemble members to include, leave as None for deterministic models.
                (switch to member list to be able to choose members, in case some are corrupt)
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
        self.paths = np.atleast_1d(path)
        if isinstance(time, str):
            time = pd.Timestamp(time)
        self.time = time
        self.ens_size = ens_size
        self.file_prefix = file_prefix

        path_features = _get_path_features(resolution)

        # process inputs
        lead_times = np.atleast_1d(lead_times)

        """
        # extract path-specific features
        # path_features = _get_path_features(...)
        path_features = []
        for path in paths:
            features = {}
            # get (ensemble) data
            ds = get_data(path, time, ens_size, file_prefix)
            features['ds'] = ds

            if resolution is None:
                resolution = 0.25 if ds[fields[0]].shape[-1] == 542080 else 1
            features['resolution'] = resolution

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
                features['lon'] = lon
                features['lat'] = lat
            elif ds.latitude.ndim==2:
                regular = True
                lat_grid, lon_grid = ds.y, ds.x #ds.latitude, ds.longitude
                lat_center = float((ds.latitude.max()-ds.latitude.min()) / 2.)
                lon_center = float((ds.longitude.max()-ds.longitude.min()) / 2.)
            else:
                raise ValueError
            features['regular'] = regular
            features['lat_grid'] = lat_grid
            features['lon_grid'] = lon_grid
            path_features.append(features)
        """


    def plot(
            self, 
            field: str, 
            lead_times: list[int] or int = 0,
            lead_times, 
            file_ref: str = None, 
            plot_ens_mean: bool = False,
            norm: bool = False,
            xlim: tuple[float] = None,
            ylim: tuple[float] = None,
            path_out: str = None,
            pressure_contour: bool = False,
            show: bool = True, 
            save: str = None, 
            swap_axes: bool = False,
        ) -> None:
        self.units = map_keys[field]['units']

        include_ref = False if file_ref is None else True
        if include_ref:
            data_truth, lat_grid_truth, lon_grid_truth = _get_ref_features(file_ref)
        
        """
        # Reference
        # ref_features = _get_ref_features(...)
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
        n, ens_size, dim_idx, ref_dim = panel_daemon(len(paths), len(lead_times), len(members), plot_ens_mean, include_ref, swap_axes)
        """
        # choose panel grid layout
        num_path = len(paths)
        comparison = num_path > 1
        if comparison:
            # comparison mode
            n = (num_path, ens_size + plot_ens_mean + include_truth)
        else:
            n, ens_size = panel_config_auto(ens_size, include_truth + plot_ens_mean)
        """

        # find vmin and vmax and add values to kwargs
        vmin = +np.inf
        vmax = -np.inf
        if include_truth:
            vmin = min(vmin, data_truth[field][lead_idx].min())
            vmax = max(vmax, data_truth[field][lead_idx].max())
        for path_idx, features in enumerate(path_features):
            ds = features['ds']
            vmin = min(vmin, float(ds[field][:,lead_idx].min()))
            vmax = min(vmax, float(ds[field][:,lead_idx].max()))
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

        # define fig and axes
        if regular:
            #projection = ccrs.LambertConformal(central_longitude=lon_center, central_latitude=lat_center, standard_parallels=(63.3, 63.3))
            projection = ccrs.LambertConformal(lon_center, lat_center, standard_parallels=(63.3, 63.3))
        else:
            projection = ccrs.PlateCarree()
        fig, axs = plt.subplots(*n, figsize=(8,6), squeeze=False, subplot_kw={'projection': projection})
            
        # actual plotting
        k = 0
        idx = [0,0,0] # model_idx, lt_idx, ens_idx
        for i in range(n[0]):
            idx[dim_idx[0]] = i
            for j in range(n[1]):
                idx[dim_idx[1]] = j
                model_idx, lt_idx, ens_idx = idx

                features = path_features[model_idx]
                ds = features['ds']
                resolution = features['resolution']
                regular = features['regular']
                lon = features['lon']
                lat = features['lat']
                lon_grid = features['lon_grid']
                lat_grid = features['lat_grid']

                data = ds[field][ens_idx, lead_idx]
                data_pressure = ds['air_pressure_at_sea_level'][ens_idx, lead_idx] if pressure_contour is not None else None
                im = _plot_panel(axs[i,j], data, lat_grid, lon_grid, data_pressure, lat, lon, xlim, ylim, title="", **kwargs)
                
                """
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
                """
                k += 1
                if k >= ens_size:
                    break
            else:
                continue
            break

        # extra panels
        if plot_ens_mean:
            """
            # data, data_pressure, ax, lat, lon, lat_grid, lon_grid, title
            # @staticmethod
            # def _plot_panel(data, data_pressure, ax, lat_grid, lon_grid, lon=None, lat=Lone, title="")
            # self._plot_panel(data, data_pressure, ax, 
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
            """
            data = ds[field][:, lead_idx].mean(axis=0)
            data_pressure = ds['air_pressure_at_sea_level'][:, lead_idx].mean(axis=0) if pressure_contour is not None else None
            im = _plot_panel(axs[i,j], data, lat_grid, lon_grid, data_pressure, lat, lon, xlim, ylim, title="ens mean", **kwargs)

        if include_ref:
            data = data_truth[field][lead_idx]
            data_pressure = data_truth['air_pressure_at_sea_level'][lead_idx] if pressure_contour is not None else None
            im = _plot_panel(axs[i,j], data, lat_grid_truth, lon_grid_truth, data_pressure, ds_truth.latitudes, ds_truth.longitudes,, xlim, ylim, title="ref", **kwargs)
            """
            data = data_truth[field][lead_idx]
            if data.ndim == 1:
                data = interpolate(data, ds_truth.latitudes, ds_truth.longitudes, resolution)
            if pressure_contour:
                data_contour = data_truth['air_pressure_at_sea_level'][lead_idx]
                if data_contour.ndim == 1:
                    data_contour = interpolate(data_contour, ds_truth.latitudes, ds_truth.longitudes, resolution)

            last_ax = axs[n[0]-1, n[1]-1]
            im = plot(last_ax, data, lon_grid_truth, lat_grid_truth, contour=data_contour, **kwargs)
            last_ax.set_title("MEPS")
            last_ax.set_xlim(xlim)
            last_ax.set_ylim(ylim)
            """

        _process_fig(freq, lead_idx, fig, axs, field, units, path_out)
        """
        # show plot
        # def _process_fig(self, freq, lead_idx, fig, axs, field, units, xlim, ylim, path_out)
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
        """


if __name__ == "__main__":

    import matplotlib

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])

    field_plotter(
        time="2022-03-05T00", 
        fields=['precipitation_amount_acc6h'], #['air_temperature_2m', 'wind_speed_10m', 'precipitation_amount_acc6h', 'air_pressure_at_sea_level'], 
        #path="/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_roll2_warmup10/inference/epoch_009/predictions/", 
        path=[
            "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_077_10mem_1year/predictions/", 
            "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_077_10mem_1year/predictions/", 
        ],
        #file_truth="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr", 
        file_truth="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", 
        lead_times=2, #[1,2,12, 16, 20],
        ens_size=3,
        plot_ens_mean=False,
        cmap=cmap,
        norm=True,
        #xlim=(-10,32),
        #ylim=(25,73),
        #xlim=(-6e5,-2e5),
        #ylim=(-6e5,0),
        resolution=0.25,
        path_out = '/pfs/lustrep3/scratch/project_465000454/nordhage/verification/ni3_c',
        pressure_contour=False,
        file_prefix='meps',
    )
