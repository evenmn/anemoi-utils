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


def panel_daemon(num_models, num_lead_times, ens_size, plot_ens_mean, include_ref, swap_axes=False):
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
    - ens_mean: tuple(int)
        Indicates which dimension or dimensions to put ens_mean
        (all ens_mean, x, y)
    - ref_dim: tuple(int)
        dimension to append reference along, (x, y)
    """
    if ens_size is None:
        # num_members can not be set (None) for two reasons:
        # 1. deterministic forecast, set num_members to 1
        # 2. plotting ensemble mean only, set num_members to 0
        if plot_ens_mean:
            ens_size = 0
        else:
            ens_size = 1

    # always append ens mean panel in ensemble dim
    num_members = ens_size + plot_ens_mean

    # assert correct dimensionalities
    dim_len = np.array([num_models, num_lead_times, num_members])
    assert not 0 in dim_len, "Cannot deal with zero dimension"
    assert 1 in dim_len, "At least one dimension needs length 1"

    # check dimensionality
    active_dim = dim_len > 1  # which dimension has more than 1 element?
    num_dims = active_dim.sum()  # number of dimensions longer than 1
    dim_len_argsort = np.argsort(dim_len) # putting longest dimension last

    idx = dim_len_argsort[-2:]  # dimension indices longer than 1 element
    n = dim_len[idx]  # number of panels in each direction

    print('active_dim:', active_dim)
    print('num_dims:', num_dims)
    print('idx:', idx)
    print('n:', n)

    ens_mean = [None, None, None]
    if plot_ens_mean:
        if idx[0] == 2:
            ens_mean[1] = num_members-1
        elif idx[1] == 2:
            ens_mean[2] = num_members-1
        else:
            ens_mean[0] = 0

    ref = [None, None]
    if include_ref:
        # add ref along longest dim, but not lead times
        if idx[0] != 1:
            ref_dim = 0
        else:
            ref_dim = 1
        ref[ref_dim] = n[ref_dim]
        n[ref_dim] += 1

    # rearrange to 2d if 1d
    if num_dims < 2:
        conf_map = [None,
                    (1,1), (1,2), (2,2), (2,2),
                    (2,3), (2,3), (2,4), (2,4),
                    (3,3), (3,4), (3,4), (3,4),
                    (4,4), (4,4), (4,4), (4,4),
                   ]
        panel_limit = len(conf_map) - 1

        if n[1] > panel_limit:
            raise ValueError(f"Panel limit reached!")
        
        if plot_ens_mean:
            ens_mean = conf_map[ens_mean[-1]]  # this is not correct, swap only the last two dims
        if include_ref:
            ref = conf_map[ref[-1]]
        n = conf_map[n[-1]]
        idx = idx[1:]

        print(ens_mean)
        print(ref)
        print(n)


    if swap_axes:
        n = reversed(n)
        idx = reversed(idx)
        ref = reversed(ref)
        ens_mean = reversed(ens_mean) # this is not correct, swap only the last two dims
    
    return tuple(n), tuple(idx), tuple(ens_mean), tuple(ref)


def _plot_panel(ax, data, lat_grid, lon_grid, data_contour=None, lat=None, lon=None, xlim=None, ylim=None, titlex="", titley="", resolution=None, **kwargs):
    """Plot a single panel. Interpolate if necessary."""
    if data.ndim == 1:
        print('resolution:', resolution)
        print('data.shape, _plot_panel:', data.shape)
        data = interpolate(data, lat, lon, resolution)
        print('data.shape, _plot_panel:', data.shape)

    if data_contour is not None:
        if data_contour.ndim == 1:
            data_contour = interpolate(data_contour, lat, lon, resolution)

    im = plot(ax, data, lon_grid, lat_grid, contour=data_contour, **kwargs)
    ax.set_title(titlex)
    ax.set_ylabel(titley)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im


def _process_fig(fig, axs, im, field, time, units, path_out, show):
    """Add super title to figure, colorbar, potentially save and show figure.

    Return fig such that it can be used outside class
    """
    fig.suptitle(time.strftime('%Y-%m-%dT%H'))
    plt.tight_layout()
    cbax = fig.colorbar(im, ax=axs.ravel().tolist())
    cbax.set_label(f"{field} ({units})")
    if path_out is not None:
        plt.savefig(path_out)
    if show:
        plt.show()
    return fig

class FieldPlotter:
    """Plot ensemble field and potentially compare to ERA5."""
    def __init__(
            self,
            time: str or pd.Timestamp,
            path: str or list[str],
            members: int or list[int] = None,
            resolution: float = None,
            freq: str = '6h',
            file_prefix: str = "",
            model_labels: list[str] = None,
            xlim: tuple[float] = None,
            ylim: tuple[float] = None,
            latlon_units: str = 'deg',
        ) -> None:
        """
        Args:
            time: str or pd.Timestamp
                Specify a time stamps to be plotted
            path: str or list[str]
                Path to directory where files to be analysed are found.
                If several paths are provided, comparison mode is invoked
                (maybe add information about NetCDF format and folder structure?)
            ens_size: int
                Number of ensemble members to include, leave as None for deterministic models.
                (switch to member list to be able to choose members, in case some are corrupt)
            resolution: float
                Resolution in degrees used in interpolation. Using 1 for o96 and 0.25 for n320 by default.
            freq: str
                Frequency of lead times. Supports pandas offset alias: 
                https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
            file_prefix: str
                filenames prefix, to use if there are multiple files at same date
            model_labels: list[str]
                Labels associated with paths to be plotted as panel titles
            xlim: tuple[float]
                limit x-range of data. No limit by default
            ylim: tuple[float]
                limit y-range of data. No limit by default
        """
        self.members = np.atleast_1d(members)
        self.ens_size = len(self.members)
        self.resolution = resolution

        self.paths = np.atleast_1d(path)
        self.model_labels = model_labels
        if self.model_labels is not None:
            self.model_labels = np.atleast_1d(self.model_labels)
            assert len(self.model_labels) == len(self.paths), "Number of model labels must be equal to number of models" 

        if isinstance(time, str):
            time = pd.Timestamp(time)
        self.time = time
        self.freq = freq
        self.file_prefix = file_prefix

        self.xlim = xlim
        self.ylim = ylim

        if latlon_units == 'rad':
            rad = True
        elif latlon_units == 'deg':
            rad = False
        else:
            raise NotImplementedError(f"Unknown latlon_units '{latlon_units}'!")

        self.path_features = self._get_path_features(resolution, rad)

    def _get_path_features(self, resolution, rad):
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
                resolution = 0.25 #if ds[field].shape[-1] == 542080 else 1
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
                if rad:
                    lon = np.rad2deg(lon)
                    lat = np.rad2deg(lat)
                lat_grid, lon_grid = mesh(lat, lon, resolution) 
                features['lon'] = lon
                features['lat'] = lat
            elif ds.latitude.ndim==2:
                regular = True
                lat_grid, lon_grid = ds.y, ds.x #ds.latitude, ds.longitude
                lat_center = float(ds.latitude.max()+ds.latitude.min()) / 2.
                lon_center = float(ds.longitude.max()+ds.longitude.min()) / 2.
                features['lon_center'] = lon_center
                features['lat_center'] = lat_center
                features['lon'] = None
                features['lat'] = None
            else:
                raise ValueError
            features['regular'] = regular
            features['lat_grid'] = lat_grid
            features['lon_grid'] = lon_grid
            path_features.append(features)
        return path_features

    def _get_ref_features(self, file_ref, pressure_contour, ds, regular, resolution, rad):
        """Reference features."""
        fields = [self.field]
        if pressure_contour:
            fields.append('air_pressure_at_sea_level')
        max_lead_time = max(self.lead_times)+1
        ds_ref = read_era5(fields, file_ref, [self.time], max_lead_time, freq=self.freq)
        data_ref = get_era5_data(ds_ref, 0, fields, max_lead_time)
        if regular:
            y, x = ds.latitude.shape
            for key, value in data_ref.items():
                data_ref[key] = value.reshape(-1, y, x)
            lat_grid_ref = ds.y
            lon_grid_ref = ds.x
        else:
            lat_grid_ref, lon_grid_ref = mesh(ds_ref.latitudes, ds_ref.longitudes, resolution)

        if rad:
            lat_grid_ref = np.rad2deg(lat_grid_ref)
            lon_grid_ref = np.rad2deg(lon_grid_ref)
        return data_ref, ds_ref, lat_grid_ref, lon_grid_ref

    def plot(
            self,
            field: str, 
            lead_times: list[int] or int = 0,
            file_ref: str = None, 
            plot_ens_mean: bool = False,
            norm: bool = False,
            xlim: tuple[float] = None,
            ylim: tuple[float] = None,
            path_out: str = None,
            pressure_contour: bool = False,
            show: bool = True, 
            swap_axes: bool = False,
            ref_label: str = 'ref',
            ref_units: str = 'deg',
            **kwargs,
        ) -> None:
        """
        Args:
            field: str
                Specify a field to be verified. Currently supports
                air_temperature_2m, wind_speed_10m, precipitation_amount_acc6, air_sea_level_pressure
            lead_times: list[int] or int
                One or multiple lead times to be plotted
            file_ref: str
                Reference file to be compared to. Not included by default.
            plot_ens_mean: bool
                Whether or not to plot ensemble mean.
            norm: bool
                Whether or not to normalize plots. In particular used with precipitation.
            xlim: tuple[float]
                xlim used in panels. No limit by default
            ylim: tuple[float]
                ylim used in panels. No limit by default
            path_out: str
                Path to where to save the image(s). If not given, images will not be saved
            pressure_contour: bool
                Whether or not to add pressure contour lines to field. Not added by default
            show: bool
                Whether or not to show plots
            swap_axes: bool
                Swap x- and y-axes to make plot vertical instead of horizontal
            ref_label: str
                Reference label to be printed as panel title
        """
        lead_times = np.atleast_1d(lead_times)
        self.field = field
        self.lead_times = lead_times

        units = map_keys[field]['units']

        if ref_units == 'rad':
            rad_ref = True
        elif ref_units == 'deg':
            rad_ref = False
        else:
            raise NotImplementedError(f"Unknown latlon_units '{latlon_units}'!")

        include_ref = False if file_ref is None else True
        if include_ref:
            data_ref, ds_ref, lat_grid_ref, lon_grid_ref = self._get_ref_features(file_ref, pressure_contour, self.path_features[0]['ds'], self.path_features[0]['regular'], self.resolution, rad_ref)
        n, dim_idx, ens_mean, ref_dim = panel_daemon(len(self.paths), len(self.lead_times), self.ens_size, plot_ens_mean, include_ref, swap_axes)

        # find vmin and vmax and add values to kwargs
        vmin = +np.inf
        vmax = -np.inf
        if include_ref:
            vmin = min(vmin, data_ref[field][lead_times].min())
            vmax = max(vmax, data_ref[field][lead_times].max())
        for path_idx, features in enumerate(self.path_features):
            ds = features['ds']
            vmin = min(vmin, float(ds[field][:,lead_times].min()))
            vmax = min(vmax, float(ds[field][:,lead_times].max()))
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
        regular = self.path_features[0]['regular'] # assume all panels have the same projection for now
        if regular:
            lon_center = self.path_features[0]['lon_center']
            lat_center = self.path_features[0]['lat_center']
            projection = ccrs.LambertConformal(lon_center, lat_center, standard_parallels=(lat_center, lat_center))
        else:
            projection = ccrs.PlateCarree()
        fig, axs = plt.subplots(*n, figsize=(8,6), squeeze=False, subplot_kw={'projection': projection})
            
        def model_label(i):
            if self.model_labels is None:
                return f'model {i}'
            if i >= len(self.model_labels):
                return ""
            return self.model_labels[i]

        def lt_label(i):
            lead_time = 6 * lead_times[i]
            return f'+{lead_time}h'

        def member_label(i):
            member_id = self.members[i]
            return f'member {member_id}'
        labels = [model_label, lt_label, member_label]

        # mapping panel to correct data
        idx = [0,0,0] # model_idx, lt_idx, ens_idx
        # actual plotting
        for i in range(n[0]):
            #if dim_idx is None:
            #    raise NotImplementedError('Need a generic way to treat the case when there is less than two dimensions')
            for j in range(n[1]):
                k = n[0] * i + j
                if len(dim_idx) == 1:
                    idx[dim_idx[0]] = k
                else:
                    idx[dim_idx[0]] = i
                    idx[dim_idx[1]] = j
                model_idx, lt_idx, ens_idx = idx
                if plot_ens_mean:
                    i_em, j_em = ens_mean
                    
                if include_ref:
                    i_ref, j_ref = ref_dim
                    if (i_ref is None or i==i_ref) and (j_ref is None or j==j_ref):
                        data = data_ref[field][lt_idx]
                        data_pressure = data_ref['air_pressure_at_sea_level'][lt_idx] if pressure_contour else None
                        label_x = ref_label if i == 0 else ''
                        label_y = labels[dim_idx[0]](i) if j == 0 else ""
                        im = _plot_panel(axs[i,j], data, lat_grid_ref, lon_grid_ref, data_pressure, ds_ref.latitudes, ds_ref.longitudes, xlim, ylim, titlex=label_x, titley=label_y, resolution=resolution, **kwargs)
                        continue
                if len(dim_idx) == 1:
                    label_x = labels[dim_idx[0]](k)
                    label_y = ""
                else:
                    label_x = labels[dim_idx[1]](j) if i == 0 else ""
                    label_y = labels[dim_idx[0]](i) if j == 0 else ""
                features = self.path_features[model_idx]
                ds = features['ds']
                resolution = features['resolution']
                regular = features['regular']
                lon = features['lon']
                lat = features['lat']
                lon_grid = features['lon_grid']
                lat_grid = features['lat_grid']

                data = ds[field][ens_idx, lt_idx]
                print('data.shape:', data.shape)
                data_pressure = ds['air_pressure_at_sea_level'][ens_idx, lt_idx] if pressure_contour else None
                im = _plot_panel(axs[i,j], data, lat_grid, lon_grid, data_pressure, lat, lon, xlim, ylim, titlex=label_x, titley=label_y, resolution=resolution, **kwargs)
        """
        if plot_ens_mean:
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
            data = ds[field][:, lead_idx].mean(axis=0)
            data_pressure = ds['air_pressure_at_sea_level'][:, lead_idx].mean(axis=0) if pressure_contour is not None else None
            im = _plot_panel(axs[i,j], data, lat_grid, lon_grid, data_pressure, lat, lon, xlim, ylim, title="ens mean", **kwargs)
        """

        fig = _process_fig(fig, axs, im, field, self.time, units, path_out, show)
        return fig, axs


if __name__ == "__main__":

    import matplotlib

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])

    fp = FieldPlotter(
        time="2023-08-07T00", 
        path=[
            #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c/inference/epoch_077/predictions/", 
            "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_kl/inference/epoch_076/predictions/", 
            "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_kl_w1e-2/inference/epoch_076/predictions/", 
            "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_kl_w1/inference/epoch_076/predictions/", 
            #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_c_safcrps_k5_s1/inference/epoch_076/predictions/", 
            #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_s0.1_mp/inference/epoch_030/predictions/",
        ],
        model_labels = ['CRPS+KL\nw=1e-4', 'CRPS+KL\nw=1e-2', 'CRPS+KL\nw=1'],
        members=0,
        #file_prefix="240hfc",
        #latlon_units='rad',
        #resolution=0.25,
    )

    fp.plot(
        field='precipitation_amount_acc6h', 
        lead_times=[0,4,8], 
        pressure_contour=True,
        cmap=cmap, 
        norm=True,
        file_ref="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/MEPS/aifs-meps-2.5km-2020-2024-6h-v6.zarr", 
        #file_ref="/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr", 
        #xlim=(-4e5,0),
        #ylim=(-6e5,0),
        #xlim=(100,180),
        #ylim=(-11, 40),
        swap_axes=True,
        ref_label='MEPS',
    )
