import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"

def field_comparison(
        time: str, 
        fields: str,
        paths: str,
        path_era: str = None, 
        lead_times: str = [0],
        ens_size: int = None,
        plot_ens_mean: bool = False,
        norm: bool = False,
        **kwargs,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    """
    paths = np.atleast_1d(paths)
    fields = np.atleast_1d(fields)
    lead_times = np.atleast_1d(lead_times)

    # get resolution, assuming all paths have the same
    data_dict = get_data(paths[0], time, fields, lead_times, ens_size)
    flat, resolution = resolution_auto(data_dict)

    # ERA5
    include_era = False if path_era is None else True
    if include_era:
        data_era5 = get_era5_data(path_era, time, fields, lead_times, resolution)

    #n, ens_size = panel_config_auto(ens_size, include_era + plot_ens_mean)
    n = (ens_size + plot_ens_mean + include_era, len(paths))
    
    lat_grid, lon_grid = mesh(resolution)
    lats, lons = np.load(f"/pfs/lustrep3/scratch/project_465000454/nordhage/anemoi-utils/files/coords_{resolution}.npy")

    for field in fields:
        units = map_keys[field]['units']
        for lead_idx, lead_time in enumerate(lead_times):
            fig, axs = plt.subplots(*n, figsize=(6,4), squeeze=False, subplot_kw={'projection': ccrs.PlateCarree()})

            # find vmin and vmax
            vmin = np.inf
            vmax = -np.inf
            if include_era:
                vmin = min(vmin, data_era5[field][lead_time].min())
                vmax = max(vmax, data_era5[field][lead_time].max())
            #for data_dict in data_list:
            vmin = min(vmin, data_dict[field][:,lead_idx].min())
            vmax = max(vmax, data_dict[field][:,lead_idx].max())
            cen = (vmax-vmin)/10.
            vmin += cen
            vmax -= cen
            if norm:
                boundaries = np.logspace(0.001, np.log(0.04*vmax), cmap.N-1)
                norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, extend='both')
                kwargs['norm'] = norm
            else:
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            kwargs['shading'] = 'auto'

            for i, path in enumerate(paths):
                if i > 0:
                    data_dict = get_data(path, time, fields, lead_times, ens_size)
                # member panel(s)
                for j in range(ens_size):
                    data = data_dict[field][j,lead_idx]
                    if flat:
                        data = interpolate(data, lats, lons, resolution)

                    # plot
                    im = plot(axs[j,i], data, lat_grid, lon_grid, **kwargs)
                    axs[j,i].set_title(f"Member {j}")

                # extra panels
                if plot_ens_mean:
                    data = data_dict[field][:,lead_idx].mean(axis=0)
                    if flat:
                        data = interpolate(data, lats, lons, resolution)
                    im = plot(axs[j+1,i], data, lat_grid, lon_grid, **kwargs)
                    axs[j+1,i].set_title("Ensemble mean")
                del data_dict, data

            if include_era:
                data = data_era5[field][lead_time]
                data = interpolate(data, lats, lons, resolution)
                im = plot(axs[j+2,i], data, lat_grid, lon_grid, **kwargs)
                axs[j+2,0].set_title("ERA5")

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
    fields = ['precipitation_amount_acc6h']

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", ["white", "white", "#3c78d8", "#00ffff", "#008800", "#ffff00", "red"])

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    paths = [
        "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_leo/inference/epoch_010/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/silly_variable_10/inference/epoch_010/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/hollow_rainbow_10/inference/epoch_010/predictions/",
        ]
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    field_comparison("2022-01-13T00", fields, paths, path_era, lead_times=[12], ens_size=2, plot_ens_mean=True, cmap=cmap, norm=True)
