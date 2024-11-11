import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"

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

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    field_plotter("2022-09-02T00", fields, path, path_era, lead_times=[5,12], ens_size=4, plot_ens_mean=True, cmap='turbo')
