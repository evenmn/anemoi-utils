import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"

def spread_error(
        time: str, 
        fields: str,
        path: str,
        path_era: str, 
        lead_time: int = slice(None),
        ens_size: int = None,
        **kwargs,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    """
    fields = np.atleast_1d(fields)
    if isinstance(lead_time, int):
        lead_time = range(lead_time)

    # get (ensemble) data
    data_dict = get_data(path, time, fields, lead_time, ens_size)
    flat, resolution = resolution_auto(data_dict)
    lead_times = data_dict[list(data_dict.keys())[0]].shape[1]

    # ERA5
    data_era5 = get_era5_data(path_era, time, fields, range(lead_times), resolution)

    for field in fields:
        units = map_keys[field]['units']

        data = data_dict[field]
        ens_mean = data.mean(axis=0)
        ens_std = data.std(axis=0)

        rmse = np.sqrt((ens_mean - data_era5[field])**2)

        #ens_std = ((ens_size+1)/(ens_size-1)) * ens_std.mean(axis=-1)
        ens_std = ens_std.mean(axis=-1)
        rmse = rmse.mean(axis=-1)

        lead_times = 6 * np.arange(lead_times)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(lead_times, ens_std, '-o', c='dodgerblue', label='STD')
        ax.plot(lead_times, rmse, '-o', c='g', label='RMSE')
        ax.legend(loc='best')
        ax.set_xlabel("Lead times (hours)")
        ax.set_ylabel(fr"Ensemble RMSE/STD ({units})")
        fig.suptitle(field)
        plt.tight_layout()
        #plt.savefig('/leonardo_work/DestE_330_24/enordhag/ens-score-anemoi/spread_error.png')
        plt.show()

if __name__ == "__main__":
    #fields = ['air_temperature_2m']
    #fields = ['precipitation_amount_acc6h']
    fields = ['wind_speed_10m']

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_leo/inference/epoch_010/predictions/"
    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/hollow_rainbow_10/inference/epoch_010/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    spread_error("2022-01-13T00", fields, path, path_era, ens_size=4)
