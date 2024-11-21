import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"

def spread_skill(
        times: str, 
        fields: str,
        paths: list[str],
        path_era: str, 
        lead_time: int = slice(None),
        ens_size: int = None,
        labels: list[str] = None,
        **kwargs,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    """
    times = np.atleast_1d(times)
    fields = np.atleast_1d(fields)
    paths = np.atleast_1d(paths)

    if isinstance(lead_time, int):
        lead_time = range(lead_time)

    # get (ensemble) data
    data_dict = get_data(paths[0], times[0], fields, lead_time, ens_size)
    flat, resolution = resolution_auto(data_dict)
    ens_size, lead_time = data_dict[list(data_dict.keys())[0]].shape[:2]  # in case lead_time is not defined

    lead_times = np.arange(lead_time)

    colors = ['green', 'dodgerblue', 'orange', 'magenta']
    if labels is None:
        labels = len(paths) * ['']

    for field in fields:
        units = map_keys[field]['units']
        fig, ax = plt.subplots(figsize=(8,5))
        fig2, ax2 = plt.subplots(figsize=(8,5))
        fig3, ax3 = plt.subplots(figsize=(8,5))
        for i, path in enumerate(paths):
            rmses = None
            ens_stds = None
            hists = None
            hists_era5 = None
            crpss = None
            for j, time in enumerate(times):
                if i > 0 and j > 0:
                    data_dict = get_data(path, time, [field], lead_times, ens_size)
                data_era5 = get_era5_data(path_era, time, [field], lead_times, resolution)
                targets = data_era5[field]
                preds = data_dict[field]

                # spread-skill
                ens_mean = preds.mean(axis=0)
                ens_std = preds.std(axis=0)
                rmse = np.sqrt(np.mean((ens_mean - targets)**2, axis=-1))
                #ens_std = ((ens_size+1)/(ens_size-1)) * ens_std.mean(axis=-1)
                if ens_stds is None:
                    ens_stds = ens_std.mean(axis=-1)
                    rmses = rmse
                else:
                    ens_stds += ens_std.mean(axis=-1)
                    rmses += rmse

                # histogram
                hist, _ = np.histogram(preds, bins=[0, 0.5, 1, 2, 4, 8, 16, 32])
                hist_era5, _ = np.histogram(targets, bins=[0, 0.5, 1, 2, 4, 8, 16, 32])

                print(hist.shape)
                if hists is None:
                    hists = hist
                    hists_era5 = hist_era5
                else:
                    hists += hist
                    hists_era5 += hist_era5

                # CRPS
                mae = np.mean(np.abs(targets[None]-preds), axis=0)
                ens_var = np.zeros(preds.shape[1:])
                for k in range(ens_size):
                    ens_var += np.sum(np.abs(preds[k][None] - preds[k+1:]), axis=0)
                ens_var *= -1.0 / (ens_size * (ens_size - 1))  # fair CRPS
                crps = mae + ens_var

                if crpss is None:
                    crpss = crps
                else:
                    crpss += crps

            rmse = rmses / len(times)
            ens_std = ens_stds / len(times)
            crps = crpss / len(times)

            ax.plot(6 * lead_times, ens_std, '-o', c=colors[i], label='STD ' + labels[i])
            ax.plot(6 * lead_times, rmse, '--o', c=colors[i], label='RMSE ' + labels[i])

            print(hists.shape)
            print(hists_era5.shape)
            ax2.hist(hists)

            ax3.hist(crps)
        ax2.hist(hists_era5)
        ax.legend(loc='best')
        ax.set_xlabel("Lead times (hours)")
        ax.set_ylabel(fr"Spread-skill ({units})")
        fig.suptitle(field)
        plt.tight_layout()
        #plt.savefig('/leonardo_work/DestE_330_24/enordhag/ens-score-anemoi/spread_error.png')
        plt.show()

if __name__ == "__main__":
    #fields = ['air_temperature_2m']
    #fields = ['precipitation_amount_acc6h']
    fields = ['wind_speed_10m']

    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    paths = [
        "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_2/inference/epoch_010_short/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_leo/inference/epoch_010/predictions/",
    ]
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    spread_skill(["2022-01-15T00", "2022-02-15T00", "2022-03-15T00"], fields, paths, path_era, ens_size=4, lead_time=40, labels=['spatial noise'])
