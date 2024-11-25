from multiprocessing import Pool

import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

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
        lead_time = np.arange(lead_time)

    # get (ensemble) data
    pool =  Pool(ens_size)
    data_dict = get_data(paths[0], times[0], fields, lead_time, ens_size, pool)
    flat, resolution = resolution_auto(data_dict)
    ens_size, lead_time = data_dict[list(data_dict.keys())[0]].shape[:2]  # in case lead_time is not defined

    lead_times = np.arange(lead_time)

    colors = ['orange', 'dodgerblue', 'green', 'orange', 'magenta']

    bins = 'auto'
    hist_leadtime = 39
    bins2 = np.array([0, 0.5, 1, 2, 4, 8, 16, 32, 64])
    bins_center2 = (bins2[1:]+bins2[:-1]) / 2

    if labels is None:
        labels = len(paths) * ['']

    for field in fields:
        units = map_keys[field]['units']
        fig, ax = plt.subplots(figsize=(8,5))
        fig2, ax2 = plt.subplots(figsize=(8,5))
        fig3, ax3 = plt.subplots(figsize=(8,5))
        fig4, ax4 = plt.subplots(figsize=(8,5))
        fig5, ax5 = plt.subplots(figsize=(8,5))
        for i, path in enumerate(paths):
            print('path:', path)
            rmses = None
            ens_stds = None
            hists = None
            hists_era5 = None
            hists2 = None
            hists2_era5 = None
            crpss = None
            maes = None
            ens_vars = None
            for j, time in tqdm(enumerate(times)):
                print('time:', time)

                # load data
                if i > 0 or j > 0:
                    data_dict = get_data(path, time, [field], lead_times, ens_size, pool)
                data_era5 = get_era5_data(path_era, time, [field], lead_times, resolution)
                targets = data_era5[field]
                preds = data_dict[field]

                # spread-skill
                ens_mean = preds.mean(axis=0)
                ens_std = preds.std(axis=0)
                rmse = np.sqrt(np.mean((ens_mean - targets)**2, axis=-1))
                if ens_stds is None:
                    ens_stds = ((ens_size+1)/(ens_size-1))*ens_std.mean(axis=-1)
                    rmses = rmse
                else:
                    ens_stds += ((ens_size+1)/(ens_size-1))*ens_std.mean(axis=-1)
                    rmses += rmse

                # histogram
                hist, bins = np.histogram(preds[:,hist_leadtime], bins=bins)
                hist_era5, _ = np.histogram(targets[hist_leadtime], bins=bins)

                if hists is None:
                    hists = hist
                    hists_era5 = hist_era5
                else:
                    hists += hist
                    hists_era5 += hist_era5

                # histogram (uneven bins)
                hist2, _ = np.histogram(preds[:,hist_leadtime], bins=bins2)
                hist2_era5, _ = np.histogram(targets[:,hist_leadtime], bins=bins2)

                if hists2 is None:
                    hists2 = hist2
                    hists2_era5 = hist2_era5
                else:
                    hists2 += hist2
                    hists2_era5 += hist2_era5

                # CRPS
                mae = np.mean(np.abs(targets[None]-preds), axis=0)
                ens_var = np.zeros(preds.shape[1:])
                for k in range(ens_size):
                    ens_var += np.sum(np.abs(preds[k][None] - preds[k+1:]), axis=0)
                ens_var *= -1.0 / (ens_size * (ens_size - 1))  # fair CRPS
                crps = mae + ens_var
                crps = crps.mean(axis=-1)

                if crpss is None:
                    crpss = crps
                else:
                    crpss += crps

            rmse = rmses / len(times)
            ens_std = ens_stds / len(times)
            crps = crpss / len(times)
            hist = hists / len(times) / ens_size
            hist_era5 = hists_era5 / len(times)
            hist2 = hists2 / len(times) / ens_size
            hist2_era5 = hists2_era5 / len(times)

            ax.plot(6 * lead_times, ens_std, '-', c=colors[i], mec='k', marker='.', label='STD ' + labels[i])
            ax.plot(6 * lead_times, rmse, '--', c=colors[i], mec='k', marker='.', label='RMSE ' + labels[i])

            bins_center = (bins[1:] + bins[:-1]) / 2
            ax2.plot(bins_center, hist, c=colors[i], label=f'{labels[i]} +{6*hist_leadtime+6}h')
            ax4.plot(bins2[1:], hist2, c=colors[i], label=f'{labels[i]} +{6*hist_leadtime+6}h')

            ax3.plot(6 * lead_times, crps, '-', c=colors[i], mec='k', marker='.', label=labels[i])
        ax2.plot(bins_center, hist_era5, '--k', label='ERA5')
        ax4.plot(bins2[1:], hist2_era5, '--k', label='ERA5')
        ax2.legend(loc='best')
        ax3.legend(loc='best')
        ax4.legend(loc='best')
        ax.legend(loc='best')
        ax.set_xlabel("Lead times (hours)")
        ax.set_ylabel(f"Spread-skill ({units})")
        ax2.set_xlabel(f"{field} ({units})")
        ax2.set_ylabel("Observation frequency")
        ax4.set_xlabel(f"{field} ({units})")
        ax4.set_ylabel("Observation frequency")
        ax4.set_xscale('log')
        #ax2.set_xticks(range(len(bins_center)))
        #ax2.set_xticks(bins_center)
        ax3.set_xlabel("Lead times (hours)")
        ax3.set_ylabel(f"CRPS ({units})")
        fig.suptitle(field)
        plt.tight_layout()
        #plt.savefig('/leonardo_work/DestE_330_24/enordhag/ens-score-anemoi/spread_error.png')
        plt.show()
    pool.close()
    pool.join()

if __name__ == "__main__":
    #fields = ['air_temperature_2m']
    #fields = ['precipitation_amount_acc6h']
    fields = ['wind_speed_10m']
    times = [
        "2022-01-02T00",
        #"2022-01-17T00",
        #"2022-02-02T00",
        #"2022-02-17T00",
        #"2022-03-02T00",
        #"2022-03-17T00",
        #"2022-04-02T00",
        #"2022-04-17T00",
        #"2022-05-02T00",
        #"2022-05-17T00",
        #"2022-06-02T00", 
        #"2022-06-17T00", 
        #"2022-07-02T00",
        #"2022-07-17T00",
        #"2022-08-02T00",
        #"2022-08-17T00",
        #"2022-09-02T00",
        #"2022-09-17T00",
        #"2022-10-02T00",
        #"2022-10-17T00",
        #"2022-11-02T00",
        #"2022-11-17T00",
    ]

    paths = [
        "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/",
        "/leonardo_work/DestE_330_24/anemoi/experiments/ni1_a_lumi/inference/epoch_099/predictions/",
    ]
    #paths = [
    #    "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_2/inference/epoch_010_short/predictions/",
        #"/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_leo/inference/epoch_010/predictions/",
    #]
    #path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"
    path_era = "/leonardo_work/DestE_330_24/anemoi/datasets/ERA5/"
    spread_skill(times, fields, paths, path_era, ens_size=8, lead_time=40, labels=['Default Noise MLP', 'Encoded noise'])
