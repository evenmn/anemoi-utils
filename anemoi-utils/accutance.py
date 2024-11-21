import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_data, get_era5_data
from utils import *

plt.rcParams["font.family"] = "serif"

def calculate_accutance(image):
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array")
                    
    # Normalize the image to range [0, 1]
    image = image / np.max(image).astype(np.float64)

    # Calculate the gradients using central differences
    dx = np.diff(image, axis=0, append=image[-1:, :])
    dy = np.diff(image, axis=1, append=image[:, -1:])
                                        
    # Compute the gradient magnitude
    gradient_magnitude = np.hypot(dx, dy)
                                                    
    # Compute the accutance as the average of the gradient magnitudes
    accutance = np.mean(gradient_magnitude)
                                                            
    return accutance

def accutance(
        times: str, 
        fields: str,
        path: str,
        path_era: str = None, 
        lead_times: str = [0],
        ens_size: int = None,
        plot_ens_mean = False,
    ) -> None:
    """Plot ensemble field and potentially compare to ERA5.
    Support for ensemble members.
    """
    fields = np.atleast_1d(fields)
    lead_times = np.atleast_1d(lead_times)

    # get (ensemble) data
    data_dict = get_data(path, times[0], fields, lead_times, ens_size)
    flat, resolution = resolution_auto(data_dict)

    lats, lons = np.load(f"/pfs/lustrep3/scratch/project_465000454/nordhage/anemoi-utils/files/coords_{resolution}.npy")

    # ERA5
    include_era = False if path_era is None else True

    for field in fields:
        accutances = np.empty((len(lead_times), ens_size + include_era + plot_ens_mean))
        for i, time in tqdm(enumerate(times)):
            if include_era:
                data_era5 = get_era5_data(path_era, time, [field], lead_times, resolution)
            if i > 0:
                data_dict = get_data(path, time, [field], lead_times, ens_size)
            units = map_keys[field]['units']
            for lead_idx, lead_time in tqdm(enumerate(lead_times)):
                # member panel(s)
                labels = []
                for k in range(ens_size):
                    data = data_dict[field][k, lead_idx]
                    if flat:
                        data = interpolate(data, lats, lons, resolution)

                    accutance = calculate_accutance(data)
                    accutances[lead_idx, k] += accutance
                    labels.append(f'Member {k}')

                # extra panels
                if plot_ens_mean:
                    data = data_dict[field][:,lead_idx].mean(axis=0)
                    if flat:
                        data = interpolate(data, lats, lons, resolution)
                    accutance = calculate_accutance(data)
                    accutances[lead_idx, k+1] += accutance
                    labels.append('Ensemble mean')

                if include_era:
                    data = data_era5[field][lead_time]
                    data = interpolate(data, lats, lons, resolution)
                    accutance = calculate_accutance(data)
                    accutances[lead_idx, k+2] += accutance
                    labels.append('ERA5')

        # show plot
        for i in range(len(labels)):
            plt.plot(6 * np.array(lead_times), accutances[:,i], '-o', label=labels[i])
        plt.legend(loc='best')
        plt.title(field)
        plt.xlabel("Lead times (h)")
        plt.ylabel("Accutance")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    fields = ['air_temperature_2m']
    fields = ['precipitation_amount_acc6h']
    #fields = ['wind_speed_10m']


    #path = "/leonardo_work/DestE_330_24/anemoi/experiments/ni2_stage_a/inference/epoch_099/predictions/"
    path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni3_b_fix/inference/epoch_010_short/predictions/"
    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/hollow_rainbow_10/inference/epoch_010/predictions/"
    #path = "/pfs/lustrep3/scratch/project_465000454/anemoi/experiments/ni1_b_leo/inference/epoch_010/predictions/"
    path_era = "/pfs/lustrep3/scratch/project_465000454/anemoi/datasets/ERA5/"

    accutance(["2022-01-13T00", "2022-01-14T00"], fields, path, path_era, lead_times=range(12), ens_size=4, plot_ens_mean=True)
