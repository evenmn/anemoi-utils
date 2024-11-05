from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

times = [
    "2022-01-02T00",
    "2022-02-02T00",
    "2022-03-02T00",
    "2022-04-02T00",
    "2022-05-02T00",
    "2022-06-02T00",
    "2022-07-02T00",
    "2022-08-02T00",
    "2022-09-02T00",
    "2022-10-02T00",
    #"2022-11-02T00",
]


spread_means = []
mse_means = []
for time in tqdm(times):
    mse_mean = np.loadtxt(f"files/mse_mean_{time}.npy")
    spread_mean = np.loadtxt(f"files/spread_mean_{time}.npy")

    mse_mean = np.sqrt(mse_mean)
    spread_mean = np.sqrt(spread_mean)

    spread_means.append(spread_mean)
    mse_means.append(mse_mean)

spread_mean = np.mean(spread_means, axis=0)
mse_mean = np.mean(mse_means, axis=0)
lead_times_days = 0.25*np.arange(len(mse_mean))

# plot
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(lead_times_days, spread_mean, '-o', mec='k', label='STD')
ax.plot(lead_times_days, mse_mean, '-o', mec='k', label='RMSE')
ax.legend(loc='best')
ax.set_xlabel("Lead times (days)")
ax.set_ylabel(r"Ensemble RMSE/STD ($K$)")
ax.set_xticks(lead_times_days.astype(int))
fig.suptitle("air_temperature_2m")
plt.tight_layout()
plt.savefig('/leonardo_work/DestE_330_24/enordhag/ens-score-anemoi/spread_error.png')
plt.show()

