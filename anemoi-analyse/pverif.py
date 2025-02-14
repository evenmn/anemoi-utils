import numpy as np
from yrlib_utils import get_station_metadata, get_available_timeseries, get

timeseries = get_available_timeseries(frost_client_id="92ecb11b-2a37-4b60-8f5f-74d7ea360d2a", variable="air_temperature")
print(timeseries)

metadata = get_station_metadata("92ecb11b-2a37-4b60-8f5f-74d7ea360d2a")


lats = []; lons = []
for key, dct in metadata.items():
    lats.append(dct['lat'])
    lons.append(dct['lon'])
    print(dct)

print(len(lats))

start = np.datetime64('2020-01-01T00', 's')
end = np.datetime64('2020-01-02T00', 's')
obs_times, obs_locations, obs_values = get(start.astype(int), end.astype(int), "air_temperature", "92ecb11b-2a37-4b60-8f5f-74d7ea360d2a", station_ids=timeseries)
print(obs_locations[0])
print(len(obs_times))
print(len(obs_locations))
print(obs_values.shape)

"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ax = plt.axes(projection=ccrs.PlateCarree())

ax.stock_img()

ax.add_feature(cfeature.LAND) #If I comment this => all ok, but I need 
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.RIVERS)
ax.coastlines()

ax.scatter(lons, lats, transform=ccrs.PlateCarree())
plt.show()
"""
