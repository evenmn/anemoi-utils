import argparse
import calendar
import datetime
import gridpp
import netCDF4
import numbers
import numpy as np
import os
import re
import requests
import scipy.interpolate
import sys
import time
import tqdm
import xarray as xr

""" Various helper functions copied from yrlib"""

def get(
    start_time,
    end_time,
    variable,
    frost_client_id,
    time_resolutions=None,
    wmo=None,
    station_holders=None,
    remove_station_holders=None,
    latrange=None,
    lonrange=None,
    level="default",
    station_ids=None,
    country=None,
    oda=False,
    host=None,
    debug=False,
    timeout=60,
    interpolation_window=0,
):
    """Retrieves data from frost.met.no"""

    """Strategy for dealing with 6H precip accumulation:
        - Ask for 1H precipitation
        - sum them up
        - take into account that we need hours before the start time
    """
    if variable == "sum(precipitation_amount PT6H)":
        times = np.arange(start_time - 6 * 3600, end_time + 1, 3600)
    else:
        times = np.arange(start_time, end_time + 1, 3600)

    # Variables, such as 24H precip, use time-offset. This means the data must be looked up for the
    # incorrect time, and then corrected with time-offset to the correct time.
    # For example, 24h precip for 20220601T06 is stored on 20220601T00 and has time offset PT6H.
    if variable in ["sum(precipitation_amount P1D)", "max(air_temperature P1D)", "min(air_temperature P1D)"]:
        lookup_times = [curr_time // 86400 * 86400 for curr_time in times]
    else:
        lookup_times = times

    lookup_variable = variable
    if variable == "sum(precipitation_amount PT6H)":
        lookup_variable = "sum(precipitation_amount PT1H)"

    num_times = len(times)
    # TODO: This is needed for METAR
    num_lookup_times = len(lookup_times) * 2

    # TODO: Pass a time into get_station_metadata? Problematic if the time is in the future, because
    # then an empty list is returned
    # INFO 2024/01/05 this is not anymore the case: a request for a future time gives metadata that
    # are almost similar to those with now time
    station_info = get_station_metadata(
        frost_client_id,
        lookup_variable,
        station_holders=station_holders,
        remove_station_holders=remove_station_holders,
        country=country,
    )

    # Compute time range
    start = unixtime_to_reference_time(np.min(lookup_times))
    end = unixtime_to_reference_time(np.max(lookup_times) + 3600)

    ids = list(station_info.keys())
    if station_ids is not None:
        ids = station_ids
    if latrange is not None and lonrange is not None:
        ids = [
            id
            for id in station_info.keys()
            if (station_info[id]["lat"] >= latrange[0])
            & (station_info[id]["lat"] <= latrange[1])
            & (station_info[id]["lon"] >= lonrange[0])
            & (station_info[id]["lon"] <= lonrange[1])
        ]
    if wmo is not None:
        ids = [id for id in ids if station_info[id]["wmo"] == wmo]
    values = dict()

    if host is None:
        if oda:
            host = "frost-prod.met.no"
        else:
            host = "frost.met.no"
    if oda:
        # endpoint = "https://%s/api/v1/obs/met.no/kvkafka/get?" % host
        endpoint = "https://%s/api/v1/obs/met.no/filter/get?" % host
    else:
        endpoint = "https://%s/observations/v0.jsonld" % host

    # Calculate how many stations to read at one time
    # Shouldn't try more than 50 because this will exceed the URL character limit
    # Also shouldn't get more than 100000 datapoints
    max_frost_obs = 100000
    total_num_obs = len(ids) * num_lookup_times

    # TODO: Might be too much data still, because there could be multiple levels, or other
    # timeseries
    stations_per_request = min(50, max_frost_obs // num_lookup_times)
    request_time = 0
    num_requests = 0
    if debug:
        print("Fething %d observations" % (num_lookup_times * len(ids)))
        print("Fetching %d times" % num_lookup_times)
        print("Fetching %d stations, %d at a time" % (len(ids), stations_per_request))
        print(stations_per_request * num_lookup_times)
    it_ids = len(ids)

    units = None

    while it_ids > 0:
        if it_ids > stations_per_request:
            # get last 50
            sub_idList = ids[it_ids - stations_per_request : it_ids]
            it_ids = it_ids - stations_per_request
        else:
            # get the rest if <50
            sub_idList = ids[:it_ids]
            it_ids = 0

        parameters = get_request_parameters(
            sub_idList, lookup_variable, start, end, level, time_resolutions, oda
        )
        s_time = time.time()
        r = requests.get(
            endpoint,
            parameters,
            auth=(frost_client_id, ""),
            timeout=timeout,
        )
        e_time = time.time()
        if debug:
            print(parameters)
        num_requests += 1
        request_time += e_time - s_time

        if r.status_code == 200:
            if oda:
                temp, units = parse_oda(r.json())
            else:
                temp, units = parse_kdvh(r.json())
            for id in temp:
                values[id] = temp[id]
        elif r.status_code == 412:
            print("STATUS: No valid data was found for the list of query Ids.")
        elif r.status_code == 403:
            print("STATUS: Too much data was retrived.")
            print(parameters)
        else:
            print("STATUS: Unknown frost error: %d: %s" % (r.status_code, r.text))

    locations = list()
    ids = [id for id in ids if id in station_info]
    if units == "text":
        dtype = "object"
    else:
        dtype = np.float32
    out_values = np.nan * np.zeros([num_times, len(ids)], dtype)
    for i in range(len(ids)):
        id = ids[i]
        lat = station_info[id]["lat"]
        lon = station_info[id]["lon"]
        elev = station_info[id]["elev"]
        id_num = int(id.replace("SN", ""))
        locations += [Location(lat, lon, elev, id_num)]
        if id in values:
            x = np.array([v[0] for v in values[id]])
            y = np.array([v[1] for v in values[id]])
            interp = scipy.interpolate.interp1d(x, y, "nearest", fill_value="extrapolate")
            out_values[:, i] = interp(times)
        else:
            pass
            # print("No observations for %s" % id)

    # Remove missing stations
    if units == "text":
        # For text data, we cannot run np.isnan
        # So we have to determine it by iterating over all values
        flat = out_values.flatten()
        mask = np.zeros(len(flat), np.float32)
        for i in range(len(flat)):
            mask[i] = isinstance(flat[i], str)
        mask = np.reshape(mask, out_values.shape)
        I = np.where(np.sum(mask, axis=0) > 0)[0]
    else:
        I = np.where(np.sum(np.isnan(out_values) == 0, axis=0) > 0)[0]
    out_values = out_values[:, I]
    locations = [locations[i] for i in I]
    if debug:
        if dtype != "object":
            print("Frost valid data: %d" % (np.sum(np.isnan(out_values) == 0)))
        print("Frost request time: %.2f s" % (request_time))
        print("Frost requests: %d" % (num_requests))

    # Compute 6h accumulations
    if variable == "sum(precipitation_amount PT6H)":
        assert lookup_variable == "sum(precipitation_amount PT1H)"
        T, S = out_values.shape
        new_out_values = np.nan * np.zeros([T - 5, S], np.float32)
        for s in range(S):
            new_out_values[:, s] = np.convolve(out_values[:, s], np.ones([6], dtype="int"), "valid")
        out_values = new_out_values
        times = times[5:]
        assert out_values.shape[0] == len(times)

    return times, locations, out_values


def get_station_metadata(
    frost_client_id,
    variable=None,
    unixtime=None,
    wmo=None,
    station_holders=None,
    remove_station_holders=None,
    country=None,
    debug=False,
):
    # Probably not worth setting a custom timeout for metadata requests, therefore
    # don't pass one into this and other functions below.
    station_holders = to_list(station_holders)
    remove_station_holders = to_list(remove_station_holders)

    station_dict = dict()
    parameters = {
        "types": "SensorSystem",
        "fields": "id,name,shortname,geometry,masl,wmoid,stationholders",
    }
    if country is not None:
        parameters["country"] = country
    if unixtime is not None:
        parameters["validtime"] = unixtime_to_reference_day(unixtime)
    if variable is not None:
        parameters["elements"] = variable

    r = requests.get(
        "https://frost.met.no/sources/v0.jsonld",
        parameters,
        auth=(frost_client_id, ""),
        timeout=30,
    )
    ids = list()

    if r.status_code == 200:
        data = r.json()["data"]
        count_discard = 0
        for i in range(len(data)):
            # print(data[i])
            id = data[i]["id"]
            id = id.split(":")[0]  # .replace('SN', '')
            if "masl" in data[i]:
                elev = data[i]["masl"]
            else:
                elev = -999  # missing value

            # filter data for WMO and non WMO
            keep_this_id = True
            is_wmo = "wmoId" in data[i]
            if wmo is not None and is_wmo != wmo:
                keep_this_id = False

            # filter out stations with incomplete data
            if "geometry" not in data[i]:
                keep_this_id = False
                # print('throwing out this id (no geometry): ' + id)

            # filters for station holders
            if "stationHolders" not in data[i]:
                keep_this_id = False
                # print('throwing out this id (no stationHolders): ' + id)

            if station_holders is not None:
                if not (any(x in data[i]["stationHolders"] for x in station_holders)):
                    keep_this_id = False
            if remove_station_holders is not None:
                if any(x in data[i]["stationHolders"] for x in remove_station_holders):
                    keep_this_id = False

            """
            # select station providers
            elif args.providers is not None:
                providers = args.providers.split(',')
                station_holders = data[i]['stationHolders']
                if not(any(x in station_holders for x in providers)):
                    keep_this_id = False
                    #print('throwing out this id (station holder): ' + str(station_holders))
            # or exclude certain station providers
            elif args.xproviders is not None:
                xproviders = args.xproviders.split(',')
                station_holders = data[i]['stationHolders']
                if any(x in station_holders for x in xproviders):
                    keep_this_id = False
                    #print('throwing out this id (exclude station holder): ' + str(station_holders))
            """

            # print('Keep this ID: ' + str(id) + ' bool: ' + str(keep_this_id))
            if keep_this_id:  # write into dict
                ids.append(id)
                station_dict[id] = {
                    "lat": data[i]["geometry"]["coordinates"][1],
                    "lon": data[i]["geometry"]["coordinates"][0],
                    "elev": elev,
                    "station_holder": data[i]["stationHolders"],
                    "wmo": is_wmo,
                }
                for key in parameters["fields"].split(','):
                    if key not in ["geometry"] and key in data[i]:
                            station_dict[id][key] = data[i][key]
            else:
                count_discard = count_discard + 1
        if debug:
            print("Number of stations: " + str(len(ids)))
            print("Number of stations discarded: " + str(count_discard))
            # print(station_dict)
    elif r.status_code == 404:
        print(f"STATUS: No data was found for the list of query Ids. {r.text}")
    else:
        raise Exception(
            f"Could not get station metadata. Frost error {r.status_code}: {r.text}"
        )
    return station_dict


def get_available_timeseries(frost_client_id, variable):
    parameters = {"fields": "sourceId", "elements": variable}

    r = requests.get(
        "https://frost.met.no/observations/availableTimeSeries/v0.jsonld",
        parameters,
        auth=(frost_client_id, ""),
        timeout=30,
    )
    ids = list()

    if r.status_code == 200:
        data = r.json()["data"]
        return [data[i]["sourceId"].split(":")[0] for i in range(len(data))]
    else:
        raise Exception(
            f"Could not get available timeseries. Frost error {r.status_code}: {r.text}"
        )


def unixtime_to_reference_time(unixtime):
    if unixtime == "now":
        return unixtime

    date, hour = unixtime_to_date(unixtime)
    minutes = unixtime // 60 % 60
    seconds = unixtime % 60
    return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
        date // 10000,
        date // 100 % 100,
        date % 100,
        hour,
        minutes,
        seconds,
    )

def unixtime_to_reference_day(unixtime):
    if unixtime == "now":
        return unixtime

    date, _ = unixtime_to_date(unixtime)
    return "%04d-%02d-%02d" % (
        date // 10000,
        date // 100 % 100,
        date % 100,
    )


def get_request_parameters(sources, variable, start, end, level, time_resolution, oda):
    parameters = dict()
    if oda:
        parameters["stationids"] = ",".join([s.replace("SN", "") for s in sources])
        parameters["elementids"] = variable
        parameters["levels"] = 0
        parameters["sensors"] = 0
        parameters["time"] = "%s/%s" % (start, end)
        parameters["incobs"] = "true"
    else:
        parameters["sources"] = ",".join(sources)
        parameters["elements"] = variable
        parameters["referencetime"] = "%s/%s" % (start, end)
        parameters["levels"] = str(level)
        if time_resolution is not None:
            parameters["timeresolutions"] = time_resolution
    return parameters


def get_name(name):
    if name == "ta":
        return "air_temperature"
    elif name == "rr1":
        return "sum(precipitation_amount PT1H)"


def get_units(frost_client_id, variable):
    """Get the units for a particular frost element"""
    parameters = {"ids": variable, "fields": "unit"}

    r = requests.get(
        "https://frost.met.no/elements/v0.jsonld",
        parameters,
        auth=(frost_client_id, ""),
        timeout=30,
    )
    ids = list()

    if r.status_code == 200:
        data = r.json()["data"]
        if len(data) == 0:
            raise Exception("No available units")
        return data[0]["unit"]
    else:
        raise Exception(
            f"Could not get element units. Frost error {r.status_code}: {r.text}"
        )


def parse_kdvh(data):
    """Parse json output from frost (kdvh)

    Args:
        data (dict): JSON output from frost kdvh API

    Returns:
        values (dict): stationId -> list of (unixtime, value)
        units (str): Units of the data

    Note: Values will not contain any stations that do not have at least 1 data point
    Note: Values will not contain any NaNs
    """
    values = dict()
    units = None
    stations_missing_info = []

    data = data["data"]
    for i in range(len(data)):
        reference_time = data[i]["referenceTime"]
        date = int(reference_time[0:4] + reference_time[5:7] + reference_time[8:10])
        hour = int(reference_time[11:13])
        minute = int(reference_time[14:16])
        second = int(reference_time[17:19])
        # if minute == 0 and second == 0:
        sourceId = str(data[i]["sourceId"])
        id = sourceId.split(":")[0]  # .replace('SN', '')
        missing_info = False
        unixtimes = set()
        for o in data[i]["observations"]:
            if "timeOffset" in o:
                time_offset = o["timeOffset"]
                time_offset_second = get_time_length(time_offset)
            else:
                time_offset_second = 0
            unixtime = (
                date_to_unixtime(date) + hour * 3600 + minute * 60 + second + time_offset_second
            )
            if unixtime in unixtimes:
                # In some cases, we can have repeated values for one unixtime (e.g. when there are
                # multiple time resolutions (e.g. for temperature PT1H and PT10M).
                continue
            unixtimes.add(unixtime)

            if "elementId" in o:
                element = o["elementId"]
            else:
                missing_info = True

            if "value" in o:
                value = o["value"]
                if value == "":
                    value = np.nan
                else:
                    if "unit" in o.keys():
                        if o["unit"] == "text":
                            value = value.replace(' ', '')
                            # Some variables are string codes
                            # print(o["unit"])
                            # print(value, "is not a float")
                        else:
                            value = float(value)
                    else:
                        missing_info = True
            else:
                missing_info = True

            if not missing_info:
                # Handle special values in frost:
                # https://frost.met.no/dataclarifications.html
                if re.match("sum\(precipitation_amount.*\)", element):
                    if value == -1:
                        value = 0
                elif element in ["cloud_base_height", "cloud_area_fraction", "cloud_area_fraction1", "low_type_cloud_area_fraction"]:
                    if value == -3:
                        value = np.nan
                elif element in ["wind_from_direction"]:
                    if value == -3:
                        value = np.nan
                elif re.match(".*surface_snow_thickness.*", element):
                    if value == -1:
                        value = 0
                    elif value == -3:
                        value = np.nan

                if units is None and "unit" in o:
                    units = o["unit"]

                if not missing_info and (isinstance(value, str) or not np.isnan(value)):
                    if id not in values:
                        values[id] = list()
                    values[id] += [(unixtime, value)]
            else:
                if sourceId not in stations_missing_info:
                    stations_missing_info.append(sourceId)

    if stations_missing_info:
        print(f'WARNING in parse_kdvh, missing unit/elementId/value for {len(stations_missing_info)} stations, e.g.: {stations_missing_info[0]}')
    return values, units


def get_time_length(string):
    """Converts frost time length psecification (e.g. PT1H) to number of seconds"""
    if not isinstance(string, str):
        raise ValueError(f"input must be a string, not {type(string)}")

    if len(string) <= 1 or string[0] != "P":
        raise ValueError(f"string must be of the format P<token>")

    if string[1] == "T":
        # Hourly and more frequent periods
        m = re.search(r"PT([0-9]*)([A-Z]{1})", string)
        num = int(m.group(1))
        period = m.group(2)
        multiplier = get_time_unit_multiplier(period, True)
        return num * multiplier
    else:
        m = re.search(r"P([0-9]*)([A-Z]{1})", string)
        num = int(m.group(1))
        period = m.group(2)
        multiplier = get_time_unit_multiplier(period, False)
        return num * multiplier

def get_time_unit_multiplier(period, hourly):
    if period == "D":
        return 86400
    elif period == "H":
        return 3600
    elif period == "M":
        if hourly:
            # Minutes
            return 60
        else:
            # Months
            raise NotImplementedError()
    else:
        raise NotImplementedError()

def to_list(ar):
    """Ensure output is a list, converting a scaler to a list if necesssary"""
    if isinstance(ar, numbers.Number):
        return [ar]
    return ar


def unixtime_to_date(unixtime):
    """Convert unixtime to YYYYMMDD

    Arguments:
       unixtime (int): unixtime [s]

    Returns:
       int: date in YYYYMMDD
       int: hour in HH
    """
    if not isinstance(unixtime, numbers.Number):
        raise ValueError("unixtime must be a number")

    dt = datetime.datetime.utcfromtimestamp(int(unixtime))
    date = dt.year * 10000 + dt.month * 100 + dt.day
    hour = dt.hour
    return date, hour

def date_to_unixtime(date, hour=0, min=0, sec=0):
    """Convert YYYYMMDD(HHMMSS) to unixtime

    Arguments:
       date (int): YYYYMMDD
       hour (int): HH
        min (int): MM
        sec (int): SS

    Returns:
       int: unixtime [s]
    """
    if not isinstance(date, int):
        raise ValueError("Date must be an integer")
    if not isinstance(hour, numbers.Number):
        raise ValueError("Hour must be a number")
    if not isinstance(min, numbers.Number):
        raise ValueError("Min must be a number")
    if not isinstance(sec, numbers.Number):
        raise ValueError("Sec must be a number")
    if hour < 0 or hour >= 24:
        raise ValueError("Hour must be between 0 and 24")
    if min < 0 or min >= 60:
        raise ValueError("Minute must be between 0 and 60")
    if sec < 0 or sec >= 60:
        raise ValueError("Second must be between 0 and 60")

    year = date // 10000
    month = date // 100 % 100
    day = date % 100
    ut = calendar.timegm(datetime.datetime(year, month, day).timetuple())
    return ut + (hour * 3600) + (min * 60) + sec


class Location:
    """Class representing a single point on earth

    Use this when detailed metadata about a location is needed, for example in quality control. This
    class may expand on the matadata it can store in the future. The
    preferred way is to use gridpp.Points when performance is needed.

    """

    def __init__(self, lat, lon, elev, id=None, prid=None, name=None, laf=1):
        """Initialize a location object

        Args:
            lat (float): Latitude [degrees]. Must be between -90,90
            lon (float): Longitude [degrees]
            elev (float): Elevation [m]
            id (int): Station id
            prid (anything): Provider id (e.g. Netatmo, etc)
            name (str): Name of location
            laf (float): Land area fraction [1]
        """

        # TODO: Deal with lon being outside -360 and 360. Decide whether we should use -180,180 or
        # 0,360
        if np.isnan(lat) or np.isnan(lon):
            raise ValueError("Lat and lon must be valid numbers")
        if lat > 90 or lat < -90:
            raise ValueError("Lat must be between -90 and 90")

        self.lat = float(lat)
        self.lon = float(lon)
        self.elev = float(elev)
        self.prid = prid
        self.id = id
        self.laf = laf
        self.name = name

    def __eq__(self, other):
        # TODO: Check id
        if np.isnan(self.elev) and np.isnan(other.elev):
            return self.lat == other.lat and self.lon == other.lon
        return (
            self.lat == other.lat
            and self.lon == other.lon
            and self.elev == other.elev
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.lat, self.lon, self.elev))

def get_common_indices(x, y):
    Ix = list()
    Iy = list()
    for i in range(len(x)):
        if x[i] in y:
            Ix += [i]
            if isinstance(y, list):
                Iy += [y.index(x[i])]
            else:
                Iy += [np.where(x[i] == y)[0][0]]
    return Ix, Iy

def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)
