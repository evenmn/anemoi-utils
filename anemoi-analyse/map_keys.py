
def kelvin2celsius(ds, slc):
    """Convert Kelvin to Celsius."""
    t_idx = ds.variables.index('2t')
    return ds[slc][:,t_idx,0] - 273.15

def wind_magnitude(ds, slc):
    """Convert wind in x- and y-dirs to
    wind magnitude"""
    u_idx = ds.variables.index('10u')
    v_idx = ds.variables.index('10v')
    u10 = ds[slc][:,u_idx,0]
    v10 = ds[slc][:,v_idx,0]
    w10 = (u10**2+v10**2)**0.5
    return w10

def precip_accu6(ds, slc):
    """Precip m to mm."""
    p_idx = ds.variables.index('tp')
    return ds[slc][:,p_idx,0] * 1000

def mslp_(ds, slc):
    """Get mslp, convert from Pa to hPa."""
    m_idx = ds.variables.index('msl')
    return ds[slc][:,m_idx,0] / 100

map_keys = {
    'air_temperature_2m': {
        'standard': 't2m',
        'era5': ['2t'],
        'frost': 'air_temperature',
        'units': 'C',
        'transform': kelvin2celsius,
        'thresholds': [-30, -20, -10, 0, 10, 20, 25, 30, 35, 40, 45, 50],
        'long_name': 'Air temperature 2m',
    },
    'wind_speed_10m': {
        'standard': 'ws10m',
        'era5': ['10u', '10v'],
        'frost': 'wind_speed',
        'units': 'm/s', 
        'transform': wind_magnitude,
        'thresholds': [10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.6],
        'long_name': 'Wind speed 10m',
    },
    'precipitation_amount_acc6h': {
        'standard': 'precip6h',
        'era5': ['tp'],
        'frost': 'sum(precipitation_amount PT6H)',
        'units': 'mm',
        'transform': precip_accu6,
        'thresholds': [0.5, 1, 5, 10, 20, 30, 40, 50],
        'long_name': 'Precipitation',
    },
    'air_pressure_at_sea_level': {
        'standard': 'mslp',
        'era5': ['msl'],
        'frost': 'air_pressure_at_sea_level',
        'units': 'hPa',
        'transform': mslp_,
        'thresholds': [970, 980, 990, 1000, 1010, 1020, 1030],
        'long_name': 'Mean sea level pressure',
    },
}
