import ee
import xarray

# testing xarray

ic = ee.ImageCollection('NASA/VIIRS/002/VNP09GA').filterDate(start_date, end_date)

ds = xarray.open_dataset(
    ic,
    engine='ee',
    projection=ic.first().select(0).projection(),
    geometry=roi
)

print('test')