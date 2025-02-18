from goes2go import GOES
from goes2go.data import goes_nearesttime

#from datetime import datetime, timedelta
#import pandas as pd
#import boto3
#import botocore

#session = boto3.session.Session()
#s3_client = session.client(
#    'noaa-goes16.s3.amazonaws.com',
#    config=botocore.config.Config(signature_version=botocore.UNSIGNED, retries={'max_attempts': 3}),
#    verify=False  # Disable SSL verification (use with caution)
#)

G = GOES(satellite=16, product="ABI-L2-MCMIP", domain='C')

df = G.df(start='2022-07-04 01:00', end='2022-07-04 01:30')

#g = goes_nearesttime(datetime(2020, 12, 25, 10),
#                     satellite='goes16',
#                     product='ABI',
#                     return_as='xarray')

print('test')