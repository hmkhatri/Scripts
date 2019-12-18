# The script uses 1/48 deg MITgcm model run to construct smaller files for regional data

# Import Modules
import xarray as xr
import xrft
import dask.array as da
from dask.distributed import Client, LocalCluster
from xgcm import Grid
import numpy as np
from xmitgcm import llcreader
import bottleneck
import intake

# Load grid data from web
cat = intake.Catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/llc4320.yaml")
dsgrid = cat["LLC4320_grid"].to_dask()
dsgrid = llcreader.faces_dataset_to_latlon(dsgrid, metric_vector_pairs=[('dxC', 'dyC'), ('dyG', 'dxG')])

model = llcreader.ECCOPortalLLC4320Model()
#ds = model.get_dataset(varnames=['U','V'], k_levels=[1,3,5,10,30],type='latlon')
ds = model.get_dataset(varnames=['U','V'], k_levels=[1],type='latlon') #,k_chunksize=5)

ds = xr.merge([ds, dsgrid])
ds = ds.isel(time = np.arange(0, 1)) #len(ds.time), 12))

grid = Grid(ds, coords={'X': {'center': 'i', 'right': 'i_g'}, 'Y': {'center': 'j', 'right': 'j_g'}})

#ds1 = model.get_dataset(varnames=['W'], type='latlon')
#grid1 = Grid(ds1, coords={'X': {'center': 'i', 'right': 'i_g'}, 'Y': {'center': 'j', 'right': 'j_g'}, 'Z': {'center': 'k', 'outer': 'k_l'}})
#ds1 = xr.merge([ds1, dsgrid])

# Compute vorticity and divergence 
vorticity = ( - grid.diff(ds.U * ds.dxC, 'Y', boundary='fill') 
              + grid.diff(ds.V * ds.dyC, 'X', boundary='fill') ) / ds.rAz

divergence =  ( grid.diff(ds.U * ds.dxC, 'X', boundary='fill') 
              + grid.diff(ds.V * ds.dyC, 'Y', boundary='fill') ) / ds.rA

# Select data for a small region
lat1, lat2 = (29., 41.); # 10x10 deg box near gulf stream
lon1, lon2 = (-121.,-109.); # 1 deg should be avoided on each side for the final data

u1 = grid.interp(ds.U, 'X', boundary='fill')
v1 = grid.interp(ds.V, 'Y', boundary='fill')
vort = grid.interp(grid.interp(vorticity, 'X', boundary='fill'), 'Y', boundary='fill')
#w1 = grid1.interp(ds1.W, 'Z', boundary='fill')
#w1 = w1.sel({k = [1,3,5,10,30]})
#eta = ds.Eta

#eta = eta.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)
u1 = u1.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)
v1 = v1.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)
vort1 = vort.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)
div1 = divergence.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)
#w1 = w1.where((ds.YC > lat1) & (ds.YC < lat2) & (ds.XC > lon1) & (ds.XC < lon2), drop = True)