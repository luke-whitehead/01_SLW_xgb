# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:45:27 2023

@author: lwh41
"""

import sys
import os
import argparse
import logging
import struct
import datetime as dt

import numpy as np
from netCDF4 import Dataset

import matplotlib.pyplot as plt



TYPES = {
    'float32': 'f',
    'float64': 'd',
    'int16': 'h',
    'int32': 'i',
    'uint8': 'B',
    'uint16': 'H',
    'uint32': 'I',
}


HEADER_AFTERPULSE = [
    ['ap_header', 'uint32', 'afterpulse header'],
    ['ap_file_version', 'uint16', 'afterpulse file version'],
    ['ap_number_channels', 'uint8', 'afterpulse number of channels', None, 'count'],
    ['ap_number_bins', 'uint32', 'afterpulse number of bins', None, 'count'],
    ['ap_energy', 'float64', 'afterpulse energy', None, 'uJ'],
    ['ap_background_average_copol', 'float64', 'afterpulse co pol background average', 'count us-1'],
    ['ap_background_average_crosspol', 'float64', 'afterpulse cross pol background average', 'count us-1'],
]



def read_header(f, header_dsc):
    types = [x[1] for x in header_dsc]
    fmt = '<' + ''.join([TYPES[x[1]] for x in header_dsc])
    size = struct.calcsize(fmt)
    fields = [x[0] for x in header_dsc]
    buf = f.read(size)
    if len(buf) == 0:
        return None
    if len(buf) < size:
        raise IOError('Incomplete header')
    res = struct.unpack_from(fmt, buf)
    return {k: np.array(v, dtype=t) for k, v, t in zip(fields, res, types)}


def read_afterpulse(f):
    d = read_header(f, HEADER_AFTERPULSE)
    if d['ap_header'] != 0xAAEEEEAA:
        raise IOError('Invalid afterpulse header')
    n = int(d['ap_number_bins'])
    for x in ['ap_range', 'ap_copol', 'ap_crosspol']:
        buf = f.read(8*n)
        if len(buf) < 8*n:
            raise IOError('Incomplete %s data' % x)
        a = struct.unpack_from('<' + 'd'*n, buf)
        d[x] = np.array(a, np.float64)
    return d

def read_overlap(f):
    buf = f.read()
    n = len(buf)//16
    f.seek(0)
    d = {'ol_number_bins': np.array(n, np.uint32)}
    for x in ['ol_range', 'ol_overlap']:
        buf = f.read(8*n)
        if len(buf) < 8*n:
            raise IOError('Incomplete %s data' % x)
        a = struct.unpack_from('<' + 'd'*n, buf)
        d[x] = np.array(a, np.float64)
    return d

def read_dead_time(f):
    buf = f.read()
    n = len(buf)//4
    d = {'dt_number_coeff': np.array(n, np.uint32)}
    a = struct.unpack_from('<' + 'f'*n, buf)
    d['dt_coeff'] = np.array(a, np.float32)
    d['dt_coeff_degree'] = np.arange(n - 1, -1, -1, dtype=np.uint32)
    return d

def calc_dtcf(x, coeff):
    n = len(coeff)
    return np.sum([(x*1e3)**(n-i-1)*coeff[i] for i in range(n)], axis=0)

C = 299792458 

def calc_nrb(d, channel, name, name2):
    raw = d[channel]
    n, m = raw.shape
    background = d['background_average' + name2]
    ap = d.get('ap' + name, np.zeros(m, np.float64))
    energy = d['energy_monitor']*1e-3
    ap_energy = d.get('ap_energy', 1.)
    ap_background = d.get('ap_background_average' + name, 0.)
    ol_range = d.get('ol_range')
    ap_range = d.get('ap_range')
    overlap = d.get('ol_overlap', np.ones(m, np.float64))
    dt_coeff = d.get('dt_coeff', np.array([1.]))

    nrb = np.full((n, m), np.nan, np.float64)

    for i in range(n):
        range_ = 0.5*d['bin_time'][i]*C*(np.arange(m) + 0.5)*1e-3
        ap2 = np.interp(range_, ap_range, ap) if ap_range is not None else ap
        overlap2 = np.interp(range_, ol_range, overlap) if ol_range is not None else overlap
        nrb[i] = (raw[i,:]*calc_dtcf(raw[i,:], dt_coeff) - \
            background[i]*calc_dtcf(background[i], dt_coeff) - \
            ap2*calc_dtcf(ap2, dt_coeff)*energy[i]/ap_energy + \
            ap_background*calc_dtcf(ap_background, dt_coeff)*energy[i]/ap_energy)* \
            range_**2/(overlap2*energy[i])

    return nrb

cal_dir = 'S:/atmos_disk/Luke_W_480/MPL_cal/'
ap_file = cal_dir+'MMPL5054_Afterpulse_201903220500.bin'
ol_file = cal_dir+'MMPL5054_Overlap_201903270700.bin'
dt_file = cal_dir+'MMPL5054_SPCM34184_Deadtime7.bin'

with open(ap_file, 'rb') as f:
    afterpulse = read_afterpulse(f)

with open(ol_file, 'rb') as f:
    overlap = read_overlap(f)

with open(dt_file, 'rb') as f:
    dead_time = read_dead_time(f)
    
    
# inv_r2 = (overlap['ol_range']**2)**(-1)
    
# plt.plot(overlap['ol_range'][:200], overlap['ol_overlap'][:200], 'k-')
# plt.plot(overlap['ol_range'][:200], inv_r2[:200]/70, 'k-')
# plt.plot(overlap['ol_range'][:200], overlap['ol_overlap'][:200]*inv_r2[:200]/70, 'r')

# plt.show()
# plt.plot(overlap['ol_range'][:200], np.log10(inv_r2[:200]), 'k')
# plt.plot(overlap['ol_range'][:200], np.log10(overlap['ol_overlap'][:200]*inv_r2[:200]), 'r')

# plt.show()

# plt.plot(afterpulse['ap_range'][5:], afterpulse['ap_copol'][5:])