#!/usr/bin/env python3

# converter for uncrewed aircraft system (UAS) profile data in whitespace-delimited ASCII
#
# Expected input: one header line followed by data rows, e.g.
#   time  lon(rad)  lat(rad)  pressure(Pa)  height(m)  RH  SH  TD(K)  Moist_var  T(K)  T_var  U(m/s)  U_var  V(m/s)  V_var
#   2024-09-11 21:05:21  4.607914  0.631079  98765.47  225.63  0.398959  0.010708  288.482854  1.00e+00  303.681627  1.00 ...
# where the time column holds a date and a time separated by a space, lon/lat are in radians,
# and the *_var columns are observation error variances (Moist_var applies to TD only).  Values of -999.9 denote missing data.

import sys
import os
import time
from datetime import datetime
import numpy as np
import logging

# These modules need the path to lib-python modules
import pyiodaconv.ioda_conv_engines as iconv
from pyiodaconv.orddicts import DefaultOrderedDict

os.environ["TZ"] = "UTC"

# IODA variable name: [input column, input error variance column or constant error, units]
# RH and SH have no error variance in the input, so they get constant errors:
# 3% (0.03) for RH and 1.0e-3 kg/kg for SH
varDict = {'airTemperature': ['T(K)', 'T_var', 'K'],
           'relativeHumidity': ['RH', 0.03, '1'],
           'specificHumidity': ['SH', 1.0e-3, 'kg kg-1'],
           'dewPointTemperature': ['TD(K)', 'Moist_var', 'K'],
           'windEastward': ['U(m/s)', 'U_var', 'm s-1'],
           'windNorthward': ['V(m/s)', 'V_var', 'm s-1']}

locationKeyList = [("latitude", "float", "degrees_north"),
                   ("longitude", "float", "degrees_east"),
                   ("dateTime", "long", "seconds since 1970-01-01T00:00:00Z"),
                   ("pressure", "float", "Pa"),
                   ("height", "float", "m")]

meta_keys = [m_item[0] for m_item in locationKeyList]

# input columns holding the MetaData (other than dateTime)
meta_columns = {'longitude': 'lon(rad)',
                'latitude': 'lat(rad)',
                'pressure': 'pressure(Pa)',
                'height': 'height(m)'}

iso8601_string = locationKeyList[meta_keys.index('dateTime')][2]
epoch = datetime.fromisoformat(iso8601_string[14:-1])

metaDataName = iconv.MetaDataName()
obsValName = iconv.OvalName()
obsErrName = iconv.OerrName()
qcName = iconv.OqcName()

# netCDF default fill values (same as netCDF4.default_fillvals), defined here so netCDF4 is not required
float_missing_value = np.float32(9.969209968386869e+36)
int_missing_value = np.int32(-2147483647)
double_missing_value = np.float64(9.969209968386869e+36)
long_missing_value = np.int64(-9223372036854775806)
string_missing_value = '_'

missing_vals = {'string': string_missing_value,
                'integer': int_missing_value,
                'long': long_missing_value,
                'float': float_missing_value,
                'double': double_missing_value}
dtypes = {'string': object,
          'integer': np.int32,
          'long': np.int64,
          'float': np.float32,
          'double': np.float64}

# values at or below this threshold in the input are treated as missing
input_missing_threshold = -999.0


def main(args):

    file_names = args.input
    output_file = args.output

    start_time = time.time()

    data = {}
    for key in meta_keys:
        data[key] = np.array([])
    for key in varDict.keys():
        data[key] = np.array([])
        data[key + '_err'] = np.array([])

    for fname in file_names:
        logging.info(f"Reading file:  {fname}")
        data = read_file(fname, data)

    nlocs = len(data['dateTime'])
    if nlocs == 0:
        logging.error("No data to write, stopping execution.")
        sys.exit(1)
    logging.info(f" found a total of {nlocs} observations")

    GlobalAttrs = {
        'converter': os.path.basename(__file__),
        'ioda_version': 2,
        'description': 'Uncrewed aircraft system (UAS) profile observations',
        'source': 'UAS ASCII file',
        'sourceFiles': ", ".join(file_names)
    }
    if args.date:
        dtg = datetime.strptime(args.date, '%Y%m%d%H')
        GlobalAttrs['datetimeReference'] = dtg.isoformat() + "Z"

    DimDict = {'Location': nlocs}

    varDims = {}
    for key in varDict.keys():
        varDims[key] = ['Location']

    varAttrs = DefaultOrderedDict(lambda: DefaultOrderedDict(dict))

    # Set units of the MetaData variables and all _FillValues.
    for key in meta_keys:
        dtypestr = locationKeyList[meta_keys.index(key)][1]
        if locationKeyList[meta_keys.index(key)][2]:
            varAttrs[(key, metaDataName)]['units'] = locationKeyList[meta_keys.index(key)][2]
        varAttrs[(key, metaDataName)]['_FillValue'] = missing_vals[dtypestr]

    # Set units and FillValue attributes for groups associated with observed variable.
    for key in varDict.keys():
        units = varDict[key][2]
        varAttrs[(key, obsValName)]['units'] = units
        varAttrs[(key, obsErrName)]['units'] = units
        varAttrs[(key, obsValName)]['coordinates'] = 'longitude latitude'
        varAttrs[(key, obsErrName)]['coordinates'] = 'longitude latitude'
        varAttrs[(key, qcName)]['coordinates'] = 'longitude latitude'
        varAttrs[(key, obsValName)]['_FillValue'] = float_missing_value
        varAttrs[(key, obsErrName)]['_FillValue'] = float_missing_value
        varAttrs[(key, qcName)]['_FillValue'] = int_missing_value

    # Fill the final IODA data:  MetaData then ObsValues, ObsErrors, and QC
    ioda_data = {}
    for key in meta_keys:
        dtypestr = locationKeyList[meta_keys.index(key)][1]
        ioda_data[(key, metaDataName)] = np.array(data[key], dtype=dtypes[dtypestr])

    for key in varDict.keys():
        logging.info(f" the variable: {key} will be placed into ioda_data")
        obs = np.array(data[key], dtype=np.float32)
        ioda_data[(key, obsValName)] = obs
        ioda_data[(key, obsErrName)] = np.array(data[key + '_err'], dtype=np.float32)
        ioda_data[(key, qcName)] = np.where(obs == float_missing_value, int_missing_value, 2).astype(np.int32)

    logging.debug("Writing file: " + output_file)

    # setup the IODA writer and write everything out.
    writer = iconv.IodaWriter(output_file, locationKeyList, DimDict)
    writer.BuildIoda(ioda_data, varDims, varAttrs, GlobalAttrs)

    logging.info("--- {:9.4f} total seconds ---".format(time.time() - start_time))


def to_missing(values):
    # replace input missing values (-999.9) with the IODA float fill value
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= input_missing_threshold, float_missing_value, values)


def read_file(file_name, data):

    with open(file_name, 'r') as f:
        header = f.readline().split()
        if not header or header[0] != 'time':
            logging.error(f"{file_name} does not start with the expected header line, skipping")
            return data

        # the time column spans two whitespace-separated tokens (date and time)
        columns = ['date', 'time'] + header[1:]
        ncols = len(columns)

        dates = []
        rows = []
        for lineno, line in enumerate(f, start=2):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != ncols:
                logging.warning(f"{file_name} line {lineno}: expected {ncols} fields, found {len(tokens)}, skipping")
                continue
            try:
                dtg = datetime.strptime(f"{tokens[0]} {tokens[1]}", '%Y-%m-%d %H:%M:%S')
                values = [float(t) for t in tokens[2:]]
            except ValueError as e:
                logging.warning(f"{file_name} line {lineno}: {e}, skipping")
                continue
            dates.append(np.int64(round((dtg - epoch).total_seconds())))
            rows.append(values)

    if not rows:
        logging.warning(f"{file_name} contains no valid observations")
        return data

    table = np.array(rows, dtype=np.float64)
    col = {name: table[:, i] for i, name in enumerate(columns[2:])}

    missing_cols = [c for c in list(meta_columns.values()) + [v[0] for v in varDict.values()] if c not in col]
    if missing_cols:
        logging.error(f"{file_name} is missing required columns: {', '.join(missing_cols)}")
        sys.exit(1)

    data['dateTime'] = np.append(data['dateTime'], np.array(dates, dtype=np.int64))

    # convert radians to degrees, with longitude in [-180, 180)
    lat = np.degrees(col[meta_columns['latitude']])
    lon = np.degrees(col[meta_columns['longitude']])
    lon = (lon + 180.0) % 360.0 - 180.0
    data['latitude'] = np.append(data['latitude'], lat)
    data['longitude'] = np.append(data['longitude'], lon)
    data['pressure'] = np.append(data['pressure'], to_missing(col[meta_columns['pressure']]))
    data['height'] = np.append(data['height'], to_missing(col[meta_columns['height']]))

    for key, (obs_col, var_col, _) in varDict.items():
        obs = to_missing(col[obs_col])
        if var_col is None:
            err = np.full(obs.shape, float_missing_value)
        elif isinstance(var_col, float):
            err = np.where(obs != float_missing_value, var_col, float_missing_value)
        elif var_col in col:
            variance = col[var_col]
            err = np.where((variance > 0) & (obs != float_missing_value), np.sqrt(np.abs(variance)), float_missing_value)
        else:
            logging.warning(f"{file_name} has no {var_col} column, {key} errors set to missing")
            err = np.full(obs.shape, float_missing_value)
        data[key] = np.append(data[key], obs)
        data[key + '_err'] = np.append(data[key + '_err'], err)

    return data


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=('Read an uncrewed aircraft system (UAS) ASCII profile file'
                     ' and convert into IODA output file')
    )

    required = parser.add_argument_group(title='required arguments')
    required.add_argument('-i', '--input', nargs='+',
                          action='store', default=None, required=True,
                          help='input files')
    required.add_argument('-o', '--output',
                          action='store', default=None, required=True,
                          help='output file')

    parser.set_defaults(debug=False)
    parser.set_defaults(verbose=False)

    optional = parser.add_argument_group(title='optional arguments')
    optional.add_argument('-d', '--date',
                          action='store', default=None,
                          help='date reference string (YYYYMMDDHH)')
    optional.add_argument('--debug', action='store_true',
                          help='enable debug messages')
    optional.add_argument('--verbose', action='store_true',
                          help='enable verbose debug messages')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    for file_name in args.input:
        if not os.path.isfile(file_name):
            parser.error(f'Input (-i option) file: {file_name} does not exist')

    main(args)
