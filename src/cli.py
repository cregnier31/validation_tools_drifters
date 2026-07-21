#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse
from datetime import datetime
import sys

def parse_args():
    """
    Parse and validate --date1 and --date2 command-line arguments.
    Returns:
        tuple: (date_start, date_end) as strings in YYYYMMDD format.
    """
    parser = argparse.ArgumentParser(description="Run stats plotting pipeline")
    parser.add_argument('--date1', type=str, help='Start date in YYYYMMDD format')
    parser.add_argument('--date2', type=str, help='End date in YYYYMMDD format')
    parser.add_argument('--lead', type=str, help='leadtime HCST or FCST')
    parser.add_argument('--sys', type=str, help='SYSTEM to validate')
    parser.add_argument('--scheme', type=str, default="K_smag", help='Scheme of the dispersion')
    parser.add_argument('--typeobs', type=str, default="COPERNICUS_DAILY", help='type of observation')
    parser.add_argument('--engine', type=str,  default="parcels", help='Engine choosen forthe dispersion')
    args = parser.parse_args()
    list_lead = ['HDCT', 'FCST']
    list_scheme = ['AdvectionRK4', 'K_smag', 'AdvectionEE']
    list_obs_type = ['COPERNICUS_DAILY', 'AOML']
    list_engine = ['parcels', 'opendrift']
    # --- Exit if missing arguments ---
    if not args.date1 or not args.date2 or not args.lead:
        print("Error: You must provide --date1 and --date2 in YYYYMMDD format and --lead (HCST or FCST)")
        print("Example: python your_script.py --date1 20221205 --date2 20250930")
        sys.exit(1)
    if not args.sys:
        print("Error: SYSTEM is missing")
        print("Example: python your_script.py --sys GLO12")
        sys.exit(1)
    if not args.scheme:
        print("Error: Scheme is missing")
        print("Example: python your_script.py --scheme K_smag")
        sys.exit(1)

    date_start = args.date1
    date_end = args.date2
    leadtime = args.lead
    system = args.sys
    scheme = args.scheme
    engine = args.engine

    if not args.typeobs:
        # Set typeobs by default
        obs_type =  "COPERNICUS_DAILY"
    else:
        obs_type = args.typeobs
        if obs_type not in list_obs_type:
            print(f"Error: Obs_type must be one of them {list_obs_type}")
            sys.exit(1)

    # --- Validate format ---
    if not (len(date_start) == 8 and date_start.isdigit() and len(date_end) == 8 and date_end.isdigit()):
        print("Error: Both --date1 and --date2 must be in YYYYMMDD format (8 digits).")
        sys.exit(1)

    # --- Validate order ---
    if date_end < date_start:
        print(f"Error: date2 ({date_end}) must be greater than date1 ({date_start}).")
        sys.exit(1)
    # --- Validate order ---
    if leadtime not in list_lead:
        print(f"Error: Lead must be one of them {list_lead}")
        sys.exit(1)
    if scheme not in list_scheme:
        print(f"Error: Scheme must be one of them {list_scheme}")
        sys.exit(1)

    date_start = datetime.strptime(date_start, "%Y%m%d")
    date_end = datetime.strptime(date_end, "%Y%m%d")

    return date_start, date_end, leadtime, system, scheme, obs_type, engine
