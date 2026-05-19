#!/usr/bin/env python
#-*- coding: utf-8 -*-
import argparse
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
    args = parser.parse_args()
    list_lead = ['HDCT', 'FCST']
    # --- Exit if missing arguments ---
    if not args.date1 or not args.date2 or not args.lead:
        print("❌ Error: You must provide --date1 and --date2 in YYYYMMDD format and --lead (HCST or FCST)")
        print("   Example: python your_script.py --date1 20221205 --date2 20250930")
        sys.exit(1)

    date_start = args.date1
    date_end = args.date2
    leadtime = args.lead 

    # --- Validate format ---
    if not (len(date_start) == 8 and date_start.isdigit() and len(date_end) == 8 and date_end.isdigit()):
        print("❌ Error: Both --date1 and --date2 must be in YYYYMMDD format (8 digits).")
        sys.exit(1)

    # --- Validate order ---
    if date_end <= date_start:
        print(f"❌ Error: date2 ({date_end}) must be greater than date1 ({date_start}).")
        sys.exit(1)
    # --- Validate order ---
    if leadtime not in list_lead:
        print(f"❌ Error: Lead must be one of them {list_lead}")

    return date_start, date_end, leadtime
