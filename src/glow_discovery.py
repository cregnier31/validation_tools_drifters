"""Discover actual GLOW YYYYMMDD.zarr forecast cycles on CloudFerro S3."""

import re
from datetime import datetime

import s3fs


def discover_glow_cycles(config, first, last):
    """Return sorted available GLOW forecast initialization datetimes, inclusive."""
    model = config["model_params"]["GLOW_FCST"]
    bucket = model["bucket_name"]
    prefix = model["fname_prefix"].strip("/")
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": model["endpoint_url"]},
    )
    directory = f"{bucket}/{prefix}"
    dates = set()
    for item in fs.ls(directory, detail=False):
        name = str(item).rstrip("/").rsplit("/", 1)[-1]
        match = re.fullmatch(r"(\d{8})\.zarr", name)
        if match is None:
            continue
        try:
            day = datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue
        if first <= day <= last:
            dates.add(day)
    return sorted(dates)
