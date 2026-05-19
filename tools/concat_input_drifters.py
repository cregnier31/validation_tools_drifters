import pandas as pd
import glob
import os

# Directory containing CSV files
#input_dir = "/utmp/smer/regnierc/WORK/DRIFT_GLO12/project_fidji_new_test/fidji_parcels_NbmonthsDrift-12_20250402-20260405/ZONE_180W180E-90S90N/DRIFTERS"
input_dir = "/utmp/smer/regnierc/WORK/DRIFT_GLO12/project_fidji_new_test/fidji_parcels_NbmonthsDrift-12_20260101-20260211/ZONE_180W180E-90S90N/DRIFTERS"

# Find all CSV files
csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

print(f"Found {len(csv_files)} CSV files")

# Read and concatenate
df = pd.concat(
    (pd.read_csv(f) for f in csv_files),
    ignore_index=True
)

# Output file
dir_out = "/utmp/smer/regnierc/WORK/DRIFT_GLO12/project_fidji_new_test/"
output_file = os.path.join(input_dir, f"{dir_out}concatenated-drifters_20260101-20260201.csv")

# Save
df.to_csv(output_file, index=False)

print(f"Saved concatenated file to: {output_file}")
