# MRSimTracks Analysis

This repository contains custom analysis scripts for running particle advection on Patient Aneurysm CFD data and Stenotic Phantom point-cloud datasets using the `MRSimTracks` engine.

## Execution Requirements

Because these scripts rely on the `MRSimTracks` physics engine, you must use the Python environment where `MRSimTracks` was installed to run them.

For example, if `MRSimTracks` is located at `..\MRSimTracks`, you should run the scripts using its virtual environment like so:

```powershell
# From this MRSim_Analysis folder:
..\MRSimTracks\.venv\Scripts\python.exe run_phantom_tracking.py
..\MRSimTracks\.venv\Scripts\python.exe visualize_phantom.py
```

## Data Privacy

**WARNING**: Medical meshes (`.vtu`), trajectory files (`.h5`), and point cloud datasets (`.npy`, `.pkl`) are extremely large and contain sensitive/private geometry data. 

A strict `.gitignore` has been provided to prevent these files from being committed to GitHub. Do not override it.
