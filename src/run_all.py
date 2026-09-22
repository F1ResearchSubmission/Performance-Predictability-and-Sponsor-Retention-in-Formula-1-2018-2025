"""Run the offline pipeline. Does not call the optional network collector."""
from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
SCRIPTS=['02_aggregate_performance.py','03_build_sponsor_panel.py','04_model.py',
         '05_robustness.py','06_figure1.py','07_few_cluster_inference.py',
         '08_validate_and_sensitivity.py']
if __name__=='__main__':
    for name in SCRIPTS:
        print('\nRunning '+name,flush=True)
        runpy.run_path(str(HERE/name),run_name='__main__')
