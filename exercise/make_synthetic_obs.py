"""Create SYNTHETIC 'observed' discharge for the exercise (no real gauge data in the repo).

obs = simulated discharge with noise, a 10% volume bias, one-day lag, plus planted
defects (gaps, a flat run, a spike) for check_discharge to find.
"""

from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent / "data"
sim = pd.read_csv(here / "wflow_output_sample.csv", index_col=0, parse_dates=True)["river_q_6136200"]

rng = np.random.default_rng(42)
obs = sim.shift(1).bfill() * 1.1 * rng.lognormal(0, 0.15, len(sim))
obs.iloc[100:110] = np.nan
obs.iloc[300:320] = obs.iloc[299]
obs.iloc[500] = obs.max() * 8
obs.rename("q_obs").to_frame().to_csv(here / "observed_synthetic.csv", index_label="time")
