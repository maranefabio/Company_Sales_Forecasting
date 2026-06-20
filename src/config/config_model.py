import numpy as np

# Cross-Validation configuration constants.

VALID_TARGETS: tuple[str, ...] = ('ASP', 'QT')   # Valid forcast targets
WARMUP_PERIOD = np.timedelta64(366, 'D')         # one full year of history before the first cutoff
END_BUFFER = np.timedelta64(3, 'M')              # reserve the last 3 months so a 90-day horizon always fits
CUTOFF_FREQ = '3MS'                              # generate a new cutoff every 3 months (quarterly)
CROSS_VALID_HORIZON = '90 days'                  # forecast horizon evaluated at each cutoff
