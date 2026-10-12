The BZ reaction is a chemical oscillator. Two objectives are optimised jointly:
  - reward_oscillation : high = shorter mean periods and lower period CV
  - reward_signal      : high = longer capped duration and higher mean maxima/amplitude
Chemical inputs [M]: conc_MA_M [0–0.8], conc_NaBrO3_M [0–0.7],
                     conc_H2SO4_M [0–1.5], conc_Ru_M [0.35e-3].
Reward formula (TIME_CAP=3400 s, z = z-score):
  period_cv = period_std / period_mean
  total_time_capped = min(total_time, TIME_CAP)
  reward_oscillation = (-22.6*z(period_mean) - 2.4*z(period_cv)) / 25.0
  reward_signal      = (100*z(total_time_capped) + 3.8*z(maxima_mean) + 2.4*z(amplitude_mean)) / 106.2
  Attempted experiments with n_osc == 0 receive -5 for both rewards.
  Unmeasured candidates retain NaN rewards; they are not failures.
In recompute mode, z-score statistics are refitted after each batch, so previous
rewards and hypervolume can change. Frozen normalization uses saved statistics.
Use the campaign's supplied improvement rule rather than inferring progress
from raw hypervolume changes alone.
Chemical prior guidance: NaBrO3/MA ratios near 0.8–1 may be favorable; high
H2SO4 may quench oscillations and low H2SO4 may produce irregular behavior.
These are qualitative priors, not feasibility constraints or measured outcomes.
Revise hypotheses using the supplied literature and campaign observations;
do not exclude other operating windows solely because they differ from this prior.
The configured candidate pool defines the actual admissible experiments.
