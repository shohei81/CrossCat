"""Model- and sampler-wide constants for the CrossCat experiments."""

NUM_ROWS = 10
NUM_CONT_COLS = 1
NUM_CAT_COLS = 1
NUM_VIEWS = 2
NUM_CLUSTERS = 3
ALPHA_VIEW = 1.0
ALPHA_CLUSTER = 1.0
PRIOR_MEAN = 0.0
NUM_CATEGORIES = 3
ALPHA_CAT = 1.0

# Hyperpriors for SBP concentration parameters (Gamma(a, b))
ALPHA_VIEW_A = 1.0
ALPHA_VIEW_B = 1.0
ALPHA_CLUSTER_A = 1.0
ALPHA_CLUSTER_B = 1.0
ALPHA_CAT_A = 1.0
ALPHA_CAT_B = 1.0

# Hyperparameters for Normal-Gamma prior on continuous cluster parameters
NG_KAPPA0 = 1.0
NG_ALPHA0 = 1.0
NG_BETA0 = 1.0
NG_ALPHA0_A = 1.0
NG_ALPHA0_B = 1.0
NG_BETA0_A = 1.0
NG_BETA0_B = 1.0

# Hyperpriors for column-wise Normal-Gamma location hyperparameters
MU0_PRIOR_MEAN = 0.0
MU0_PRIOR_VAR = 10.0
KAPPA0_A = 1.0
KAPPA0_B = 1.0
