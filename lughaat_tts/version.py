"""Version constants.

FRONTEND_VERSION is written into config.json as "urdu_frontend_version" at export
time (plan rule 2.2). Bump it whenever the normalizer, lexicon, rules, neural G2P or
the English accent mapping changes in a way that alters phoneme output. Training and
inference phonemes must always come from the same FRONTEND_VERSION.
"""

__version__ = "1.0.0"
FRONTEND_VERSION = "ur-frontend-1.0.0"
