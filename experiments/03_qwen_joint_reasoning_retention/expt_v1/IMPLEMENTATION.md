# JR1 implementation receipt

**Status:** INITIALIZED — development/readiness measurements are running; no
prospective training cell has started.

`jr1_campaign.py` supplies the campaign-local primitives and first readiness
path. The objective implementation uses shifted causal labels with prompt and
padding masking; the direct-preference objective takes sequence-summed policy
and frozen-reference log probabilities; the reasoning estimator is specified
as leave-one-out group-relative advantage with invalid partial groups rejected.

The initial deterministic test set covers answer parsing, exact-answer scoring,
prompt masking, preference-margin direction, and both valid and invalid
leave-one-out groups. The first real masked-supervision update is separately
recorded in the private calibration receipt and is development preparation, not
a counted cell. The harness refuses to manufacture replacement benchmark
labels and requires the state-root environment variable so private paths do
not enter source control.

The remaining admission work is intentionally still pending: public-data
provenance hashes, eight-sample reward informativeness, all-four-objective
one-device updates, controller/schedule replay, frozen matrix accounting, and
full-denominator evaluation. No results claim can be drawn from initialization.
