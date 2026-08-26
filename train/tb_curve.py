"""Dump a scalar curve from a run's TensorBoard event file."""
import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "val/recon_loss"
ev = sorted(glob.glob(run + "/tb/version_*/events*"))
acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]
if tag not in tags:
    print("not found. available:", sorted(tags))
    sys.exit(1)
s = acc.Scalars(tag)
print("%s  n=%d" % (tag, len(s)))
prev = None
for e in s:
    d = "" if prev is None else "  d=%+.5f" % (e.value - prev)
    print("step %7d  %.5f%s" % (e.step, e.value, d))
    prev = e.value
