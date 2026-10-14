# Run notebook 04 headlessly and print the 5-minute cross-correlation used on the lead-lag slide.
import importlib.util, json, sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("nb04", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_, defs = mod.app.run()
ch = defs["changes"]
rows = []
for k in range(-6, 7):
    later = ch.groupby("key").d_kalshi.shift(-k)
    rows.append({"lag": 5 * k, "corr": round(float(ch.d_futures.corr(later)), 4)})
print(json.dumps({"n": len(ch), "sessions": int(ch.session.nunique()), "rows": rows}))
