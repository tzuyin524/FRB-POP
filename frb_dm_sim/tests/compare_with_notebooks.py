"""Run an ORIGINAL notebook and the NEW package on identical synthetic inputs; compare the .npy outputs.
usage: python compare_with_notebooks.py <key> [<key> ...]      (keys as in telescopes.py)
"""
import json, os, re, sys, tempfile, shutil, importlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import astropy_shim; astropy_shim.install()
import matplotlib; matplotlib.use("Agg")
import make_synthetic_data

NB = {"askap_fe": "askap_fe", "askap_ics": "askap_ics", "pks": "pks", "fast": "fast", "meerkat_ib": "meerkat_ib",
      "meerkat_cb": "meerkat_cb", "dsa110": "dsa110", "apertif": "apertif"}
UP = os.environ.get("NB_DIR", "/mnt/user-data/uploads")

def run_old(key, data_dir, old_out):
    nb = json.load(open(f"{UP}/DM_comparison_mc_{NB[key]}.ipynb"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))
    src = src.replace("/home/thsu/FRB_population/DM_distribution", old_out).replace("/home/thsu/FRB_population", data_dir)
    src = src.replace("plt.show()", "plt.close('all')")
    g = {"__name__": "__old__"}
    exec(compile(src, f"old_{key}", "exec"), g)
    f = [x for x in os.listdir(old_out)]
    return np.load(os.path.join(old_out, f[0])), f[0]

def main(keys):
    tmp = tempfile.mkdtemp(); data = os.path.join(tmp, "data"); make_synthetic_data.make(data)
    os.environ["FRB_DATA_DIR"] = data
    import telescopes; importlib.reload(telescopes)
    from core import run_survey
    ok = True
    for key in keys:
        old_out = os.path.join(tmp, "old_" + key); os.makedirs(old_out)
        old, fname = run_old(key, data, old_out)
        res = run_survey(telescopes.get_telescope(key), verbose=False)
        new = res.dm_distribution
        same_shape = old.shape == new.shape
        close = same_shape and np.allclose(old, new, rtol=1e-9, atol=0)
        print(f"{key:11s} old {old.shape} new {new.shape}  identical-within-1e-9: {close}"
              + ("" if close else f"   max rel diff: {np.max(np.abs(old-new)/np.abs(old)) if same_shape else 'shape'}"))
        ok &= bool(close)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main(sys.argv[1:] or list(NB))
