import argparse
import pickle
import json
import base64
import numpy as _np

#!/usr/bin/env python3
"""
Simple .pkl -> .json converter.
Usage:
    python pkl_to_json.py input.pkl [-o output.json]
If the pickle contains multiple top-level objects they will be written as a list.
"""

def to_json_safe(obj):
        # primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
        # bytes -> try utf-8, else base64
        if isinstance(obj, (bytes, bytearray)):
                try:
                        return obj.decode("utf-8")
                except Exception:
                        return {"__bytes_base64__": base64.b64encode(bytes(obj)).decode("ascii")}
        # numpy types
        if _np is not None:
                if isinstance(obj, _np.ndarray):
                        return to_json_safe(obj.tolist())
                if isinstance(obj, (_np.generic,)):
                        return obj.item()
        # dict-like
        if isinstance(obj, dict):
                return {str(k): to_json_safe(v) for k, v in obj.items()}
        # iterable types
        if isinstance(obj, (list, tuple, set)):
                return [to_json_safe(v) for v in obj]
        # pandas DataFrame / Series support (if pandas is installed)
        try:
                import pandas as _pd  # noqa: F401
                if hasattr(obj, "to_dict"):
                        try:
                                return to_json_safe(obj.to_dict(orient="records") if getattr(obj, "ndim", None) == 2 else obj.to_dict())
                        except Exception:
                                pass
        except Exception:
                pass
        # generic objects with tolist / to_dict
        if hasattr(obj, "tolist"):
                try:
                        return to_json_safe(obj.tolist())
                except Exception:
                        pass
        if hasattr(obj, "to_dict"):
                try:
                        return to_json_safe(obj.to_dict())
                except Exception:
                        pass
        # fallback to string representation
        return str(obj)

def load_pickle_objects(path):
        objs = []
        with open(path, "rb") as f:
                try:
                        while True:
                                objs.append(pickle.load(f))
                except EOFError:
                        pass
        return objs

def main():
        p = argparse.ArgumentParser(description="Convert pickle (.pkl) to JSON.")
        p.add_argument("input", help="Input .pkl file")
        p.add_argument("-o", "--output", help="Output .json file (default: same name as input)", default=None)
        args = p.parse_args()

        objs = load_pickle_objects(args.input)
        data = objs[0] if len(objs) == 1 else objs

        jsonable = to_json_safe(data)
        out_path = args.output or args.input.rsplit(".", 1)[0] + ".json"
        with open(out_path, "w", encoding="utf-8") as f:
                json.dump(jsonable, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")

if __name__ == "__main__":
        main()