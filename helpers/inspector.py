import numpy as np


def summarize_array(arr):
    return f"<np.ndarray shape={arr.shape}, dtype={arr.dtype}>"


def preview(value, maxlen=120):
    s = str(value)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def inspect_object(obj, name="object", depth=1, _indent=0):
    indent = "  " * _indent
    print(f"{indent}- {name}: {type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{indent}  Shape: {obj.shape}, Dtype: {obj.dtype}")
        print(f"{indent}  Preview: {preview(obj.flatten()[:10])}")
    elif isinstance(obj, dict):
        print(f"{indent}  Keys: {list(obj.keys())}")
        if depth > 0:
            for k, v in list(obj.items()):
                inspect_object(
                    v, name=f"{name}[{repr(k)}]", depth=depth - 1, _indent=_indent + 1
                )
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}  Length: {len(obj)}")
        if depth > 0:
            for i, item in enumerate(obj[:5]):  # only first 5 elements
                inspect_object(
                    item, name=f"{name}[{i}]", depth=depth - 1, _indent=_indent + 1
                )
    elif isinstance(obj, (int, float, str, bool)):
        print(f"{indent}  Value: {preview(obj)}")
    elif hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
        print(f"{indent}  Attributes:")
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(obj, attr)
                print(f"{indent}    - {attr}: {type(val)} {preview(val)}")
            except Exception as e:
                print(f"{indent}    - {attr}: <Error reading: {e}>")
    else:
        print(f"{indent}  Value: {preview(obj)}")
