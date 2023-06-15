import glob
import os
import json


def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_json(path):
    return json.loads(read_file(path))


def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, "r") as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [
        to_jsonl(elem)
        for elem in data
    ]
    write_file("\n".join(lines), path)


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def show_json(obj, do_print=True):
    string = json.dumps(obj, indent=2)
    if do_print:
        print(string)
    else:
        return string


def create_containing_folder(path):
    fol_path = os.path.split(path)[0]
    os.makedirs(fol_path, exist_ok=True)
    return path


def fsspec_torch_save(obj, path):
    import fsspec
    import torch
    with fsspec.open(path, "wb") as f:
        torch.save(obj, f)


def fsspec_torch_load(path, map_location=None):
    import fsspec
    import torch
    with fsspec.open(path, mode="rb") as f:
        return torch.load(f, map_location=map_location)


def fsspec_write_file(data, path, mode="w"):
    import fsspec
    with fsspec.open(path, mode=mode) as f:
        f.write(data)


def fsspec_exists(path):
    import fsspec
    if path.startswith("s3"):
        fs = fsspec.filesystem('s3')
        return fs.exists(path)
    else:
        return os.path.exists(path)


def fsspec_isfile(path):
    import fsspec
    if path.startswith("s3"):
        fs = fsspec.filesystem('s3')
        return fs.isfile(path)
    else:
        return os.path.isfile(path)


def fsspec_isdir(path):
    import fsspec
    if path.startswith("s3"):
        fs = fsspec.filesystem('s3')
        return fs.isdir(path)
    else:
        return os.path.isdir(path)


def fsspec_listdir(path):
    import fsspec
    if path.startswith("s3"):
        fs = fsspec.filesystem('s3')
        return [os.path.split(x["Key"])[-1] for x in fs.listdir(path)]
    else:
        return os.listdir(path)


def fsspec_glob(pattern):
    print(f"fsspec_glob: {pattern}")
    import fsspec
    if pattern.startswith("s3"):
        fs = fsspec.filesystem('s3')
        return [
            f"s3://{path}"
            for path in fs.glob(pattern)
        ]
    else:
        return glob.glob(pattern)

