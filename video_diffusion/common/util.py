import os
import sys
import copy
import inspect
import datetime
from typing import List, Tuple, Optional, Dict


def glob_files(
    root_path: str,
    extensions: Tuple[str],
    recursive: bool = True,
    skip_hidden_directories: bool = True,
    max_directories: Optional[int] = None,
    max_files: Optional[int] = None,
    relative_path: bool = False,
) -> Tuple[List[str], bool, bool]:
    """glob files with specified extensions

    Args:
        root_path (str): _description_
        extensions (Tuple[str]): _description_
        recursive (bool, optional): _description_. Defaults to True.
        skip_hidden_directories (bool, optional): _description_. Defaults to True.
        max_directories (Optional[int], optional): max number of directories to search. Defaults to None.
        max_files (Optional[int], optional): max file number limit. Defaults to None.
        relative_path (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[List[str], bool, bool]: _description_
    """
    paths = []
    hit_max_directories = False
    hit_max_files = False
    for directory_idx, (directory, _, fnames) in enumerate(os.walk(root_path, followlinks=True)):
        if skip_hidden_directories and os.path.basename(directory).startswith("."):
            continue

        if max_directories is not None and directory_idx >= max_directories:
            hit_max_directories = True
            break

        paths += [
            os.path.join(directory, fname)
            for fname in sorted(fnames)
            if fname.lower().endswith(extensions)
        ]

        if not recursive:
            break

        if max_files is not None and len(paths) > max_files:
            hit_max_files = True
            paths = paths[:max_files]
            break

    if relative_path:
        paths = [os.path.relpath(p, root_path) for p in paths]

    return paths, hit_max_directories, hit_max_files


def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"


def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict
