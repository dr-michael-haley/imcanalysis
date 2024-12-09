#!/usr/bin/env python3
"""
This script recursively renames files within a specified directory, replacing
an old substring in the filenames with a new substring. By default, it operates
on files matching a specified file extension (default: '*.tiff').

Usage:
    ./script.py <base_directory> <old> <new> [<ext>]

Example:
    ./script.py /path/to/directory old_string new_string "*.tiff"

This will rename all files in `/path/to/directory` and its subdirectories,
replacing occurrences of "old_string" in filenames with "new_string" for all
files ending with ".tiff".
"""

import sys
from pathlib import Path


def recursive_rename(base_directory='.', old='', new='', ext='*.tiff'):
    """
    Recursively rename files in a given directory.

    Parameters
    ----------
    base_directory : str
        The starting directory in which to perform the renaming.
    old : str
        The old substring that should be replaced in filenames.
    new : str
        The new substring that replaces the old substring.
    ext : str
        A glob pattern for files to be renamed (default: '*.tiff').

    Returns
    -------
    None
    """
    base_path = Path(base_directory)

    # Iterate over all files matching the given extension pattern, recursively
    for file in base_path.rglob(ext):
        old_name = file.name

        # Check if the old substring is present in the file name
        if old in old_name:
            # Replace the substring and form the new file path
            new_name = old_name.replace(old, new)
            new_path = file.with_name(new_name)

            # Rename the file
            file.rename(new_path)

            print(f"Renamed: {file} -> {new_path}")


if __name__ == "__main__":
    # Check for minimum required arguments
    # Expected arguments: base_directory old new [ext]
    if len(sys.argv) < 4:
        print("Usage: script.py <base_directory> <old> <new> [<ext>]")
        sys.exit(1)

    # Parse command line arguments
    base_directory = sys.argv[1]
    old_string = sys.argv[2]
    new_string = sys.argv[3]

    # If the extension is not provided, default to '*.tiff'
    extension = sys.argv[4] if len(sys.argv) > 4 else '*.tiff'

    # Call the main function
    recursive_rename(base_directory=base_directory, old=old_string, new=new_string, ext=extension)
