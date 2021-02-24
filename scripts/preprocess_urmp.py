import os
from pathlib import Path

import click

INSTRUMENTS = ("vn", "vc", "fl", "cl", "tpt", "sax", "tbn", "ob", "va", "bn", "hn", "db")

def create_directory(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Failed to create directory %s" % path)
        else:
            print("Created directory %s..." % path)
    else:
        print("Directory %s already exists. Skipping..." % path)

def create_directories(target_root):
    create_directory(target_root)
    for instrument in INSTRUMENTS:
        create_directory(os.path.join(target_root, instrument))