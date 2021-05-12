import shutil

from wzk.files import safe_create_dir, rel2abs_path
from definitions import (PROJECT_DATA_SAMPLES,
                         DLR_HOMELOCAL_DATA_SAMPLES, DLR_USERSTORE_DATA_SAMPLES,
                         WORLD_DB, PATH_DB)


def get_sample_dir(directory, full_path=True):

    directory = rel2abs_path(path=directory, abs_dir=PROJECT_DATA_SAMPLES)

    if not full_path:
        directory = directory[len(PROJECT_DATA_SAMPLES):]

    return directory


def __copy_us2lcl(sample_dir, src='US'):
    """
    Copy sample files from the net drive USERSTORE to HOMELOCAL on the machine, to minimize net traffic.
    Copy sample files to the net drive USERSTORE, to back it up and access it from all machines on the DLR
    """

    sample_dir = get_sample_dir(directory=sample_dir, full_path=False)
    us = DLR_USERSTORE_DATA_SAMPLES + sample_dir
    hl = DLR_HOMELOCAL_DATA_SAMPLES + sample_dir
    if src == 'US':
        src, dst = us, hl
    elif src == 'HL':
        src, dst = hl, us
    else:
        raise ValueError(f"Unknown source / destination for copy {sample_dir}")
    safe_create_dir(dst)

    try:
        shutil.copy(src=src + WORLD_DB, dst=dst)
    except FileNotFoundError:
        print(f"No {WORLD_DB} found for {sample_dir}")

    try:
        shutil.copy(src=src + PATH_DB, dst=dst)
    except FileNotFoundError:
        print(f"No {PATH_DB} found for {sample_dir}")


def copy_userstore2homelocal(sample_dir):
    __copy_us2lcl(sample_dir, src='US')


def copy_homelocal2userstore(sample_dir):
    __copy_us2lcl(sample_dir, src='HL')


# ----------------------------------------------------------------------------------------------------------------------
# DLR Remote Access

# VNC
# On remote PC (ie. pandia):
#   vncpasswd  -> set password (optional)
#   vncserver -geometry 1680x1050 -depth 24 -> start server, with scaled resolution for mac
#
# On local PC (ie. my laptop):
#   ssh -l tenh_jo -L 5901:pandia:5901 ssh.robotic.dlr.de  -> set link port to the remote display
#   open VNC and connect to home: 127.0.0.1:1

# SSH
# ssh -D 8080 -l tenh_jo ssh.robotic.dlr.de
