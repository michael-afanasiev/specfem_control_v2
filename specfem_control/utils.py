#!/usr/bin/env python

import os
import errno
import shutil


class colours:
    red = '\033[91m'
    ylw = '\033[93m'
    blu = '\033[94m'
    rst = '\033[0m'


def print_blu(message):
    print colours.blu + message + colours.rst


def print_ylw(message):
    print colours.ylw + message + colours.rst


def print_red(message):
    print colours.red + message + colours.rst


def mkdir_p(path):
    """
    Makes a directory and doesn't fail if the directory already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise


def safe_copy(source, dest):
    """
    Sets up a file copy that won't fail for a stupid reason.
    """
    source = os.path.join(source)
    dest = os.path.join(dest)

    if (os.path.isdir(source)):
        return
    if not (os.path.isdir(dest)):
        return
    try:
        shutil.copy(source, dest)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def safe_copy_file(source, dest):
    """
    Sets up a file copy that won't fail for a stupid reason.
    """
    source = os.path.join(source)
    dest = os.path.join(dest)

    if (os.path.isdir(source)):
        return
    if (os.path.isdir(dest)):
        return
    try:
        shutil.copyfile(source, dest)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def safe_sym_link(source, dest):
    """
    Sets up symbolic links that won't fail for a stupid reason.
    """

    source = os.path.join(source)
    dest = os.path.join(dest)

    if (os.path.isdir(source)):
        return

    try:
        os.symlink(source, dest)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def sym_link_directory(source, dest):
    """
    Recursively symlinks all files in a 'source' directory.
    """
    for file in os.listdir(source):
        safe_sym_link(os.path.join(source, file), os.path.join(dest, file))


def copy_directory(source, dest, exc=None, only=None):
    """
    Recursively copies all files in a 'source' directory.
    """
    for file in os.listdir(source):
        if exc and file in exc:
            continue
        if only and file not in only:
            continue
        safe_copy(os.path.join(source, file), dest)
