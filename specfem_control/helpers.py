#!/usr/bin/env python

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