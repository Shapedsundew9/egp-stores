"""Logging configuration for all tests."""


from logging import basicConfig, DEBUG, INFO, getLogger
from os.path import join, dirname, basename, splitext, exists, isfile, islink, isdir
from os import makedirs, listdir, unlink
from shutil import rmtree


def get_logger(file, name):
    """Create a logger for unit test 'name'.
    
    If the logging folder does not exist it will be created.
    If the logging folder exists all files & folders within will be deleted.
    
    """
    location = join(dirname(file), 'logs', name)
    if not exists(location):
        makedirs(location)
    else:
        for filename in listdir(location):
            file_path = join(location, filename)
            try:
                if isfile(file_path) or islink(file_path):
                    unlink(file_path)
                elif isdir(file_path):
                    rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    basicConfig(
        format='[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
        filename=join(location, splitext(basename(file))[0] + '.log'),
        filemode='w',
        level=DEBUG)
    return getLogger(name), location
