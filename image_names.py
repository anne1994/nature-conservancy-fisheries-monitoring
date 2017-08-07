#! /usr/bin/python2.7
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from os.path import isdir, join, exists
from shutil import copyfile
from os import listdir, makedirs




model_dir = '/data/models/slim/'


# all training images


folders=os.listdir("/data/models/slim/train/")
image_dir = "/data/models/slim/train/images/"

if not exists(image_dir):
    print 'creating directory %s' % image_dir
    makedirs(image_dir)

print("folders", folders)
for label in folders:
    cdir = join("/data/models/slim/train/",label)
    print("cdir", cdir)
    files=listdir(cdir)

    for f in files:
        os.rename( os.path.join(cdir,f),os.path.join(cdir, label+f ))

    files=listdir(cdir)
    for f in files:
        copyfile(join(cdir,f),join(image_dir,f))
print 'done'


