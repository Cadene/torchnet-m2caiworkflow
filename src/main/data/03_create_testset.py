# coding: utf-8
import numpy as np
import pandas as pd
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dirinterim", default="data/interim", help="Directory to save testset.txt")
parser.add_option("--dirimagestest", default="data/interim/imagesTest")

(options, args) = parser.parse_args()

os.system("ls "+options.dirimagestest+" > "+options.dirinterim+"/testset.txt")
os.system("sed -i \"s/.jpg/.jpg, -1/\" "+options.dirinterim+"/testset.txt")
