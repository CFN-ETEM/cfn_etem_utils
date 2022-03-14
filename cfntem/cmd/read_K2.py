import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM

def main():
    filepath_input = "/home/xiaqu/Documents/data/Converter_Project/Capture6/Capture6_.gtg"
    datacube = py4DSTEM.io.read(filepath_input, mem='MEMMAP')
    print("not implemented")