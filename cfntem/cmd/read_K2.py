
from cfntem.io.read_K2 import read_gatan_K2_bin

def main():
    filepath_input = "/home/xiaqu/Documents/data/Converter_Project/Capture6/Capture6_.gtg"
    datacube = read_gatan_K2_bin(filepath_input, mem='MEMMAP', sync_block_IDs=False)
    print(datacube.data.shape)

if __name__ == '__main__':
    main()
