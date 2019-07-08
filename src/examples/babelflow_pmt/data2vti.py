import argparse
import numpy as np
import vtk
from functools import reduce


class DataBlock:
    def __init__(self, bin_file, param_file):
        np_data = np.fromfile(bin_file, dtype=np.float32)
        with open(param_file, "r") as f:
            line = f.readline()
            lo0, lo1, lo2, hi0, hi1, hi2 = list(map(int, line.strip().split()))
            self.low = np.array([lo0, lo1, lo2])
            self.high = np.array([hi0, hi1, hi2])
        self.shape = self.high - self.low + 1
        self.block_data = np_data.reshape(self.shape, order='F')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bins", nargs="+", help="binary files")
    parser.add_argument("-p", "--params", nargs="+", help="parameter files")
    parser.add_argument("-o", "--outf", help="output files")
    return parser.parse_args()


def data2vti(data):
    res = data.shape
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(res[0], res[1], res[2])
    float_array = vtk.vtkFloatArray()
    float_array.SetName("Scalars")
    float_array.SetNumberOfComponents(1)
    float_array.SetNumberOfTuples(reduce(lambda x, y: x * y, res))
    count = 0
    for z in range(res[2]):
        for y in range(res[1]):
            for x in range(res[0]):
                float_array.SetTuple1(count, data[x, y, z])
                count += 1
    image_data.GetPointData().AddArray(float_array)
    return image_data


def write_vti(filename, image_data):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()


def main():
    opt = parse_args()
    blocks = []
    low = np.ones([3]) * float("inf")
    high = np.zeros([3])
    for b, p in zip(opt.bins, opt.params):
        block = DataBlock(b, p)
        low = np.minimum(block.low, low)
        high = np.maximum(block.high, high)
        blocks.append(block)
    shape = high - low + 1
    data = np.zeros(shape.astype(np.int32))
    for b in blocks:
        data[b.low[0]:b.high[0] + 1, b.low[1]:b.high[1] + 1, b.low[2]:b.high[2] + 1] = b.block_data

    img_data = data2vti(data)
    write_vti(opt.outf, img_data)


if __name__ == "__main__":
    main()
