include("Generator.jl")
using Generator, HDF5

#=
Generates labeled blob,
compatibile with Caffe and Mocha.jl
Blob is 4D-tensor: width, height, channels, num
=#

# Dataset parameters
const DataSize = 1000
const TrainPart = 0.8

# Sample parameters
const SamplePoints = 100
const SampleSize   = 28          # Number of pixels
const SampleNoise  = 62*rand()   # Strandard deviation of the noise

println("Remove old data")
try rm("data/train.hdf5") end
try rm("data/test.hdf5") end

data  = Array(Float32, SampleSize, SampleSize, 1, DataSize)
label = Array(Float32, DataSize)
exactLabel = Array(Float32, DataSize)

println("Start generation")
tic()
for j=1:DataSize
  """ Generate the data """
  x, y, exactLabel[j] = getSample(SamplePoints, SampleSize, SampleNoise)
  data[:, :, 1, j]  = pointsToDense(x, y, SampleSize)
  if exactLabel[j]<0.4
    label[j] = 1
  else
    label[j] = 0
  end
end
toc()

Index(p) = round(Int, DataSize*p)

# Write files
h5write("data/train.hdf5", "data",  data[:, :, 1, 1:Index(TrainPart)])
h5write("data/train.hdf5", "label", label[1:Index(TrainPart)])
h5write("data/test.hdf5", "data",   data[:, :, 1, Index(TrainPart):end])
h5write("data/test.hdf5", "label",  label[Index(TrainPart):end])

println("Label max:\t", maximum(exactLabel))
println("Label mean:\t", mean(exactLabel))
println("Label std:\t", std(exactLabel))
println("Related:\t", sum(label))
println("Unrelated:\t", (DataSize-sum(label)))
