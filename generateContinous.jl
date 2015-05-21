using HDF5, JLD, Generator.getSample
#using MAT

#=
Generates HDF5 file with labeled datasets:
"X_train"
"y_train"
"X_valid"
"y_valid"
"X_test"
"y_test"

All numbers are kept as Float32 for size
and compatibility with Theano GPU implementation.
=#

# Number od samples
const DataSize   = 1000

# Sample size
const SamplePoints = 100
const SampleSize = 30

# Distortion applied
const SampleNoise= 1000*rand()  # Strandard deviation of the noise

# Dataset splitting parameters
const TrainPart  = 0.7          # Proportion of data used for training
const ValidPart  = 0.15         # Proportion of data used for validation

const TestPart   = 1-TrainPart-ValidPart

# Initialize the arrays
inputs = Array(Float32, DataSize, 2*SamplePoints)
targets= Array(Float32, DataSize)

for j=1:DataSize
  """ Generate the data """
  inputs[j, 1:SamplePoints], inputs[j, SamplePoints+1:end], targets[j] = getSample(SamplePoints, SampleSize, SampleNoise)
end

println("Target max:\t", maximum(targets))
println("Target mean:\t", mean(targets))
println("Target std:\t", std(targets))

Index(p) = round(DataSize*p)

jldopen("dataset.jld", "w") do file
  """ Partition and save the data """
  write(file, "X_train", inputs[1:Index(TrainPart), :])
  write(file, "y_train", targets[1:Index(TrainPart)])

  write(file, "X_valid", inputs[Index(TrainPart):Index(TrainPart+ValidPart), :])
  write(file, "y_valid", targets[Index(TrainPart):Index(TrainPart+ValidPart)])

  write(file, "X_test", inputs[Index(1-TestPart):end, :])
  write(file, "y_test", targets[Index(1-TestPart):end])
end

#=
c = jldopen("dataset.jld", "r") do file
    read(file, "X_valid")
end

print(size(c))
=#
#=
file = matopen("dataset.mat", "w")
  write(file, "inputs", inputs)
  write(file, "outputs", outputs)
close(file)
=#
