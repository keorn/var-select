using HDF5, JLD
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

# Dataset size
const DataSize   = 1000         # Number od samples
const SampleSize = 100          # Number of x's in sample
const SampleMag  = 100          # Size of the largest x in a sample

# Parameters of model generation
const ModelComplexity = 10      # Recursion depth for model generation
const Ops = [:+, :-, :.*];      # Operators used to generate model
const ModelNumbers = 5*convert(Float32, randn()); # Numbers used in generation

# Distortion applied
const SampleNoise= 10000*rand() # Strandard deviation of the noise

# Dataset splitting parameters
const TrainPart  = 0.7          # Proportion of data used for training
const ValidPart  = 0.15         # Proportion of data used for validation

const TestPart   = 1-TrainPart-ValidPart

function getLabel(y0, y)
  """ Simple 1-R^2, y0 is from model, y is the data """
  SSres = sum((y-y0).^2)
  SStot = sum((y-mean(y)).^2)
  return SSres/SStot
end

# Helper function for model(x, j)
NumberOrX(x) = randbool() ? x : ModelNumbers

function model(x, j)
  """ Recursive function that generates expressions from:
  - operators
  - input vector x
  - random numbers """
  if j < ModelComplexity
      return Expr(:call, Ops[rand(1:size(Ops, 1))], NumberOrX(x), model(x, j+1))
  else
      return x
  end
end

function GenerateSample(SampleSize::Int, SampleMag::Int, SampleNoise)
  """ Generate data using f(x) and apply gaussian noise """
  x  = sort!(2*SampleMag*rand(Float32, SampleSize)-SampleMag)
  y0 = eval(model(x, 0))  # Generates model and evaluates it
  y  = y0 + SampleNoise*convert(Array{Float32,1}, randn(SampleSize))
  label = getLabel(y0,y)
  return [x, y], label
end

function GenerateSampleUniformNoise(SampleSize::Int, SampleMag::Int)
  """ Generate data using f(x) and apply gaussian noise - crudely attempts to get uniform distribution of R2"""
  # FIXME - apologies for the hacking - I'm new to Julia!
  finished = false
  x = 0
  y = 0
  label = 0
  while !finished
    x  = sort!(2*SampleMag*rand(Float32, SampleSize)-SampleMag)
    y0 = eval(model(x, 0))  # Generates model and evaluates it
    SampleNoise = rand() * std(y0)
    y  = y0 + SampleNoise*convert(Array{Float32,1}, randn(SampleSize))
    label = getLabel(y0,y)
#     println(label)
    finished = !isnan(label)
    if !finished
      println(y0)
      println(y)
    end
  end
  return [x, y], label
end

# Initialize the arrays
inputs = Array(Float32, DataSize, 2*SampleSize)
targets= Array(Float32, DataSize)

for j=1:DataSize
  """ Generate the data """
  inputs[j, :], targets[j] = GenerateSample(SampleSize, SampleMag, SampleNoise)
#  inputs[j, :], targets[j] = GenerateSampleUniformNoise(SampleSize, SampleMag)
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
