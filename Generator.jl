module Generator
""" functions used for data generation """
export getSample, pointsToImage


using Images


# Parameters of model generation
const ModelComplexity = 5                 # Recursion depth for model generation
const Ops = [:+, :-, :.*]                 # Operators used to generate model
const ModelNumbers = 10*convert(Float32, randn()) # Numbers used in generation
const coordinateSystems = Function[       # Array of coordiante system converions
    (c1, c2)->(c1, c2)                    # Cartesian coordinates
    (c1, c2)->(c1.*cos(c2), c1.*sin(c2))  # Polar coordinates
]



function getLabel(y0, y)
  """ Simple 1-R^2, y0 is from model, y is the data """
  SSres = sum((y-y0).^2)
  SStot = sum((y-mean(y)).^2)
  return SSres/SStot
end

# Helper function for model(x, j)
numberOrX(x) = randbool() ? x : ModelNumbers

function model(x, j)
  """ Recursive function that generates expressions from:
  - operators
  - input vector x
  - random numbers """
  if j < ModelComplexity
      return Expr(:call, Ops[rand(1:size(Ops, 1))], numberOrX(x), model(x, j+1))
  else
      return x
  end
end

function sampleModel(SamplePoints::Int)
  """ Sample points from the model """
  coordinate1 = 2*pi*rand(Float32, SamplePoints)
  coordinate2 = eval(model(coordinate1, 0))
  return coordinateSystems[rand(1:end)](coordinate1, coordinate2)
end

function getSample(SamplePoints::Int, SampleSize::Int, SampleNoise)
  """ Generate data using generated model and apply gaussian noise """
  x, y0 = sampleModel(SamplePoints)  # Generates model and evaluates it
  y  = y0 + SampleNoise*convert(Array{Float32,1}, randn(SamplePoints))
  label = getLabel(y0,y)
  return x, y, label
end

# Discretization functions

function discretizeVector(vector::Array{Float32,1}, bins::Int)
  """ Change each value in vector to one of 1:bins """
  vectorInRange = vector - minimum(vector)              # move values so they start at 0
  vectorInterval = maximum(vector) - minimum(vector)    # get existing interval
  if vectorInterval == 0.
    return ones(size(vector))
  end
  vectorInRange = vectorInRange/vectorInterval*(bins-1) # scale values in to bins range
  vectorDiscrete = round(Uint32, vectorInRange)         # round the values
  return vectorDiscrete += 1                            # move values so they are array indices
end


function pointsToImage(x::Array{Float32,1}, y::Array{Float32,1}, SampleSize::Int)
    """ Take coordinates of points and create an image,
    where pixel brightness is proportional to point density in that region """
    xDiscrete = discretizeVector(x, SampleSize);
    yDiscrete = discretizeVector(y, SampleSize);
    SampleSpace = zeros(Uint16, SampleSize, SampleSize);
    for i = 1:size(x)[1]
        try
          SampleSpace[xDiscrete[i], yDiscrete[i]]+= 1;
        catch
          println(xDiscrete[i], ' ', yDiscrete[i])
          show(xDiscrete)
        end
    end
    return sc(grayim(SampleSpace))
end

end