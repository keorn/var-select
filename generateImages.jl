using Images, Generator

#=
Generates labeled images dataset,
based on Nvidia DIGITS Image Folder Specification.
=#

# Number od samples
const DataSize   = 1000

# Sample parameters
const SamplePoints = 100
const SampleSize = 28          # Number of pixels
const SampleNoise = 62*rand() # Strandard deviation of the noise

labels= Array(Float32, DataSize)

println("Remove old data")
try rm("images", recursive=true) end
try mkpath("images/true") end
try mkpath("images/false") end

println("Start generation")
tic()
for j=1:DataSize
  """ Generate the data """
  x, y, labels[j] = getSample(SamplePoints, SampleSize, SampleNoise)
  if labels[j]<0.4
    imwrite(pointsToImage(x, y, SampleSize) ,string("images/true/", j, ".png"))
  else
    imwrite(pointsToImage(x, y, SampleSize) ,string("images/false/", j, ".png"))
  end
end
toc()

println("Label max:\t", maximum(labels))
println("Label mean:\t", mean(labels))
println("Label std:\t", std(labels))