#ENV["MOCHA_USE_NATIVE_EXT"] = "true"
#ENV["OMP_NUM_THREADS"] = 1
#blas_set_num_threads(1)

using Mocha
srand(12345678)

data  = HDF5DataLayer(name="train-data",source="data/train.txt",batch_size=64)
conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])
conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])
fc1   = InnerProductLayer(name="ip1",output_dim=500,neuron=Neurons.ReLU(),bottoms=[:pool2],
                          tops=[:ip1])
fc2   = InnerProductLayer(name="ip2",output_dim=2,bottoms=[:ip1],tops=[:ip2])
loss  = SoftmaxLossLayer(name="loss",bottoms=[:ip2,:label])

backend = CPUBackend()
init(backend)

common_layers = [conv, pool, conv2, pool2, fc1, fc2]
net = Net("var-train", backend, [data, common_layers..., loss])

exp_dir = "snapshots"
solver_method = SGD()
params = make_solver_parameters(solver_method, max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
    load_from=exp_dir)
solver = Solver(solver_method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
accuracy = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
test_net = Net("var-test", backend, [data_test, common_layers..., accuracy])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
