include("distributed.jl")
addProcesses(7)


@everywhere using BCAP
@everywhere include("test-target-algorithms.jl")
include("ann.jl")


function main()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:PSO)
    benchmark = TestTargetAlgorithms.getBenchmark(2)

    ll(p) = begin
        r = BCAP.call_target_algorithm(targetAlgorithm, p, benchmark, calls_per_instance=1, Errors_shared = SharedArray{Float64}(length(benchmark), 1))
        ll_y = BCAP.LLSolution(  instance_values = r )

        # F(x, ll_y)
        ll_y.solved_instances
    end

    n = 50

    xtrain = bounds[1,:] .+  (bounds[2,:] - bounds[1,:]) .* rand(length(bounds[1,:]), n)
    ytrain = zeros(Bool, length(benchmark), n)

    println("evaluatring...")
    println("evaluating...")
    for i = 1:n
        ytrain[:,i] = ll(view(xtrain, :, i))
    end

    display(xtrain)
    display(ytrain)

    m = aproximateANN(xtrain, ytrain, xtrain, ytrain, debug=true)

    # for i=1:n
    #     display([(m(xtrain[:,i]) .> 0) ytrain[:,i]])
    #     println("--")
    # end


end

main()
