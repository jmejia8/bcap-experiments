import LinearAlgebra: norm
import Statistics: mean, std
using Distributed
using SharedArrays
@everywhere using BCAP
using Plots
plotly(legend=false)
@everywhere include("test-target-algorithms.jl")


function F(x, y)
        mean_y = mean(y.instance_values, dims=2)[:,1]
        not_solved = .!y.solved_instances

        m = norm(mean_y[ not_solved ])
        if isnan(m)
            m = 0.0
        end

        n = 1.0 +  log10(1.0 + m)


        m + (10^n)*sum(not_solved) + 0.005norm(x,1)
end

function target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed = 1, calls_per_instance = 1)
    if typeof(benchmark) <: Instance
        return targetAlgorithm(Φ, benchmark, seed)
    end


    Errors_shared = SharedArray{Float64}(length(benchmark), calls_per_instance)


    @sync @distributed for i = 1:length(benchmark)
        for r = 1:calls_per_instance
            err = targetAlgorithm(Φ, benchmark[i], seed)

            Errors_shared[i, r] = err
        end
    end

    Errors = Matrix(Errors_shared)

    Errors_shared = nothing

    return  Errors

end


function run_experiment(algorithm, nrun = 1)

    @info("Initializing...")



    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    parameters = Parameters(bounds, parmstype)

    benchmark = TestTargetAlgorithms.getBenchmark(2)

    parallel_target(Φ, benchmark, seed) = target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed=seed)

    bcap = BCAP_config(p = 3)
    res = configure(parallel_target, parameters, benchmark, debug = true, ul_func = F, bcap_config=bcap)

    display(res)


    res
end


function main()
    runs = 1

    for a = [:DE]
        for t = 1:runs
            println("========================================")
            println("========================================")
            println("========================================")
            println(a, t)
            println("========================================")
            return run_experiment(a, t)
        end
    end
end




function deplot()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:DE)
    benchmark = TestTargetAlgorithms.getBenchmark(2)

    # benchmark = benchmark[8:10]
    parallel_target(Φ, benchmark, seed) = target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed=seed)


    ff(x, y) = begin
        p = [50, x, y]
        r = BCAP.call_target_algorithm(parallel_target, p, benchmark, calls_per_instance=1)
        ll_y = BCAP.LLSolution(  instance_values = r )
        sum(ll_y.solved_instances)
    end

    x = range(0, 2, length=50)
    y = range(0, 1, length=50)

    println("Plotting...")
    surface(x, y, ff)

end

function ecaplot()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:ECA)
    benchmark = TestTargetAlgorithms.getBenchmark(2)

    # benchmark = benchmark[8:10]
    parallel_target(Φ, benchmark, seed) = target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed=seed)


    ff(x, y) = begin
        p = [50, x, y]
        r = BCAP.call_target_algorithm(parallel_target, p, benchmark, calls_per_instance=1)
        ll_y = BCAP.LLSolution(  instance_values = r )

        log.(1 + F(x, ll_y))
        #sum(ll_y.solved_instances)
    end

    x = range(0, 10, length=150)

    println("Plotting...")
    plot(x, s->ff(2, s), markershape=:o  )

end

main()
# ecaplot()
