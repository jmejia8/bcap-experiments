import LinearAlgebra: norm
import Statistics: mean, std
using Plots
plotly(legend=false)

include("distributed.jl")
addProcesses(7)


@everywhere using BCAP
@everywhere include("test-target-algorithms.jl")


function F(x, y)
        mean_y = mean(y.instance_values, dims=2)[:,1]
        not_solved = .!y.solved_instances
        r = 0.01norm(x,1)

        if sum(not_solved) == 0
            return r
        end

        m = (mean_y[ not_solved ])

        I = m .> 1000

        m[I] = min.(900, 10log10.(m[I]))


        mean(m) + 1e3sum(not_solved) + r
end


function run_experiment(algorithm, nrun = 1)

    @info("Initializing...")



    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    parameters = Parameters(bounds, parmstype)

    benchmark = TestTargetAlgorithms.getBenchmark(10)

    # parallel_target(Φ, benchmark, seed) = target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed=seed)

    bcap = BCAP_config()
    res = configure(targetAlgorithm, parameters, benchmark, budget=200, debug = true, ul_func = F, bcap_config=bcap)

    display(res)

    for sol in res.population
        @show sol.x
        @show sol.F
        @show sol.f
    end

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
    benchmark = TestTargetAlgorithms.getBenchmark(10)

    ff(x, y) = begin
        p = [150, x, y]
        r = BCAP.call_target_algorithm(targetAlgorithm, p, benchmark, calls_per_instance=1, Errors_shared = SharedArray{Float64}(length(benchmark), 1))
        ll_y = BCAP.LLSolution(  instance_values = r )

        F(x, ll_y)
        #sum(ll_y.solved_instances)
    end

    x = range(0, 2, length=50)
    y = range(0, 1, length=50)

    println("Plotting...")
    surface(x, y, ff)



end

function ecaplot()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:ECA)
    benchmark = TestTargetAlgorithms.getBenchmark(2)


    ff(x, y) = begin
        p = [50, x, y]
        r = BCAP.call_target_algorithm(targetAlgorithm, p, benchmark, calls_per_instance=1, Errors_shared = SharedArray{Float64}(length(benchmark), 1))
        ll_y = BCAP.LLSolution(  instance_values = r )

        F(x, ll_y)
        #sum(ll_y.solved_instances)
    end

    x = 2:20
    y = range(0, 5, length=50)

    println("Plotting...")
    # plot(y, s->ff(2, s), markershape=:o  )
    surface(x, y, ff)

end


function randecaplot()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:ECA)
    benchmark = TestTargetAlgorithms.getBenchmark(10)


    ff(x, y) = begin
        p = [50, x, y]
        r = BCAP.call_target_algorithm(targetAlgorithm, p, benchmark, calls_per_instance=1, Errors_shared = SharedArray{Float64}(length(benchmark), 1))
        ll_y = BCAP.LLSolution(  instance_values = r )

        F(x, ll_y)
        #sum(ll_y.solved_instances)
    end

    n = 30
    x = rand(2:20, n)
    y = 5rand(n)

    z = [ ff(a, b) for a = x for b = y ]

    println("Plotting...")
    # plot(y, s->ff(2, s), markershape=:o  )
    scatter(x, y, z)

end


# main()
# deplot()
# ecaplot()

randecaplot()
