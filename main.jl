import LinearAlgebra: norm
import Statistics: mean, std


include("distributed.jl")
addProcesses(4)


@everywhere using BCAP
@everywhere include("test-target-algorithms.jl")


function F(x, y)
        mean_y = mean(y.instance_values, dims=2)[:,1]
        not_solved = .!y.solved_instances
        r = 0.001norm(x,1)

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

    benchmark = TestTargetAlgorithms.getBenchmark(2)

    # parallel_target(Φ, benchmark, seed) = target_algorithm_parallel(targetAlgorithm, Φ, benchmark; seed=seed)

    bcap = BCAP_config()
    res = configure(targetAlgorithm, parameters, benchmark, budget=200, debug = true, ul_func = F, bcap_config=bcap)

    display(res)

    # for sol in res.population
    #     @show sol.x
    #     @show sol.F
    #     @show sol.f
    # end

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

main()
