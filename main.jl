import LinearAlgebra: norm
import Statistics: mean, std
using JLD

include("distributed.jl")
addProcesses(3)


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


function run_experiment(algorithm, nrun, benchmark_name)

    @info("Initializing...")



    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    parameters = Parameters(bounds, parmstype)

    benchmark = []
    if benchmark_name == :cec17_2
        benchmark = TestTargetAlgorithms.getBenchmark(2)
    elseif benchmark_name == :cec17_10
        benchmark = TestTargetAlgorithms.getBenchmark(10)
    elseif benchmark_name == :BPP
        benchmark = TestTargetAlgorithms.getBenchmark(:BPP)
    end

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

    !isdir("output") && mkdir("output")

    runs = 10
    benchmark_name = :cec17_10

    for a = [:DE]
        for t = 1:runs
            fname = "output/$(a)-$(benchmark_name)-$(t).jld"

            if isfile(fname)
                @info "Solved "
                println(fname)
            end

            println("========================================")
            println("========================================")
            println("========================================")
            println(a, t)
            println("========================================")
            res = run_experiment(a, t, benchmark_name)
            @info "Saving in "
            println(fname)

            save(fname, "result", res)
        end
    end
end

main()
