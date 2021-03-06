import LinearAlgebra: norm
import Statistics: mean, std
using JLD

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

        m[I] = min.(999.0, m[I])


        mean(m) + 1e2sum(not_solved) + r
end


function run_experiment(algorithm, nrun, benchmark_name, calls_per_instance=4000)

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

    bcap = BCAP_config(N = 10)
    b = calls_per_instance
    res = configure(targetAlgorithm, parameters, benchmark, budget=b,
                        debug = true,
                        ul_func = F,
                        bcap_config=bcap)

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

    runs = 1
    benchmark_name = :cec17_10

    for a = [ :GGA]
        if a == :GGA
            benchmark_name = :BPP
        end
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
            res = run_experiment(a, t, benchmark_name, 2000)
            @info "Saving in "
            println(fname)


            save(fname, "result", res)
        end
    end
end

main()
