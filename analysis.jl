using Bilevel
using JLD
import DelimitedFiles.writedlm
include("distributed.jl")
addProcesses(3)


@everywhere using BCAP
@everywhere include("test-target-algorithms.jl")


function getBenchmark(benchmark_name)
    benchmark = []
    if benchmark_name == :cec17_2
        benchmark = TestTargetAlgorithms.getBenchmark(2)
    elseif benchmark_name == :cec17_10
        benchmark = TestTargetAlgorithms.getBenchmark(10)
    elseif benchmark_name == :BPP
        benchmark = TestTargetAlgorithms.getBenchmark(:BPP)
    end

    benchmark
end

function gen_data(Φ, algorithm, benchmark)
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    a = SharedArray{Float64}(10, 31)
    arr = BCAP.call_target_algorithm(targetAlgorithm, Φ, benchmark, calls_per_instance= 31, Errors_shared = a)
    display(arr)
    arr
end

function main()
    !isdir("csv") && mkdir("csv")
    algorithm = :DE
    benchmark = :cec17_2


    println("=======================================")
    println("=======================================")

    for algorithm in [:DE, :PSO, :ECA]
        for benchmark in [:cec17_2, :cec17_10]
            for r in 1:10
                fname = "csv/$(algorithm)-$(benchmark)-$(r).csv"

                isfile(fname) && continue
                @info("Generating $fname")

                res = load("output/$(algorithm)-$(benchmark)-$(r).jld")["result"]

                @show(res.best_sol.x)

                arr = gen_data(res.best_sol.x, algorithm, getBenchmark(benchmark))
                writedlm(fname, arr, ',' )

                println("--------------------")
            end
        end
    end


end


main()
