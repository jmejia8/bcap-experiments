using Bilevel
using JLD
import DelimitedFiles: writedlm, readdlm
import Statistics: mean, median, std
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

function gen_data_irace()
    for algorithm in [:DE, :PSO, :ECA]
        for benchmark in [:cec17_10]
            t = readdlm("irace_csv/irace-$(algorithm)-$(benchmark).csv", ',')
            for r in 1:10
                fname = "csv/irace-$(algorithm)-$(benchmark)-$(r).csv"

                isfile(fname) && continue
                @info("Generating $fname")
                Φ = t[r,:]
                @show(Φ)

                arr = gen_data(Φ, algorithm, getBenchmark(benchmark))
                writedlm(fname, arr, ',' )

                println("--------------------")
            end
        end
    end
end

function gen_data_bcap()
    !isdir("csv") && mkdir("csv")

    algorithm = :DE
    benchmark = :cec17_2


    println("=======================================")
    println("=======================================")

    for algorithm in [:DE, :PSO, :ECA]
        for benchmark in [ :cec17_10]
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


function gen_table_bcap()

    for algorithm in [ :DE, :ECA, :PSO]
        @show algorithm
        for benchmark in [ :cec17_10]
            for r in 1:10
                fname = "csv/$(algorithm)-$(benchmark)-$(r).csv"

                res = load("output/$(algorithm)-$(benchmark)-$(r).jld")["result"]
                p = res.best_sol.x

                table = readdlm(fname, ',')
                mm = median(table, dims=2)

                print(r, " & ")
                print(join(p, " & ") )
                println(" & ", sum(mm .≈ 0.0), " \\\\ \\hline")
            end
        end
    end


end

function main()
    gen_table_bcap()
end


main()
