using Bilevel
using JLD
import DelimitedFiles: writedlm, readdlm
import Statistics: mean, median, std
import Printf: @printf
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

function gen_data(Φ, algorithm, benchmark, ncalls = 31)
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    a = SharedArray{Float64}(length(benchmark), ncalls)
    arr = BCAP.call_target_algorithm(targetAlgorithm, Φ, benchmark, calls_per_instance= ncalls, Errors_shared = a)
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

    for algorithm in [:GGA]
        for benchmark in [ :BPP]
            for r in 1:10
                fname = "csv/bcap-$(algorithm)-$(benchmark)-$(r).csv"

                isfile(fname) && continue
                @info("Generating $fname")

                res = load("output/$(algorithm)-$(benchmark)-$(r).jld")["result"]

                @show(res.best_sol.x)

                arr = gen_data(res.best_sol.x, algorithm, getBenchmark(benchmark, :test, 1))
                writedlm(fname, arr, ',' )

                println("--------------------")
            end
        end
    end


end


function gen_table_bcap()

    for algorithm in [:GGA]
        @show algorithm
        for benchmark in [ :BPP]
            arr = []
            tab = nothing
            if algorithm == :ABC
                tab = readdlm("irace_csv/bcap-$(algorithm)-$(benchmark).csv", ',')
            end
            for r in 1:10
                fname = "csv/bcap-$(algorithm)-$(benchmark)-$(r).csv"

                if algorithm == :ABC
                    p = tab[r,:]
                else
                    res = load("output/$(algorithm)-$(benchmark)-$(r).jld")["result"]
                    p = res.best_sol.x
                end

                table = readdlm(fname, ',')
                mm = median(table, dims=2)
                mmean = mean(mean(table, dims=2))

                push!(arr, (p, sum(mm .≈ 0.0), mmean))


            end
            sort!(arr, lt = (a, b) -> a[2] < b[2] || (a[2] == b[2] && a[3] > b[3]), rev = true)
            for r in 1:10
                p, mm, mmean = arr[r]
                print("& ", r, " & ")
                print(join(p, " & ") )
                print(" & ", mm)
                @printf(" & %.3e \\\\\n", mmean)

            end
            println(" \\hline")
        end
    end


end


function gen_table_bcap()
    for algorithm in [:ABC, :ECA, :DE, :PSO]
        for benchmark in [:cec17_10]
            t = readdlm("irace_csv/irace-$(algorithm)-$(benchmark).csv", ',')
            arr = []
            @show algorithm
            for r in 1:10
                fname = "csv/irace-$(algorithm)-$(benchmark)-$(r).csv"

                Φ = t[r,:]
                table = readdlm(fname, ',')
                mm = median(table, dims=2)
                mmean = mean(mean(table, dims=2))

                push!(arr, (Φ, sum(mm .≈ 0.0), mmean))
            end

            sort!(arr, lt = (a, b) -> a[2] < b[2] || (a[2] == b[2] && a[3] > b[3]), rev = true)
            for r in 1:10
                p, mm, mmean = arr[r]
                print("& ", r, " & ")
                print(join(p, " & ") )
                print(" & ", mm)
                @printf(" & %.3e \\\\\n", mmean)

            end
            println(" \\hline")

        end
    end
end

function main()
    @info("gen csv")
    gen_data_bcap()
    @info("gen table")
    gen_table_bcap()
end


main()
