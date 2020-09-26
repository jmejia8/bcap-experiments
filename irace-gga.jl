include("distributed.jl")

addProcesses()

@everywhere using BCAP
@everywhere using Irace
@everywhere include("test-target-algorithms.jl")
using DataFrames
import CSV



function run_irace(nruns = 1)

    @everywhere benchmark = TestTargetAlgorithms.getBenchmark(:BPP)
    @everywhere bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:GGA)

    @everywhere target_runner(experiment, scenario) = begin
        instance = experiment[:instance]
        configuration = experiment[:configuration]
        Φ = Vector(configuration[1,:])
        value = targetAlgorithm(Φ, benchmark[instance], 1)
        return Dict(:cost => value)
    end


    @everywhere parameters_table = """
        P_size "" i (50, 400)
        p_m "" r (0, 0.9)
        p_c "" r (0, 0.6)
        k_ncs "" i (1, 6)
        k_cs "" i (1, 6)
        B_size "" r (0, 0.5)
        life_span "" i (1, 50)
    """
    @sync @distributed for nrun in 1:nruns
        maxExperiments = 5000
        print(">>>>>> RUN $nrun \t max experiments: $maxExperiments <<<<<<<<")
        scenario = Dict( :targetRunner => target_runner,
                         :instances => collect( 1:length(benchmark) ),
                         :maxExperiments => maxExperiments,
                         :digits => 6,
                         :deterministic => 1,
                         :logFile => "gga-run$(nrun)-exprmts$(maxExperiments).rdata")

        # Irace.checkIraceScenario(scenario, parameters_table)
        # get the results
        tuned = Irace.irace(scenario, parameters_table)

        println("========================================================")
        println("========================================================")
        println("===================== RUN $nrun ============================")
        println("========================================================")
        println("========================================================")
        # print the best configuration
        Irace.configurations_print(tuned)
    end
end

function gen_data(Φ, algorithm, benchmark)
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(algorithm)
    a = SharedArray{Float64}(length(benchmark))
    arr = BCAP.call_target_algorithm(targetAlgorithm, Φ, benchmark,
                                    calls_per_instance = 1,
                                    seed = 1,
                                    Errors_shared = a)
    arr
end

function gen_data_irace_gga()
    algorithm = :GGA
    benchmark = :BPP
    !isdir("csv") && mkdir("csv")
    t = CSV.read("csv/irace-$(algorithm)-$(benchmark).csv", DataFrame)
    for r in 1:10
        fname = "csv/irace-bpp-$(algorithm)-$(benchmark)-$(r).csv"

        isfile(fname) && continue
        @info("Generating $fname")
        Φ = Vector(t[r,:])
        @show(Φ)

        b = TestTargetAlgorithms.getBenchmark(benchmark, :test)
        println("|benchmark| = ", length(b))
        arr = gen_data(Φ, algorithm, b)
        @show sum(arr)
        writedlm(fname, arr, ',' )

        println("--------------------")
    end
end

function main()
    gen_data_irace_gga()
    # run_irace(10)
end

main()
