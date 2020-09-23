include("distributed.jl")

addProcesses(4)

@everywhere using BCAP
@everywhere using Irace
@everywhere include("test-target-algorithms.jl")



function main(nrun = 1)

    @everywhere benchmark = TestTargetAlgorithms.getBenchmark(:BPP)
    @everywhere bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:GGA)

    @everywhere target_runner(experiment, scenario) = begin
        instance = experiment[:instance]
        configuration = experiment[:configuration]
        Φ = Vector(configuration[1,:])
        value = targetAlgorithm(Φ, benchmark[instance], experiment[:seed])
        return Dict(:cost => value)
    end


    values = SharedArray{Float64}(length(benchmark))
    targetRunnerParallel(experiments, the_target_runner, scenario) = begin
        print(">>>>>>  I am parallel")
        # if length(experiments) > length(benchmark)
        #     values = SharedArray{Float64}(length(benchmark))
        # end

        for i in 1:length(experiments)
            values[i] = target_runner(experiments[i], scenario)[:cost]
        end

        return [Dict(:cost => v) for v in Vector(values)[1:length(experiments)]]
    end

    parameters_table = """
        P_size "" i (50, 400)
        p_m "" r (0, 0.9)
        p_c "" r (0, 0.6)
        k_ncs "" i (1, 6)
        k_cs "" i (1, 6)
        B_size "" r (0, 0.5)
        life_span "" i (1, 50)
    """
    scenario = Dict( :targetRunner => target_runner,
                     :targetRunnerParallel => targetRunnerParallel,
                     :instances => collect( 1:length(benchmark) ),
                  :maxExperiments => 500,
                  # Do not create a logFile
                  :logFile => "gga-test-$nrun.rdata")

    # Irace.checkIraceScenario(scenario, parameters_table)
    # get the results
    tuned = Irace.irace(scenario, parameters_table)

    # print the best configurations
    # Irace.configurations_print(tuned)
end

main()
