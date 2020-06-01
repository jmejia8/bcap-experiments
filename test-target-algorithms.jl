module TestTargetAlgorithms

using Metaheuristics
import Random: seed!
import ...Instance

const desired_accu = 1e-4

using CEC17


function getBenchmark(D::Int)
    bounds = Array([ -100.0ones(D) 100.0ones(D) ]')

    benchmark = Instance[]
    for i = 1:10
        fun = x -> abs(cec17_test_func(x, i) - 100i)
        inst = Instance(desired_accu, Dict(:f => fun, :bounds => bounds), i)
        push!(benchmark, inst)
    end

    benchmark
end

function getBenchmark(p::Symbol)
    if p != :BPP
        error("Instance for BPP")
    end

    benchmark = Instance[]
    nams = ["N2c2w1_s.txt","N2c3w2_d.txt","N2c3w2_l.txt","N2c3w2_s.txt","N2c3w4_a.txt",
    "N3c1w1_o.txt","N3c3w2_b.txt","N3c3w2_h.txt","N3c3w4_c.txt","N4c3w2_a.txt",
    "N4c3w4_s.txt","N1w1b3r7.txt","N2w1b2r5.txt","N2w1b2r8.txt","N4w1b2r6.txt",
    "N4w2b1r3.txt","Hard2.txt","Hard3.txt","t60_00.txt","t120_01.txt","u250_12.txt",
    "hBPP832.txt","hBPP40.txt","hBPP360.txt","hBPP645.txt","hBPP742.txt","hBPP766.txt",
    "hBPP60.txt","hBPP13.txt","hBPP195.txt","hBPP709.txt","hBPP785.txt","hBPP47.txt",
    "hBPP181.txt","hBPP485.txt","hBPP640.txt","hBPP144.txt","hBPP561.txt","hBPP781.txt",
    "hBPP900.txt","hBPP178.txt","hBPP419.txt","hBPP531.txt","hBPP814.txt","BPP83.txt",
    "BPP_56.txt","BPP_71.txt","TEST0014.txt","TEST0030.txt","TEST0005.txt"]
    ops = [43,41,41,43,43,98,81,82,88,203,216,18,34,34,168,100,56,55,20,40,105,60,
        59,62,58,64,62,63,67,64,67,68,71,72,71,74,73,72,71,75,80,80,83,81,18,21,
        21,23,27,28]


    hard_instances = [22,23,25,27,28,29,31,32,34,35,40,41,42,48]


    # [Instance(0.5i, nothing, i) for i = 1:10]

    for i = 1:length(nams)

        push!(benchmark, Instance(ops[i], " --max_gen 100 --instance /home/jesus/develop/repos/GGA-CGT/instances/$(nams[i])", i))

    end


    benchmark
end

#                    N    Ne   limit
const ABC_bounds  = [10   0.1     0;
                     500  1.0   1000]
const ABC_parmstype = [Int, Float64, Int]

#                    N    K   η_max
const ECA_bounds  = [ 10  2    0.0;
                      500 10    4]
const ECA_parmstype = [Int, Int, Float64]

#                     N  F  CR
const DE_bounds   = [10  0  0;
                     500 2  1.0]
const DE_parmstype = [Int, Float64,  Float64]

#                    N   C1   C2   omega
const PSO_bounds  = [10    0   0    0;
                     500   4   4    1.0]
const PSO_parmstype = [Int, Float64, Float64, Float64]

const options = Options(f_calls_limit=0, f_tol = desired_accu)
const information = Information(f_optimum=0.0)

function getTargetAlgorithm(name)
    if name == :ECA
        return ECA_bounds, ECA_parmstype, ECA_target
    elseif name == :DE
        return DE_bounds, DE_parmstype, DE_target
    elseif name == :PSO
        return PSO_bounds, PSO_parmstype, PSO_target
    elseif name == :GGA
        b = [
             50   0    0    1   1  0   1;
             400 0.9  0.6   6   6  0.5  50
            ]
        return b, [Int, Float64, Float64, Int, Int, Float64, Int], GGA_target
    else
        @error "Only `:ECA`, `:DE`, `:PSO` or `GGA` are permited."
    end
end

function ECA_target(Φ, instance, seed = 0)
    seed!(seed)
    Φ[1] = round(Φ[1])
    Φ[2] = round(Φ[2])

    eca = ECA(  N = Int(Φ[1]),
                K = Int(Φ[2]),
                η_max = Φ[3],
                options = options,
                information = information)

    res = optimize(instance.value[:f], instance.value[:bounds], eca)

    res.best_sol.f < instance.optimum ? 0.0 : res.best_sol.f
end

function DE_target(Φ, instance, seed = 0)
    seed!(seed)
    Φ[1] = round(Φ[1])

    alg = DE(   N  = Int(Φ[1]),
                F  = Φ[2],
                CR = Φ[3],
                options = options,
                information = information)

    res = optimize(instance.value[:f], instance.value[:bounds], alg)

    res.best_sol.f < instance.optimum ? 0.0 : res.best_sol.f
end

function PSO_target(Φ, instance, seed = 0)
    seed!(seed)
    Φ[1] = round(Φ[1])

    alg = PSO(  N  = Int(Φ[1]),
                C1 = Φ[2],
                C2 = Φ[3],
                ω  = Φ[4],
                options = options,
                information = information)

    res = optimize(instance.value[:f], instance.value[:bounds], alg)

    res.best_sol.f < instance.optimum ? 0.0 : res.best_sol.f
end

function GGA_target(Φ, instance, seed = 0)
    flags = ["--P_size", "--p_m","--p_c", "--k_ncs", "--k_cs","--B_size","--life_span"]
    shell_target(Φ, instance, seed; exe="GGA-CGT", exe_path="/home/jesus/develop/repos/GGA-CGT", flags=flags)
end

function shell_target(Φ, instance, seed=0; exe = "", exe_path = "", flags = String[], seed_flag="--seed")
    if isempty(flags)
        error("Configure your target runner.")
        return 0
    end

    vals = " " .* string.(Φ) .* " "
    exe_name = joinpath(exe_path, exe)
    settings = (flags .* vals)
    instance_str = instance.value


    cmd = string(exe_name, " ", prod(settings), instance_str, " ", seed_flag, " ", string(seed))
    cmd  = split(cmd)

    fx = 1e6

    try
        fx = parse(Float64, read(`$cmd`, String))
    catch
        println(`$cmd`)

        @error("Target algorithm fail, penalizing result 1e6.")
        println(Φ)
        fx = 1e6
    end

    if true
        o = fx - instance.optimum
        if o < 0
            println("Check instance ")
            display(instance)
            @error("Ill-posed benchmark")
            return 0
        end
        fx = o
    end

    return fx

end



export getTargetAlgorithm, getBenchmark

end
