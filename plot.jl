using BiApprox, MLKernels
using Plots
plotly(legend=false)



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
    benchmark = TestTargetAlgorithms.getBenchmark(2)


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

    X = [x y]

    println("Computing stuff...")
    z = [ ff(x[i], y[i]) for i = 1:length(x) ]

    method = KernelInterpolation(z, X, kernel= PolynomialKernel(), λ = 0.1)
    ff2 = approximate(method)

    println("Plotting...")
    # plot(y, s->ff(2, s), markershape=:o  )

    a = 2:20
    b = range(0, 5, length=length(a))
    surface(a, b, (w,z) -> ff2( [w, z] ))
    surface!(a, b, ff, fill=:blues)
    scatter!(x, y, z)

end


function randdeplot()
    bounds, parmstype, targetAlgorithm = TestTargetAlgorithms.getTargetAlgorithm(:DE)
    benchmark = TestTargetAlgorithms.getBenchmark(2)


    ff(x, y) = begin
        p = [50, x, y]
        r = BCAP.call_target_algorithm(targetAlgorithm, p, benchmark, calls_per_instance=1, Errors_shared = SharedArray{Float64}(length(benchmark), 1))
        ll_y = BCAP.LLSolution(  instance_values = r )

        F(x, ll_y)
        #sum(ll_y.solved_instances)
    end

    n = 60
    x = 2rand(n)
    y = rand(n)

    X = [x y]

    println("Computing stuff...")
    z = [ ff(x[i], y[i]) for i = 1:length(x) ]

    method = KernelInterpolation(z, X, kernel= PolynomialKernel(), λ = 0.1)
    ff2 = approximate(method)

    println("Plotting...")
    # plot(y, s->ff(2, s), markershape=:o  )

    a = range(0, 2, length=20)
    b = range(0, 1, length=length(a))
    surface(a, b, (w,z) -> ff2( [w, z] ))
    surface!(a, b, ff, fill=:blues)
    scatter!(x, y, z)

end


randdeplot()
