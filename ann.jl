using Flux
using Flux: mae, mse, throttle, @epochs
using Flux.Data: DataLoader
import LinearAlgebra: norm
import Printf: @printf

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += mae(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        yy = map( a -> a > 0, cpu(model(x)))
        s = 0
        for i = 1:size(x,2)
            s += Int(norm(yy[:,i] - y[:,i], 1) < 1 )
        end
        acc += s/size(x,2)
    end
    acc/length(data_loader)
end

function aproximateANN(xtrain, ytrain, xtest, ytest; epochs = 500, batchsize = 10, debug=false)
    nparms = size(xtrain, 1)
    ndata  = size(xtrain, 2)
    nclasses = size(ytrain, 1)

    m = Chain(Dense(nparms, 2nparms, relu),
        Dense(2nparms, 3nparms, relu),
        Dense(3nparms, 4nparms, relu),
        Dense(4nparms, nclasses, sigmoid),
    )

    opt = ADAMW()
    loss(x,y) =  mae(m(x), y)


    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize)
    #
    #

    evalcb = () -> @show(accuracy(train_data, m))

    m_best = deepcopy(m)
    ab_best = 0
    for e in 1:epochs
        Flux.train!(loss, params(m), train_data, opt)

        a, b = accuracy(train_data, m), accuracy(test_data, m)
        debug && @printf("epoch: %d\t acc_train: %.2f\t acc_test: %.2f\t loss: %.3g",e , a, b, loss_all(train_data, m))

        if a + b > ab_best
            m_best = deepcopy(m)
            ab_best = a + b
            debug && print("*")
        end

        debug && println("")

    end

    debug && println("acc: ", 0.5ab_best)

    return m_best

end

function test()

    ndata = 100
    nclasses = 10
    nparms = 5


    xtrain = -5 .+ 10rand(nparms, ndata)
    ytrain = zeros(Bool, nclasses, ndata)
    ytrain[1:nparms,:] = xtrain .< 0

    xtest = -5 .+ 10rand(nparms, ndata)
    ytest = zeros(Bool, nclasses, ndata)
    ytest[1:nparms,:] = xtest .< 0

    @time aproximateANN(xtrain, ytrain, xtest, ytest, debug=true)

end
