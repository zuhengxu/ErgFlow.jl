using ForwardDiff, LinearAlgebra, Distributions, Random, Plots, StatsBase, SpecialFunctions
using BenchmarkTools, Plots, IrrationalConstants

###########33
# momentum Refreshement
#############
stream(x, u) = (sin(2.0*x + u) + 1.0)/2.0
constant(x, u) = π/4.0
stream_x(x, u) = (sin(2.0*x) + 1.0)/2.0
stream_u(x, u) = (sin(π/4.0*u) + 1.0)/2.0
# sample(x, i) = rand()

###############
# pseudo time shift
###############
mixer(u::Float64) = (u + π/16) % 1
inv_mixer(u::Float64) = (u + 1.0- π/16) % 1

###################3
# rotation matrix
####################
function rotation_mat(θ::Float64)
```
2d-rotation matrix for counterclockwise rotation by angle θ
```
    return  [cos(θ) -sin(θ); sin(θ) cos(θ)]
end



# X = [0:0.1:100 ;]
# plot(X, [stream(x, i) for (x, i) in zip(X, [1:1001 ;])] )
# plot(rand(1001))

################33
# normal
##############3
function pdf_normal(x)
    return 1.0/sqrt(2.0π)*exp(-0.5*x^2.0)
end

function lpdf_normal(x::Float64)
    return -0.5*(x^2.0 + log(2.0π))
end
function lpdf_normal(x::Vector{Float64})
    d = size(x, 1)
    return -0.5*x'*x -0.5* d*log(2.0π)
end

function ∇lpdf_normal(x)
    return -x 
end

function cdf_normal(x) 
    z = x / sqrt(2.0)
    if abs(x) > 5.7
        return 1.0- 0.5*erfc(z)
    else
        return 0.5 + 0.5*erf(z)
    end
end 

function invcdf_normal(x)
    if x > 0.5
        z = 2.0 - 2.0*x 
        return sqrt(2.0)*erfcinv(z)
    else 
        z = 2.0*x - 1.0
        return erfinv(z)
    end
end 

############################
# Laplace
############################

function cdf_laplace_std(x) 
    return 0.5 - 0.5 * sign(x) * expm1(-abs(x))
end 

function invcdf_laplace_std(x)
    return sign(0.5 - x) * log1p(-2.0*abs(x - 0.5))
end 

function pdf_laplace_std(x)
    return 0.5 * exp(-abs(x))
end

function lpdf_laplace_std(x::Float64)
    return log(0.5) - abs(x)
end

function lpdf_laplace_std(x::Vector{Float64})
    d = size(x, 1)
    return d*log(0.5) - sum(abs, x)
end

function ∇lpdf_laplace_std(x::Float64)
    return -sign(x)
end

function ∇lpdf_laplace_std(x::Vector{Float64})
    return -sign.(x)
end

function randl(size)
    return rand(Laplace(), size)
end

randl()=rand(Laplace())

############################
# Logistic
############################

function log_sigmoid(x)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end

function cdf_logistic_std(x) 
    # return 1. / (1 + exp(-x))
    return exp(log_sigmoid(x))
end 

function invcdf_logistic_std(x)
    # return log(x/(1. - x))
    return log(x) - log1p(-x)
end 

function pdf_logistic_std(x)
    return exp(-x) / (1 + exp(-x))^2
end

function lpdf_logistic_std(x::Float64)
    # return -x - 2. * logsumexp([0., -x])
    return -x -2. * log1p(exp(-x))
end

function lpdf_logistic_std(x::Vector{Float64})
    # d = size(x,1)
    # return sum(-x .- 2. * logsumexp_mult(hcat(zeros(d), -x)))
    return sum(lpdf_logistic_std.(x))
end

function ∇lpdf_logistic_std(x::Float64)
    return -1 + 2.0*exp(-x)/(exp(-x) + 1)
    # return -expm1(x)/(exp(x) + 1)
end

function ∇lpdf_logistic_std(x::Vector{Float64})
    # return -1 .+ 2. * exp.(-x) ./ (exp.(-x) .+ 1)
    return ∇lpdf_logistic_std.(x)
end

function randlogistic(size)
    return rand(Logistic(), size)
end

randlogistic()=rand(Logistic())

#####################
# ESH momentum (not even a valid density)
# TODO: look at its leapfrog 
#####################

function ∇lpdf_esh(x::Float64)
    return -1.0/x
end

function ∇lpdf_esh(x::Vector)
    d = size(x, 1)
    return -d.*x./(norm(x)^2.0)
end

###################3
## Exp(1/2): for the use of refreshing norm of isotropic Gaussian vector
###################
function cdf_exp(x::Float64)
    return  -expm1(-0.5*x)
end

function invcdf_exp(x::Float64)
    return -2.0*log1p(-x)
end

function pdf_exp(x::Float64)
    return 0.5*exp(-0.5*x)
end

function lpdf_exp(x::Float64)
    return -log(2.0) - 0.5*x  
end




###############
# self_defined function
##############
function logmeanexp_slice(w; dims = d)
    a = maximum(w, dims = dims)
    wl = mean(expm1.(w .- a) .+ 1.0, dims = dims)
    return a .+ log.(wl)
end


function logsumexp_stream(X)
    alpha = -Inf
    r = 0.0
    for x ∈ X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    return log(r) + alpha
end

function logsumexp(X)
    alpha = maximum(X)  # Find maximum value in X
    log(sum(exp, X.-alpha)) + alpha
end

function logmeanexp(X)
    N = size(X, 1)
    return logsumexp_stream(X) - log(N)
end
# n = 10_000
# X = 500.0*randn(n)

# @btime logsumexp($X)
# @btime logsumexp_stream($X)

