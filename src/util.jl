using IrrationalConstants
include("momentum.jl")
###########33
# momentum Refreshement
#############
stream(x, u) = (sin(2.0*x + u) + 1.0)/2.0
constant(x, u) = π/16.0
stream_x(x, u) = (sin(2.0*x) + 1.0)/2.0
stream_u(x, u) = (sin(π/4.0*u) + 1.0)/2.0
# resample(x, i) = rand()

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

###############
# logsumexp function
##############
function logmeanexp_slice(w; dims = d)
```
logsumexp function works on a specific slice of array 
```
    a = maximum(w, dims = dims)
    wl = mean(expm1.(w .- a) .+ 1.0, dims = dims)
    return a .+ log.(wl)
end


function logsumexp_stream(X)
```
logsumexp function without memory allocation
adapt from "http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html"
```
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

