using Base.Threads: @threads
using ProgressMeter, Flux
using Zygote:Buffer
# using LogExpFunctions

##### struct of ergflow 
struct HamFlow <: ErgodicFlow
    # Ham Flow struct (we dont put n_mcmc inside for ease of changing K in flow_fwd) 
    d::Int64
    n_lfrg::Int64 # number of n_lfrg between refresh
    # target 
    logp::Function # log target density
    ∇logp::Function
    # VI distribution
    q_sampler::Function
    logq0::Function
    # momentum 
    ρ_sampler::Function
    lpdf_mom::Function
    ∇logp_mom::Function
    cdf_mom::Function 
    invcdf_mom::Function
    pdf_mom::Function
    # pseudo refreshment
    generator::Function
    time_shift::Function
    inv_timeshift::Function
end


##################
# refreshments
##################
function pseudo_refresh_coord(o::HamFlow, z, ρ, u)
    # buf = Buffer(ρ)
    ρ1  = copy(ρ)   
    for i in 1:o.d 
        ξ = (o.cdf_mom(ρ1[i]) + o.generator(z[i], u)) % 1.0
        ρ1[i] = o.invcdf_mom(ξ)
        u = o.time_shift(u)
    end
    return ρ1, u 
end

function pseudo_refresh_coord!(o::HamFlow, z, ρ, u)
    for i in 1:o.d 
        ξ = (o.cdf_mom(ρ[i]) + o.generator(z[i], u)) % 1.0
        ρ[i] = o.invcdf_mom(ξ)
        u = o.time_shift(u)
    end
    return u 
end

function pseudo_refresh(o::HamFlow, z, ρ, u)
    buf = Buffer(ρ)
    for i in 1:o.d 
        ξ = (o.cdf_mom(ρ[i]) + o.generator(z[i], u)) % 1.0
        buf[i] = o.invcdf_mom(ξ)
    end
    u = o.time_shift(u)
    return copy(buf), u     
end

function pseudo_refresh!(o::HamFlow, z, ρ, u)
    for i in 1:o.d 
        ξ = (o.cdf_mom(ρ[i]) + o.generator(z[i], u)) % 1.0
        ρ[i] = o.invcdf_mom(ξ)
    end
    u = o.time_shift(u)
    return u     
end
# inverse of refreshment
function inv_refresh_coord(o::HamFlow, z, ρ, u)
    # buf = Buffer(ρ)
    ρ1  = copy(ρ)   
    for i in o.d:-1:1
        u = o.inv_timeshift(u)
        ξ = (o.cdf_mom(ρ1[i])+ 1.0 - o.generator(z[i], u)) % 1.0
        ρ1[i] = o.invcdf_mom(ξ)
    end
    return ρ1, u
end

function inv_refresh_coord!(o::HamFlow, z, ρ, u)
    for i in o.d:-1:1
        u = o.inv_timeshift(u)
        ξ = (o.cdf_mom(ρ[i])+ 1.0 - o.generator(z[i], u)) % 1.0
        ρ[i] = o.invcdf_mom(ξ)
    end
    return u
end

function inv_refresh(o::HamFlow, z, ρ, u)
    buf = Buffer(ρ)
    u = o.inv_timeshift(u)
    for i in o.d:-1:1
        ξ = (o.cdf_mom(ρ[i])+ 1.0 - o.generator(z[i], u)) % 1.0
        buf[i] = o.invcdf_mom(ξ)
    end
    return copy(buf), u
end

function inv_refresh!(o::HamFlow, z, ρ, u)
    u = o.inv_timeshift(u)
    for i in o.d:-1:1
        ξ = (o.cdf_mom(ρ[i])+ 1.0 - o.generator(z[i], u)) % 1.0
        ρ[i] = o.invcdf_mom(ξ)
    end
    return u
end



# inplace fwd/bwd flow 
function flow_fwd!(o::HamFlow, ϵ::Vector{Float64}, refresh!::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        leapfrog!(o, ϵ, z, ρ)
        u = refresh!(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return u 
end

function flow_bwd!(o::HamFlow, ϵ::Vector{Float64}, inv_ref!::Function, z, ρ, u, n_mcmc::Int) 
    for i in 1:n_mcmc - 1 
        u = inv_ref!(o, z, ρ, u)
        leapfrog!(o, -ϵ, z, ρ)
    end
    return u
end



############################
# density estimation
############################
function log_density_est(z, ρ, u, o::HamFlow, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; 
                        nBurn::Int = 0)
    # density estimation 
    logJ = 0.0
    T = Buffer(zeros(n_mcmc))
    T[1] = o.lpdf_mom(ρ) + o.logq0(z, μ, D)
    for i in 1:n_mcmc-1
        logJ += o.lpdf_mom(ρ)
        ρ, u = inv_ref(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
        logJ -= o.lpdf_mom(ρ)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
        T[i + 1] = o.logq0(z,μ,D) + o.lpdf_mom(ρ) + logJ
    end
    lpdfs = copy(T)
    logqN = logmeanexp(@view(lpdfs[nBurn+1:end]))
    return logqN, logJ
end

function log_density_update(o::HamFlow, ϵ, μ, D, lpdf::Real, z, ρ, u, logJ_prod, N::Int, inv_ref::Function)
    # extend backwards by 1, and update jacobian prod
    logJ_prod += o.lpdf_mom(ρ)
    ρ, u_new = inv_ref(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
    logJ_prod -= o.lpdf_mom(ρ)
    z_new, ρ_new = leapfrog(o, -ϵ, z, ρ)

    # compute new mix component
    logq = o.logq0(z_new,μ,D) + o.lpdf_mom(ρ_new) + logJ
    # update log density    
    lpdf_new = logsumexp([lpdf+ log(N-1), logq]) - log(N)
    return lpdf_new, z_new, ρ_new, u_new, logJ_prod
end

# compute the evolution of density
function log_density_stream(z, ρ, u, o::HamFlow, a::HF_params, inv_ref::Function, n_mcmc::Int)
    logJ = 0.0
    logqs = zeros(n_mcmc)
    lpdf =  o.lpdf_mom(ρ) + o.logq0(z, a.μ, a.D)
    logqs[1] = lpdf
    for i in 1:n_mcmc - 1
        lpdf, z, ρ, u = log_density_update(o, a.leapfrog_stepsize, a.μ, a.D, lpdf, z, ρ, u, logJ, i+1, inv_ref)
        logqs[i+1] = lpdf
    end
    return logqs
end


############################
### single_elbo estimate 
############################
function single_elbo_naive(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
    d = o.d
    K = rand(nBurn+1:n_mcmc) # Num T - 1
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    z1, ρ1, u1 = copy(z), copy(ρ), copy(u) 
    # using Buffer for Zygote autograd
    T = Buffer(zeros(n_mcmc))
    logJ = Buffer(zeros(n_mcmc))
    # T = Vector{Float65}(undef,n_mcmc) 
    # logJ = Vector{Float65}(undef, n_mcmc) 

    T[K] = o.logq0(z1, μ, D) + o.lpdf_mom(ρ1)
    logJ[1] = 0.0
    # flow forward
    @inbounds for i in 1:K-1 
        z1, ρ1 = leapfrog(o, ϵ, z1, ρ1)
        logJ[K-i+1] = -o.lpdf_mom(ρ1)
        ρ1, u1 = refresh(o, z1, ρ1, u1)
        logJ[K-i+1] += o.lpdf_mom(ρ1)
        T[K-i] = o.logq0(z1, μ, D) + o.lpdf_mom(ρ1)
    end
    el = o.logp(z1) + o.lpdf_mom(ρ1)
    # flow backward
    @inbounds for i in K+1:n_mcmc 
        logJ[i] = o.lpdf_mom(ρ)
        ρ, u = inv_ref(o, z, ρ, u)
        logJ[i] -= o.lpdf_mom(ρ)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
        T[i] = o.logq0(z, μ, D) + o.lpdf_mom(ρ)
    end
    # ! compute lpdf for qk
    lpdfs = copy(T) .+ cumsum(copy(logJ))
    # T .+= cumsum(logJ)
    logqN =  logmeanexp(@view(lpdfs[nBurn+1:end]))
    return el - logqN
end

# function single_elbo_naive(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
#     d = o.d
#     K = rand(nBurn+1:n_mcmc) # Num T - 1
#     z = D .* o.q_sampler(d) .+ μ
#     ρ, u = o.ρ_sampler(d), rand()
#     # flow forward
#     z, ρ, u = flow_fwd(o, ϵ, refresh, z, ρ, u, K)
#     el = o.logp(z) + o.lpdf_mom(ρ)
#     # density estimation 
#     logqN, _ = log_density_est(z, ρ, u, o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn)
#     return el-logqN
# end

function single_elbo_long(o::HamFlow, ϵ::Vector{Float64}, μ::Vector{Float64}, D::Vector{Float64}, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
    # init sample
    d = o.d
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    z0, ρ0, u0 = copy(z), copy(ρ), copy(u) 
 
    # save logjs, logq0,logqn
    logjs = zeros(2*n_mcmc-2)    
    logq0s = zeros(2*n_mcmc-1)
    logqns = zeros(n_mcmc)    

    # flow bwd n-1 step 
    @inbounds for i in n_mcmc-1:-1:1
        logjs[i] = o.lpdf_mom(ρ0)
        ρ0, u0 = inv_ref(o, z0, ρ0, u0)
        logjs[i] -= o.lpdf_mom(ρ0)
        z0, ρ0 = leapfrog(o, -ϵ, z0, ρ0)
        logq0s[i] = o.logq0(z0, μ, D) + o.lpdf_mom(ρ0)
    end
    logq0s[n_mcmc] = o.lpdf_mom(ρ) + o.logq0(z, μ, D)
    logp = o.logp(z) + o.lpdf_mom(ρ) 
    logqns[1] = logmeanexp(@view(logq0s[n_mcmc:-1:1]) .+ cumsum(vcat([0.0], @view(logjs[n_mcmc-1:-1:1]))))

    # flow fwd n-1 step
    @inbounds for i in n_mcmc:2*n_mcmc - 2 
        z, ρ = leapfrog(o, ϵ, z, ρ)
        logjs[i] = -o.lpdf_mom(ρ)
        ρ, u = refresh(o, z, ρ, u)
        logjs[i] += o.lpdf_mom(ρ)
        logq0 = o.logq0(z, μ, D) + o.lpdf_mom(ρ)
        logq0s[i+1] = logq0
        # update logqn(t^n x)
        logqns[i-n_mcmc+2] = logmeanexp(@view(logq0s[i+1:-1:i-n_mcmc+2]) .+ cumsum(vcat([0.0], @view(logjs[i:-1:i-n_mcmc+2]))))
        logp += o.logp(z) + o.lpdf_mom(ρ) 
    end
    logp /= n_mcmc
    return logp - mean(logqns)
end


function DensityTripleSave(z0::Vector{Float64}, ρ0::Vector{Float64}, u0::Float64, o::HamFlow, ϵ::Vector{Float64}, μ::Vector{Float64}, D::Vector{Float64}, inv_ref::Function, n_mcmc::Int)
    # save t^k(x), logjs, logqn
    logjs = zeros(n_mcmc-1)    
    logq0s = zeros(n_mcmc)
    
    # logq0(x0)
    logq0s[n_mcmc] = o.lpdf_mom(ρ0) + o.logq0(z0, μ, D)
    # flow bwd n-1 step 
    @inbounds for i in n_mcmc-1:-1:1
        logjs[i] = o.lpdf_mom(ρ0)
        ρ0, u0 = inv_ref(o, z0, ρ0, u0)
        logjs[i] -= o.lpdf_mom(ρ0)
        z0, ρ0 = leapfrog(o, -ϵ, z0, ρ0)
        logq0s[i] = o.logq0(z0, μ, D) + o.lpdf_mom(ρ0)
    end
    # logqn(x0)
    logqn = logmeanexp(@view(logq0s[n_mcmc:-1:1]) .+ cumsum(vcat([0.0], @view(logjs[n_mcmc-1:-1:1]))))
    # jacobian prod
    logj_prod = sum(@view(logjs[1:n_mcmc-1]))
    
    return logqn, logj_prod, logq0s, logjs
end

function single_elbo_fast(o::HamFlow, ϵ::Vector{Float64}, μ::Vector{Float64}, D::Vector{Float64}, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
    # init sample
    d = o.d
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    z0, ρ0, u0 = copy(z), copy(ρ), copy(u) 
 
    # save t^k(x), logjs, logqn
    logqns = zeros(n_mcmc)    
    # flow bwd n-1 step 
    logqn, logj_prod, logq0s, logjs = DensityTripleSave(z0, ρ0, u0, o, ϵ, μ, D, inv_ref, n_mcmc)
    logqns[1] = logqn
    logp = o.logp(z) + o.lpdf_mom(ρ) 

    # flow fwd n-1 step
    @inbounds for i in 1:n_mcmc-1
        z, ρ = leapfrog(o, ϵ, z, ρ)
        logJ = -o.lpdf_mom(ρ)
        ρ, u = refresh(o, z, ρ, u)
        logJ += o.lpdf_mom(ρ)
        logq0 = o.logq0(z, μ, D) + o.lpdf_mom(ρ)
        # update logqn(T^n x)
        l = logq0s[i] + logj_prod - log(n_mcmc)
        el = log(expm1(logqns[i]) - expm1(l)) + logJ
        logqns[1+i] = logsumexp([el, logq0 - log(n_mcmc)])
        # udpate jacobian prod
        logj_prod += logJ - logjs[i]
        # update logp
        logp += o.logp(z) + o.lpdf_mom(ρ) 
    end
    logp /= n_mcmc
    return logp - mean(logqns) 
end

function DensityTriple(z0, ρ0, u0, o::HamFlow, ϵ::Vector{Float64}, μ::Vector{Float64}, D::Vector{Float64}, inv_ref::Function, n_mcmc::Int)
    logJ = 0.0     
    # logq0(x0)
    logqn = o.lpdf_mom(ρ0) + o.logq0(z0, μ, D)
    # flow bwd n-1 step 
    for i in n_mcmc-1:-1:1
        logJ += o.lpdf_mom(ρ0)
        ρ0, u0 = inv_ref(o, z0, ρ0, u0)
        logJ -= o.lpdf_mom(ρ0)
        z0, ρ0 = leapfrog(o, -ϵ, z0, ρ0)
        logqn = logsumexp([logqn, o.logq0(z0, μ, D) + o.lpdf_mom(ρ0) + logJ])
    end
    logqn -= log(n_mcmc) 
    return logqn, logJ, z0, ρ0, u0 
end

function single_elbo_eff(o::HamFlow, ϵ::Vector{Float64}, μ::Vector{Float64}, D::Vector{Float64}, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
    # init sample
    d = o.d
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    z0, ρ0, u0 = copy(z), copy(ρ), copy(u) 
    
    logqn, logj_prod, z0, ρ0, u0 =  DensityTriple(z0, ρ0, u0, o, ϵ, μ, D, inv_ref, n_mcmc)
    f = o.logp(z) + o.lpdf_mom(ρ)
    g = logqn

    # flow fwd n-1 step
    @inbounds for i in 1:n_mcmc-1
        logqb = o.logq0(z0, μ, D) + o.lpdf_mom(ρ0) + logj_prod - log(n_mcmc)
        z, ρ = leapfrog(o, ϵ, z, ρ)
        logJ = -o.lpdf_mom(ρ)
        ρ, u = refresh(o, z, ρ, u)
        logJ += o.lpdf_mom(ρ)
        logqn = log(expm1(logqn) - expm1(logqb)) + logJ
        logq0 = o.logq0(z, μ, D) + o.lpdf_mom(ρ)
        logqn = logsumexp([logqn, logq0 - log(n_mcmc)])
        # update logp, logqn
        f += o.logp(z) + o.lpdf_mom(ρ) 
        g += logqn
        # update x' and get jacobian
        z0, ρ0 = leapfrog(o, ϵ, z0, ρ0)
        logJ1 = -o.lpdf_mom(ρ0)
        ρ0, u0 = refresh(o, z0, ρ0, u0)
        logJ1 += o.lpdf_mom(ρ0)
        # udpate jacobian prod
        logj_prod += logJ - logJ1
    end
    return (f - g)/n_mcmc
end
