using Base.Threads: @threads
using ProgressMeter, Flux
using Zygote:Buffer

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
    buf = Buffer(ρ)
    for i in 1:o.d 
        ξ = (o.cdf_mom(ρ[i]) + o.generator(z[i], u)) % 1.0
        buf[i] = o.invcdf_mom(ξ)
        u = o.time_shift(u)
    end
    return copy(buf), u 
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
    buf = Buffer(ρ)
    for i in o.d:-1:1
        u = o.inv_timeshift(u)
        ξ = (o.cdf_mom(ρ[i])+ 1.0 - o.generator(z[i], u)) % 1.0
        buf[i] = o.invcdf_mom(ξ)
    end
    return copy(buf), u
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



function log_density_est(z, ρ, u, o::HamFlow, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; 
                        nBurn::Int = 0, error_check = false)
    if error_check 
        err = error_checking(o, ϵ, refresh, inv_ref, z, ρ, u, n_mcmc)
    else 
        err = 10000.0
    end
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
    return logqN, err
end



### single_elbo estimate 
function single_elbo(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0)
    d = o.d
    K = rand(nBurn+1:n_mcmc) # Num T - 1
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    z1, ρ1, u1 = copy(z), copy(ρ), copy(u) 
    # !
    T = Buffer(zeros(n_mcmc))
    logJ = Buffer(zeros(n_mcmc))
    # T = Vector{Float65}(undef,n_mcmc) 
    # logJ = Vector{Float65}(undef, n_mcmc) 

    T[K] = o.logq0(z1, μ, D) + o.lpdf_mom(ρ1)
    logJ[1] = 0.0
    # flow forward
    for i in 1:K-1 
        z1, ρ1 = leapfrog(o, ϵ, z1, ρ1)
        logJ[K-i+1] = -o.lpdf_mom(ρ1)
        ρ1, u1 = refresh(o, z1, ρ1, u1)
        logJ[K-i+1] += o.lpdf_mom(ρ1)
        T[K-i] = o.logq0(z1, μ, D) + o.lpdf_mom(ρ1)
    end
    el = o.logp(z1) + o.lpdf_mom(ρ1)
    # flow backward
    for i in K+1:n_mcmc 
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
