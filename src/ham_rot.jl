using LinearAlgebra, Distributions, Random, Plots, StatsBase, SpecialFunctions, Parameters
using Base.Threads: @threads
using ProgressMeter, Flux
using Zygote:Buffer

##### struct of ergflow 
struct HamFlowRot<: ErgodicFlow
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
    lpdf_mom_norm::Function
    lpdf_mom::Function
    ∇logp_mom::Function
    cdf_mom::Function 
    invcdf_mom::Function
    pdf_mom::Function
    # pseudo refreshment
    generator::Function
    rotation::Matrix{Float64}
    inv_rotation::Matrix{Float64}
    time_shift::Function
    inv_timeshift::Function
end


struct HF_params <: flow_params
    leapfrog_stepsize::Vector{Float64}  
    μ::Vector{Float64}
    D::Vector{Float64}
end



####################
# first 2d version
###################
function pseudo_refresh(o::HamFlowRot, z, ρ, u)
    R = norm(ρ)
    ξ = (o.cdf_mom(R^2.0) + o.generator(z[1], u)) % 1.0
    R_new = sqrt(o.invcdf_mom(ξ))
    ρ_new = o.rotation * ρ ./R .*R_new 
    u = o.time_shift(u)
    return ρ_new, u 
end

function inv_refresh(o::HamFlowRot,z, ρ, u)
    u = o.inv_timeshift(u)
    R = norm(ρ)
    ξ = (o.cdf_mom(R^2.0)+ 1.0 - o.generator(z[1], u)) % 1.0
    R_old = sqrt(o.invcdf_mom(ξ))
    ρ_old = o.inv_rotation * ρ ./R .*R_old
    return ρ_old, u
end


function flow_fwd(o::HamFlowRot, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ, u = refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return z, ρ, u
end

function flow_bwd(o::HamFlowRot, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        ρ, u = inv_ref(o, z, ρ, u)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
    end
    return z, ρ, u
end


function log_density_est(z, ρ, u, o::HamFlowRot, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; 
                    nBurn::Int = 0, error_check = false, refresh!::Function = pseudo_refresh!, inv_ref!::Function = inv_refresh!)
    if error_check 
        err = error_checking(o, ϵ, refresh!, inv_ref!, z, ρ, u, n_mcmc)
    else 
        err = 10000.0
    end
    # density estimation 
    logJ = 0.0
    T = Buffer(zeros(n_mcmc))
    T[1] = o.lpdf_mom(ρ) + o.logq0(z, μ, D)
    for i in 1:n_mcmc-1
        logJ += o.lpdf_mom_norm(sum(abs2, ρ))
        ρ, u = inv_ref(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
        logJ -= o.lpdf_mom_norm(sum(abs2, ρ))
        z, ρ = leapfrog(o, -ϵ, z, ρ)
        T[i + 1] = o.logq0(z,μ,D) + o.lpdf_mom(ρ) + logJ
    end
    lpdfs = copy(T)
    logqN = logmeanexp(@view(lpdfs[nBurn+1:end]))
    return logqN, err
end

function single_elbo(o::HamFlowRot, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0, error_check = false)
    d = o.d
    K = rand(nBurn+1:n_mcmc) # Num T - 1
    z = D .* o.q_sampler(d) .+ μ
    ρ, u = o.ρ_sampler(d), rand()
    # flow forward
    z, ρ, u = flow_fwd(o, ϵ, refresh, z, ρ, u, K)
    el = o.logp(z) + o.lpdf_mom(ρ)
    # density estimation 
    logqN, _ = log_density_est(z, ρ, u, o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn, error_check = error_check)
    return el-logqN
end


function ELBO(o::HamFlowRot, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
    if print
        prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    @threads for i in 1:elbo_size
        el_single = single_elbo(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
        atomic_add!(el, 1/elbo_size * el_single)
        ProgressMeter.next!(prog_bar)
    end
    return el[]
end



function Sampler(o::HamFlowRot, a::HF_params, refresh::Function, n_mcmc::Int, N::Int; nBurn::Int64 = 0)
    d = o.d
    T = Matrix{Float64}(undef, N, d)
    M = Matrix{Float64}(undef, d, N)
    U = Vector{Float64}(undef, N)
    @info "ErgFlow Sampling"
    prog_bar = ProgressMeter.Progress(N, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        # Sample(Unif{1, ..., n_mcmc})
        n_step = rand(nBurn+1:n_mcmc) 
        z0 = a.D .* o.q_sampler(d) .+ a.μ
        ρ0, u0 = o.ρ_sampler(d), rand()
        z, ρ, u = flow_fwd(o, a.leapfrog_stepsize, refresh, z0, ρ0, u0, n_step)
        T[i, :] .= z 
        M[:, i] .= ρ
        U[i] = u
        ProgressMeter.next!(prog_bar)
    end
    return T, M, U
end

function log_density_slice_2d(X, Y, ρ, u, o::HamFlowRot, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; nBurn = 0, error_check = true) 
        n1, n2 = size(X, 1), size(Y, 1)
        T = Matrix{Float64}(undef, n1, n2)
        E = Matrix{Float64}(undef, n1, n2)
        prog_bar = ProgressMeter.Progress(n1*n2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        @threads for i = 1:n1
        # println("$i / $n1")
        for j=1:n2 
            # this step is bit hacky since we do not use u
            T[i, j], E[i, j] = ErgFlow.log_density_est([X[i],Y[j]], ρ, u, o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn, error_check = error_check)
            # T[i, j] = logmeanexp(@view(A[nBurn+1:end])) 
            ProgressMeter.next!(prog_bar)
        end
    end
    return T, E 
end