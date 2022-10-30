############### 
# specialized HamFlow for 1d target
###############

##### struct of ergflow 
struct HamFlow_1d <: ErgodicFlow
    # HMC struct (we dont put n_mcmc inside as we need to change transition steps in sampler)
    n_lfrg::Int64 # number of n_lfrg between refresh
    # sampling and likelihood functions
    logq0::Function
    q_sampler::Function
    ∇logp::Function
    # momentum 
    ρ_sampler::Function
    cdf_mom::Function 
    invcdf_mom::Function
    lpdf_mom::Function
    ∇logp_mom::Function
    # pseudo refreshment
    generator::Function
    time_shift::Function
    inv_timeshift::Function
end

struct HF1d_params <: flow_params
    leapfrog_stepsize::Float64
    # VI_params::Params
end

function leapfrog(o::HamFlow_1d, a::HF1d_params, z, ρ)
    ϵ = a.leapfrog_stepsize
    ρ += 0.5 * ϵ * o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z -= ϵ * o.∇logp_mom(ρ)
        ρ += ϵ * o.∇logp(z) 
    end
    z -= ϵ * o.∇logp_mom(ρ)
    ρ += 0.5 * ϵ * o.∇logp(z) 
    return z,ρ
end


function inv_leapfrog(o::HamFlow_1d, a::HF1d_params, z, ρ)
    ϵ = -1.0*a.leapfrog_stepsize
    ρ += 0.5 * ϵ * o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z -= ϵ * o.∇logp_mom(ρ)
        ρ += ϵ * o.∇logp(z) 
    end
    z -= ϵ * o.∇logp_mom(ρ)
    ρ += 0.5 * ϵ * o.∇logp(z) 
    return z,ρ
end

# refreshment  
function pseudo_refresh(o::HamFlow_1d, z, ρ, u)
    u1 = o.time_shift(u)
    ξ = (o.cdf_mom(ρ) + o.generator(z, u1)) % 1.0
    ρ1 = o.invcdf_mom(ξ)
    return ρ1, u1 
end

# inv refreshment 
function inv_refresh(o::HamFlow_1d, z, ρ, u)
    ξ = (o.cdf_mom(ρ) + 1.0 - o.generator(z, u)) % 1.0 
    ρ0 = o.invcdf_mom(ξ) 
    u0 = o.inv_timeshift(u)
    return ρ0, u0
end


# fwd flow 
function flow_fwd(o::HamFlow_1d, a::HF1d_params, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, a, z, ρ)
        ρ,u = pseudo_refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return z, ρ, u
end

function flow_fwd_track(o::HamFlow_1d, a::HF1d_params, z, ρ, u, n_mcmc::Int)
    T = zeros(n_mcmc, 2)
    T[1,:] .= [z, ρ]
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, a, z, ρ)
        ρ, u = pseudo_refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
        T[i + 1, :] .= [z, ρ]
        ProgressMeter.next!(prog_bar)
    end
    return z, ρ, u, T
end

function Sampler(o::HamFlow_1d, a::HF1d_params, n_mcmc, N; nBurn::Int64 = 0)
    T = zeros(N, 2)
    prog_bar = ProgressMeter.Progress(N, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        # Sample(Unif{1, ..., n_mcmc})
        n_step = rand(nBurn+1:n_mcmc) 
        z0, ρ0, u0 = o.q_sampler(), o.ρ_sampler(), rand()
        z, ρ, _ = flow_fwd(o,a, z0, ρ0, u0, n_step)
        T[i, :] .= [z, ρ]
        ProgressMeter.next!(prog_bar)
    end
    return T
end

# bwd flow 
function flow_bwd(o::HamFlow_1d, a::HF1d_params, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc-1
        ρ, u = inv_refresh(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
        z, ρ = inv_leapfrog(o, a, z, ρ)
    end
    return z, ρ, u
end

function flow_bwd_save(o::HamFlow_1d, a::HF1d_params, z, ρ, u, n_mcmc::Int)
    lpdfs = Vector{Float64}(undef, n_mcmc)
    lpdfs[1] = o.logq0(z) + o.lpdf_mom(ρ)
    logJ = 0.0
    for i in 1:n_mcmc-1
        logJ += o.lpdf_mom(ρ)
        ρ, u = inv_refresh(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
        logJ -= o.lpdf_mom(ρ)
        z, ρ = inv_leapfrog(o, a, z, ρ)

        # println(i+1, "/$n_mcmc")
        lpdfs[i + 1] = o.logq0(z) + o.lpdf_mom(ρ) + logJ
    end
    return z, ρ, u, lpdfs
end


# density_evaluation 
function density_evaluation(o::HamFlow_1d, a::HF1d_params, X::Vector{Float64}, Y::Vector{Float64}, n_mcmc::Int; nBurn::Int = 0)
    n1, n2 = size(X, 1), size(Y, 1)
    T = zeros(n1, n2, n_mcmc)
    # A = zeros(n_mcmc)
    prog_bar = ProgressMeter.Progress(n1*n2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i = 1:n1
        # println("$i / $n1")
        for j=1:n2 
            # this step is bit hacky since we do not use u
            T[i, j, :] .= flow_bwd_save(o, a, X[i], Y[j], 1.0, n_mcmc)[end]
            # T[i, j] = logmeanexp(@view(A[nBurn+1:end])) 
            ProgressMeter.next!(prog_bar)
        end
    end
    return logmeanexp_slice(T[:, :, nBurn+1:end]; dims = 3)[:, :,1]
end

# check numerical error
function single_error_checking(o::HamFlow_1d, a::HF1d_params, z, ρ, u, n_mcmc::Int)
    z0, ρ0, u0 = flow_bwd(o, a, z, ρ, u, n_mcmc)
    z1, ρ1, u1 = flow_fwd(o, a, z0, ρ0, u0, n_mcmc)
    err = abs(z1 - z) + abs(ρ1 - ρ) + abs(u1 - u)
    return err
end

function numerical_error_checking(o::HamFlow_1d, a::HF1d_params, X::Vector{Float64}, Y::Vector{Float64}, n_mcmc::Int)
    n1, n2 = size(X, 1), size(Y, 1)
    T = zeros(n1, n2)
    prog_bar = ProgressMeter.Progress(n1*n2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i = 1:n1
        # println("$i / $n1")
        for j=1:n2 
            #  we do not use u
            T[i, j] = single_error_checking(o,a, X[i], Y[j], 0.5, n_mcmc)
            ProgressMeter.next!(prog_bar)
        end
    end
    return T
end
