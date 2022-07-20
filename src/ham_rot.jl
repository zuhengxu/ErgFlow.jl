using Flux
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


####################
# not finished !!!
# Only implemented for 2d case 
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


#######################
# density estimation
#######################

function log_density_est(z, ρ, u, o::HamFlowRot, ϵ, μ, D, inv_ref::Function, n_mcmc::Int;  
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
        logJ += o.lpdf_mom_norm(sum(abs2, ρ)) # Jacobian update
        ρ, u = inv_ref(o, z, ρ, u) # ρ_k -> ρ_(k-1/2)
        logJ -= o.lpdf_mom_norm(sum(abs2, ρ)) # Jacobian update
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


