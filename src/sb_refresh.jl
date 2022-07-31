using Base.Threads: @threads


# we send a batch of a sample and estimate the standarization 
# so far this only works for Gaussian
struct SB_refresh <: ErgodicFlow
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
    pdf_mom::Function
    # estimated standarization 
    r_state::Vector{Any}
end

function oneleapfrog!(∇logp::Function, ∇logm::Function, ϵ::Vector{Float64}, z, ρ)
    ρ .+= 0.5 .* ϵ .* ∇logp(z) 
    z .-= ϵ .* ∇logm(ρ)
    ρ .+= 0.5 .* ϵ .* ∇logp(z) 
end

function compute_transformation(Σp)
    d = size(Σp,1)
    Σp = 0.5 .* (Σp + transpose(Σp))
    chol = cholesky(Σp)
    L = chol.L
    L_inv = L \ I(d)
    return L, L_inv
end

function batch_refresh!(μp::Vector{Float64}, L_inv, ps_ref::Matrix{Float64})
    s = size(ps_ref,1)
    @threads for i in 1:s
        @view(ps_ref[i,:]) .= L_inv * (@view(ps_ref[i,:]) .- μp)
    end
end

function refresh_sb(o::SB_refresh, i::Int64, ρ) 
    state = o.r_state[i]
    μ, L, L_inv = get!(state, "key", 0)
    ρ1 = L_inv * (ρ .- μ)
    return ρ1
end

function inv_refresh_sb(o::SB_refresh, i::Int64, ρ) 
    state = o.r_state[i]
    μ, L, L_inv = get!(state, "key", 0)
    ρ0 =  L* ρ .+ μ
    return ρ0
end

function refresh_error_sb(o::SB_refresh, refresh::Function, inv_ref::Function, i::Int64, ρ)
    ρ1 = refresh(o, i, ρ)
    ρ0 = inv_ref(o, i, ρ1)
    return norm(ρ0 .- ρ)
end 

function one_step_error_sb(o::SB_refresh,  refresh::Function, inv_ref::Function,ϵ::Vector{Float64}, i::Int64, z, ρ)
    z0, ρ0 = copy(z), copy(ρ)
    z, ρ = leapfrog(o, ϵ, z, ρ)
    ρ = refresh(o, i, ρ)
    ρ = inv_ref(o, i, ρ)
    z, ρ = leapfrog(o, -ϵ, z, ρ)
    error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) 
    return sqrt(error)
end

function flow_fwd(o::SB_refresh, ϵ::Vector{Float64}, z, ρ, n_mcmc::Int64)
    for i in 1: n_mcmc- 1
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ = refresh_sb(o, i, ρ)
    end
    return z, ρ
end

function flow_bwd(o::SB_refresh, ϵ::Vector{Float64}, z, ρ, n_mcmc::Int64)
    for i in 1: n_mcmc- 1
        ρ = inv_refresh_sb(o, n_mcmc - i, ρ)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
    end
    return z, ρ
end

# estimating standarization matrix
function warm_start(∇logp::Function, ∇logp_mom::Function, a::HF_params, d::Int64, sample_size::Int64, n_ref::Int64, n_lfrg::Int64)
        # samples used to estimate refresh parameters
        zs_ref = randn(sample_size, d) .* a.D' .+ a.μ' 
        ps_ref = randn(sample_size, d)

        # prepare dictionaries holding parameters
        r_states = []
        for i in 1:n_ref - 1
            push!(r_states, IdDict())
        end

        prog_bar = ProgressMeter.Progress(n_lfrg*(n_ref-1), dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        for n in 1:n_lfrg*(n_ref-1)
            @threads for i in 1:sample_size
                oneleapfrog!(∇logp, ∇logp_mom, a.leapfrog_stepsize, @view(zs_ref[i,:]), @view(ps_ref[i,:]))
            end
            if n % n_lfrg == 0
                μp = vec(mean(ps_ref, dims=1))
                Σp = cov(ps_ref, ps_ref)
                L, L_inv = compute_transformation(Σp)
                get!(r_states[Int(floor(n / n_lfrg))], "key")do
                    (μp, L, L_inv)
                end
                batch_refresh!(μp, L_inv, ps_ref)
            end
            ProgressMeter.next!(prog_bar)
        end
    return r_states
end

################3
# getting fwd/bwd trjectory
###################
function flow_fwd_trace(o::SB_refresh, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(undef, 2*(n_mcmc-1)+1, o.d)
    M = Matrix{eltype(ρ)}(undef, 2*(n_mcmc-1)+1, o.d)
    U = Vector{typeof(u)}(undef, 2*(n_mcmc-1)+1)
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:n_mcmc - 1 
        k = 2*(i-1)
        T[k+1, :] .= z 
        M[k+1,:] .= ρ
        U[k+1] = u
        z, ρ = leapfrog(o, ϵ, z, ρ)
        T[k+2,:] .= z
        M[k+2,:] .= ρ
        U[k+2] = u
        ρ = refresh(o, i, ρ)
        # println(i, "/$n_mcmc")
        ProgressMeter.next!(prog_bar)
    end
    T[end,:] .= z
    M[end,:] .= ρ
    U[end] = u
    return T, M, U
end

function flow_bwd_trace(o::SB_refresh, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(undef, 2*(n_mcmc-1)+1, o.d)
    M = Matrix{eltype(ρ)}(undef, 2*(n_mcmc-1)+1, o.d)
    U = Vector{typeof(u)}(undef, 2*(n_mcmc-1)+1)
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    T[1,:] .= z
    M[1,:] .= ρ
    U[1] = u
    for i in 1:n_mcmc - 1 
        ρ = inv_ref(o, n_mcmc - i, ρ)
        k = 2*(i-1)
        T[k+2, :] .= z 
        M[k+2,:] .= ρ
        U[k+2] = u
        # println(i, "/$n_mcmc")
        z, ρ = leapfrog(o, -ϵ, z, ρ)
        T[k+3,:] .= z
        M[k+3,:] .= ρ
        U[k+3] = u
        ProgressMeter.next!(prog_bar)
    end
    # return T[end:-1:1, :], M[end:-1:1, :], U[end:-1:1]
    return T, M, U
end

function flow_trace(o::SB_refresh, a::HF_params, refresh::Function, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    T_fwd, M_fwd, U_fwd = flow_fwd_trace(o, a.leapfrog_stepsize, refresh, z,ρ, u, n_mcmc)
    T_bwd, M_bwd, U_bwd = flow_bwd_trace(o, a.leapfrog_stepsize, inv_ref, T_fwd[end, :],M_fwd[end, :], U_fwd[end], n_mcmc)
    return T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd 
end

################
# error analysis
#################
function error_checking(o::SB_refresh, ϵ::Vector{Float64}, z0, ρ0, n_mcmc::Int)
```
T = ergflow
compute "||(z0, ρ0, u0) - T^{-1}∘T(z0, ρ0, u0)|| for given (z0, ρ0, u0) "
```
    # fwd and bwd flow
    z1, ρ1 = flow_fwd(o, ϵ, z0, ρ0, n_mcmc)
    z, ρ = flow_bwd(o, ϵ, z1, ρ1, n_mcmc)
    # compute err
    error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) 
    return sqrt(error)
end
function flow_fwd_err_tr_sb(o::SB_refresh, ϵ::Vector{Float64}, z, ρ, u)
    n_mcmc = size(o.r_state, 1) + 1
    leap_err = Vector{typeof(u)}(undef, n_mcmc)
    ref_err = Vector{typeof(u)}(undef, n_mcmc)
    total_err = Vector{typeof(u)}(undef,n_mcmc)
    leap_err[1,:] .= 0
    ref_err[1,:] .= 0
    total_err[1] = 0
    # prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    E = error_checking(o, ϵ, z, ρ, n_mcmc)
    for i in 1:n_mcmc - 1 
        # computing err
        leap_err[i+1] = leapfrog_error(o, ϵ, z, ρ)
        ref_err[i+1] = refresh_error_sb(o, refresh_sb, inv_refresh_sb,i, ρ)
        total_err[i+1] = one_step_error_sb(o, refresh_sb, inv_refresh_sb, ϵ, i, z, ρ)
        # perform fwd flow
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ = refresh_sb(o, i, ρ)
        # ProgressMeter.next!(prog_bar)
    end
    return leap_err, ref_err, total_err, E
end
