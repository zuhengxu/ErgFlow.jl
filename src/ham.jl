using LinearAlgebra, Distributions, Random, Plots, StatsBase, SpecialFunctions, Parameters
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



# leapfrog
function leapfrog(o::Union{HamFlow,HamFlowRot}, ϵ, z, ρ)
    ρ += 0.5 .* ϵ .* o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z -= ϵ .* o.∇logp_mom(ρ)
        ρ += ϵ .* o.∇logp(z) 
    end

    z -= ϵ .* o.∇logp_mom(ρ)
    ρ += 0.5 .* ϵ .* o.∇logp(z) 
    return z, ρ
end

function leapfrog!(o::HamFlow, ϵ, z, ρ)
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z .-= ϵ .* o.∇logp_mom(ρ)
        ρ .+= ϵ .* o.∇logp(z) 
    end
    z .-= ϵ .* o.∇logp_mom(ρ)
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
end

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
# inv refreshment
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

# flow fwd
function flow_fwd(o::HamFlow, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ, u = refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return z, ρ, u
end

function flow_fwd!(o::HamFlow, ϵ::Vector{Float64}, refresh!::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        leapfrog!(o, ϵ, z, ρ)
        u = refresh!(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return u 
end

# for tuning purpose
function flow_fwd_trace(o::HamFlow, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(size(z, 1), n_mcmc)
    M = Matrix{eltype(ρ)}(size(ρ, 1), n_mcmc)
    U = Vector{typeof(u)}(n_mcmc)
    T[1,:] .= z
    M[1,:] .= ρ
    U[1] = u
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ, u = refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
        T[i+1,:] .= z
        M[i+1,:] .= ρ
        U[i+1] = u
        ProgressMeter.next!(prog_bar)
    end
    return T, M, U
end

# backward flow
function flow_bwd(o::HamFlow, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int) 
    for i in 1:n_mcmc - 1 
        ρ, u = inv_ref(o, z, ρ, u)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
    end
    return z, ρ, u
end

function flow_bwd!(o::HamFlow, ϵ::Vector{Float64}, inv_ref!::Function, z, ρ, u, n_mcmc::Int) 
    for i in 1:n_mcmc - 1 
        u = inv_ref!(o, z, ρ, u)
        leapfrog!(o, -ϵ, z, ρ)
    end
    return u
end

function flow_bwd_trace(o::HamFlow, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(size(z, 1), n_mcmc)
    M = Matrix{eltype(ρ)}(size(ρ, 1), n_mcmc)   
    U = Vector{typeof(u)}(n_mcmc)
    T[1,:] .= z
    M[1,:] .= ρ
    U[1] = u
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:n_mcmc - 1 
        ρ, u = inv_ref(o, z, ρ, u)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
        # println(i, "/$n_mcmc")
        T[i+1,:] .= z
        M[i+1,:] .= ρ
        U[i+1] = u
        ProgressMeter.next!(prog_bar)
    end
    return T, M, U
end

function error_checking(o::HamFlow, ϵ::Vector{Float64}, refresh!::Function, inv_ref!::Function, z, ρ, u, n_mcmc::Int)
    z0, ρ0, u0 = copy(z), copy(ρ), copy(u)
    u = flow_fwd!(o, ϵ, refresh!, z, ρ, u, n_mcmc)
    u = flow_bwd!(o, ϵ, inv_ref!, z, ρ, u, n_mcmc)
    return sum(abs, z .- z0) + sum(abs, ρ.-ρ0) + abs(u - u0)
end




function log_density_est(z, ρ, u, o::HamFlow, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; 
                    nBurn::Int = 0, error_check = false, refresh!::Function = pseudo_refresh_coord!, inv_ref!::Function = inv_refresh_coord!)
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


function log_density_slice_2d(X, Y, ρ, u, o::ErgFlow.HamFlow, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; nBurn = 0, error_check = true) 
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
### ELBO 
# function single_elbo(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; nBurn::Int = 0, error_check = false)
#     d = o.d
#     K = rand(nBurn+1:n_mcmc) # Num T - 1
#     z = D .* o.q_sampler(d) .+ μ
#     ρ, u = o.ρ_sampler(d), rand()
#     # flow forward
#     z, ρ, u = flow_fwd(o, ϵ, refresh, z, ρ, u, K)
#     el = o.logp(z) + o.lpdf_mom(ρ)
#     # density estimation 
#     logqN, _ = log_density_est(z, ρ, u, o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn, error_check = error_check)
#     return el-logqN
# end

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


function ELBO(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
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
# same ELBO function but without @threads---cannot use multithreads inside Zygote
function ELBO_opt(o::HamFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    el = 0.0    
    for i in 1:elbo_size
        el_single = single_elbo(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
        el += 1/elbo_size * el_single
    end
    return el
end
function HamErgFlow(o::HamFlow, a::HF_params, refresh::Function, inv_ref::Function, n_mcmc::Int; niters::Int = 20000, 
                    learn_init::Bool = false, elbo_size::Int = 1, nBurn::Int = 0, optimizer = Flux.ADAM(1e-3), kwargs...) 
    
    logϵ, μ, D = log.(a.leapfrog_stepsize), a.μ, a.D
    ps = learn_init ? Flux.params(logϵ, μ, D) : Flux.params(logϵ) 
    

    #define loss
    loss = () -> begin 
        ϵ = exp.(logϵ)
        elbo = ELBO_opt(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; elbo_size = elbo_size, nBurn = nBurn)
        return -elbo
    end

    elbo_log, ps_log = vi_train!(niters, loss, ps, optimizer; kwargs...)
    return [[copy(p) for p in ps]], -elbo_log, ps_log
end


#####################
## KSD evaluation
#####################
function Sampler(o::HamFlow, a::HF_params, refresh::Function, n_mcmc::Int, N::Int; nBurn::Int64 = 0)
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

function Sampler!(T::Matrix{Float64}, o::HamFlow, a::HF_params, refresh::Function, n_mcmc::Int, N::Int; nBurn::Int64 = 0 )
    d = o.d
    @info "ErgFlow Sampling"
    prog_bar = ProgressMeter.Progress(N, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        # Sample(Unif{1, ..., n_mcmc})
        n_step = rand(nBurn+1:n_mcmc) 
        z0 = a.D .* o.q_sampler(d) .+ a.μ
        ρ0, u0 = o.ρ_sampler(d), rand()
        z, _, _ = flow_fwd(o, a.leapfrog_stepsize, refresh, z0, ρ0, u0, n_step)
        T[i,:] .= z 
        ProgressMeter.next!(prog_bar)
    end
end

# function KSD_trace(o::HamFlow, ps, ref::Function, n_mcmc::Int; learn_init = false, N = 1000, μ::Vector{Float64} = zeros(2), D::Vector{Float64} = ones(2), nBurn::Int = 0, bw = 0.05)
#     len = size(ps, 1)
#     KSDs = Vector{Float64}(undef, len)
#     # progress bar
#     prog_bar = ProgressMeter.Progress(len, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

#     for i in 1:len
#         a = learn_init ? HF_params(exp.(ps[i][1]), ps[i][2], ps[i][3]) : HF_params(exp.(ps[i][1]), μ, D) 
#         println("sample $i / $len")
#         # taking samples for fixed setting
#         Dat = Sampler(o, a, ref, n_mcmc, N; nBurn = nBurn)[1]
#         KSDs[i] = ksd(Dat, o.∇logp; bw = bw)
      
#         # update progress bar
#         ProgressMeter.next!(prog_bar)
#     end
#     return KSDs
# end