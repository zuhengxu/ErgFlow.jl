################
# leapfrog update
################
function leapfrog(o::ErgodicFlow, ϵ, z, ρ)
    for i in 1:o.n_lfrg
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
        z -= ϵ .* o.∇logp_mom(ρ)
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
    end
    return z, ρ
end

function leapfrog(∇logp::Function, ∇logm::Function, n_lfrg::Int, ϵ::Vector{Float64}, z, ρ)
    for i in 1:n_lfrg
        ρ += 0.5 .* ϵ .* ∇logp(z) 
        z -= ϵ .* ∇logm(ρ)
        ρ += 0.5 .* ϵ .* ∇logp(z) 
    end
    return z, ρ
end

function leapfrog_save!(T, M, o::ErgodicFlow, ϵ, z, ρ; freq::Int=10)
    for i in 1:o.n_lfrg
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
        z -= ϵ .* o.∇logp_mom(ρ)
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
        if i % freq == 0
            T[Int(i/freq),:] .= z 
            M[Int(i/freq),:] .= ρ
        end
    end
    return z, ρ
end


# function leapfrog(o::ErgodicFlow, ϵ, z, ρ)
# ```
# leapfrog that combines 2 consecutive steps into 1
# ```
#     ρ += 0.5 .* ϵ .* o.∇logp(z) 
#     for i in 1:o.n_lfrg-1
#         z -= ϵ .* o.∇logp_mom(ρ)
#         ρ += ϵ .* o.∇logp(z) 
#     end
#     z -= ϵ .* o.∇logp_mom(ρ)
#     ρ += 0.5 .* ϵ .* o.∇logp(z) 
#     return z, ρ
# end

# function leapfrog(∇logp::Function, ∇logm::Function, n_lfrg::Int, ϵ::Vector{Float64}, z, ρ)
#     ρ += 0.5 .* ϵ .* ∇logp(z) 
#     for i in 1:n_lfrg-1
#         z -= ϵ .* ∇logm(ρ)
#         ρ += ϵ .* ∇logp(z) 
#     end
#     z -= ϵ .* ∇logm(ρ)
#     ρ += 0.5 .* ϵ .* ∇logp(z) 
#     return z, ρ
# end

function leapfrog!(o::ErgodicFlow, ϵ, z, ρ)
```
in place leapfrog update "(combines 2 consecutive steps into 1)"
```
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z .-= ϵ .* o.∇logp_mom(ρ)
        ρ .+= ϵ .* o.∇logp(z) 
    end
    z .-= ϵ .* o.∇logp_mom(ρ)
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
end



######################
# forward/backward Flow
######################

function flow_fwd(o::ErgodicFlow, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ, u = refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
    end
    return z, ρ, u
end

function flow_bwd(o::ErgodicFlow, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    for i in 1:n_mcmc - 1 
        ρ, u = inv_ref(o, z, ρ, u)
        z, ρ = leapfrog(o, -ϵ, z, ρ)
    end
    return z, ρ, u
end

function flow_fwd_trace(o::ErgodicFlow, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(undef, n_mcmc, o.d)
    M = Matrix{eltype(ρ)}(undef, n_mcmc, o.d)
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

function flow_bwd_trace(o::ErgodicFlow, ϵ::Vector{Float64}, inv_ref::Function, z, ρ, u, n_mcmc::Int)
    T = Matrix{eltype(z)}(undef, n_mcmc, o.d)
    M = Matrix{eltype(ρ)}(undef, n_mcmc, o.d)
    U = Vector{typeof(u)}(undef, n_mcmc)
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

## this is only for vis
function flow_fwd_save(o::ErgodicFlow, ϵ::Vector{Float64}, refresh::Function, z, ρ, u, n_mcmc::Int; freq::Int = 10)
    n = Int(floor(o.n_lfrg / freq))
    T1, M1 = zeros(n, o.d, n_mcmc-1), zeros(n, o.d, n_mcmc-1)
    T = Matrix{eltype(z)}(undef, n_mcmc, o.d)
    M = Matrix{eltype(ρ)}(undef, n_mcmc, o.d)
    U = Vector{typeof(u)}(undef, n_mcmc)
    T[1,:] .= z
    M[1,:] .= ρ
    U[1] = u
    prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i in 1:n_mcmc - 1 
        z, ρ = leapfrog_save!(@view(T1[:,:,i]), @view(M1[:,:,i]), o, ϵ, z, ρ; freq = freq)
        ρ, u = refresh(o, z, ρ, u)
        # println(i, "/$n_mcmc")
        T[i+1,:] .= z
        M[i+1,:] .= ρ
        U[i+1] = u
        ProgressMeter.next!(prog_bar)
    end
    return vcat(eachslice(T1, dims =3)...), vcat(eachslice(M1, dims = 3)...), T, M, U
end



#################
# generating samples from the flow
#################3
function Sampler(o::ErgodicFlow, a::HF_params, refresh::Function, n_mcmc::Int, N::Int; nBurn::Int64 = 0)
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

function Sampler!(T::Matrix{Float64}, o::ErgodicFlow, a::HF_params, refresh::Function, n_mcmc::Int, N::Int; nBurn::Int64 = 0 )
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