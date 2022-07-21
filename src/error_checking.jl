function leapfrog_error(o::ErgodicFlow, ϵ, z, ρ)
    z1, ρ1 = leapfrog(o, ϵ, z, ρ)
    z0, ρ0 = leapfrog(o, -ϵ, z1, ρ1)
    err = sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0)
    return sqrt(err) 
end

function refresh_error(o::ErgodicFlow, refresh::Function, inv_ref::Function, z, ρ, u)
    ρ1, u1 = refresh(o, z, ρ, u)
    ρ0, u0 = inv_ref(o, z, ρ1, u1)
    err = sum(abs, ρ0 .- ρ) + sum(abs, u0 - u)
    return sqrt(err)
end 

function one_step_error(o::ErgodicFlow, refresh::Function, inv_ref::Function, ϵ, z, ρ, u)
    z0, ρ0, u0 = copy(z), copy(ρ), copy(u)
    z, ρ = leapfrog(o, ϵ, z, ρ)
    ρ, u = refresh(o, z, ρ, u)
    ρ, u = inv_ref(o, z, ρ, u)
    z, ρ = leapfrog(o, -ϵ, z, ρ)
    error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) + sum(abs2, u - u0)
    return sqrt(error)
end

function leapfrog_error(o::ErgodicFlow, a::HF_params, n_lfrg::Int; Nsample::Int = 100)
```
compute "||(z, ρ, u) - lfrg∘lfrg^{-1}(z, ρ, u)||"
```
    E = zeros(Nsample)
    prog_bar = ProgressMeter.Progress(Nsample, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i ∈ 1:Nsample
        # take initial momentum and position 
        ρ0= o.ρ_sampler(o.d)
        z0 = a.D.* o.q_sampler(o.d) .+ a.μ # q0 ∼ N(μ, D^2)
        # fwd and bwd leapfrog
        z1, ρ1 = leapfrog(o.logp, o.lpdf_mom, n_lfrg, a.leapfrog_stepsize, z0, ρ0)
        z, ρ= leapfrog(o.logp, o.lpdf_mom, n_lfrg, -a.leapfrog_stepsize, z1, ρ1)
        # compute err
        error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0)
        E[i] =  sqrt(error)
        ProgressMeter.next!(prog_bar)
    end
    return E
end

function error_checking(o::ErgodicFlow, ϵ::Vector{Float64}, refresh::Function, inv_ref::Function, z0, ρ0, u0, n_mcmc::Int)
```
T = ergflow
compute "||(z0, ρ0, u0) - T^{-1}∘T(z0, ρ0, u0)|| for given (z0, ρ0, u0) "
```
    # fwd and bwd flow
    z1, ρ1, u1 = flow_fwd(o, ϵ, refresh, z0, ρ0, u0, n_mcmc)
    z, ρ, u = flow_bwd(o, ϵ, inv_ref, z1, ρ1, u1, n_mcmc)
    # compute err
    error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) + sum(abs2, u - u0)
    return sqrt(error)
end

function error_checking_fwd(o::ErgodicFlow, a::HF_params, n_mcmc::Int; Nsample::Int = 100, refresh::Function = pseudo_refresh_coord, inv_ref::Function = inv_refresh_coord)
```
T = ergflow
compute "||(z, ρ, u) - T^{-1}∘T(z, ρ, u)||"
```
    E = zeros(Nsample)
    prog_bar = ProgressMeter.Progress(Nsample, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i ∈ 1:Nsample
        # take init sample
        ρ0, u0 = o.ρ_sampler(o.d), rand()
        z0 = a.D.* o.q_sampler(o.d) .+ a.μ
        # fwd and bwd flow
        z1, ρ1, u1 = flow_fwd(o, a.leapfrog_stepsize, refresh, z0, ρ0, u0, n_mcmc)
        z, ρ, u = flow_bwd(o, a.leapfrog_stepsize, inv_ref, z1, ρ1, u1, n_mcmc)
        # compute err
        error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) + sum(abs2, u - u0)
        E[i] =  sqrt(error)
        ProgressMeter.next!(prog_bar)
    end
    return E
end

function error_checking_bwd(o::ErgodicFlow, a::HF_params, n_mcmc::Int; Nsample::Int = 100, refresh::Function = pseudo_refresh_coord, inv_ref::Function = inv_refresh_coord)
```
T = ergflow
compute "||(z, ρ, u) - T∘T^{-1}(z, ρ, u)||"
```
    E = zeros(Nsample)
    prog_bar = ProgressMeter.Progress(Nsample, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i ∈ 1:Nsample
        # take init sample
        ρ0, u0 = o.ρ_sampler(o.d), rand()
        z0 = a.D.* o.q_sampler(o.d) .+ a.μ
        # bwd and fwd flow
        z1, ρ1, u1 = flow_bwd(o, a.leapfrog_stepsize, inv_ref, z0, ρ0, u0, n_mcmc)
        z, ρ, u = flow_fwd(o, a.leapfrog_stepsize, refresh, z1, ρ1, u1, n_mcmc)
        # compute err
        error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0) + sum(abs2, u - u0)
        E[i] = sqrt(error)
        ProgressMeter.next!(prog_bar)
    end
    return E
end

function check_time_shift(t0::Float64, Ns::Vector{Int64}, shift::Function, inv_shift::Function)
    E = zeros(size(Ns, 1))
    @threads for k in 1:size(Ns,1)
        t  = t0
        for i = 1:Ns[k] 
                t = shift(t)
        end
        for i = 1:Ns[k] 
                t = inv_shift(t)
        end
        err = norm(t-t0)
        E[k] = err 
    end
    return E 
end    
    
function flow_fwd_err_tr(o::ErgodicFlow, ϵ::Vector{Float64}, refresh::Function,inv_ref::Function, z, ρ, u, n_mcmc::Int)
    leap_err = Vector{typeof(u)}(undef, n_mcmc)
    ref_err = Vector{typeof(u)}(undef, n_mcmc)
    total_err = Vector{typeof(u)}(undef,n_mcmc)
    leap_err[1,:] .= 0
    ref_err[1,:] .= 0
    total_err[1] = 0
    # prog_bar = ProgressMeter.Progress(n_mcmc-1, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    E = error_checking(o, ϵ, refresh, inv_ref, z, ρ, u, n_mcmc)
    for i in 1:n_mcmc - 1 
        # computing err
        leap_err[i+1] = leapfrog_error(o, ϵ, z, ρ)
        ref_err[i+1] = refresh_error(o, refresh, inv_ref, z, ρ, u)
        total_err[i+1] = one_step_error(o, refresh, inv_ref, ϵ, z, ρ, u)
        # perform fwd flow
        z, ρ = leapfrog(o, ϵ, z, ρ)
        ρ, u = refresh(o, z, ρ, u)
        # ProgressMeter.next!(prog_bar)
    end
    return leap_err, ref_err, total_err, E
end

 
function flow_fwd_err_tr(o::ErgodicFlow, a::HF_params, refresh::Function,inv_ref::Function, n_mcmc::Int; Nsample::Int= 100)

    leap_E, ref_E = zeros(Nsample, n_mcmc), zeros(Nsample, n_mcmc) 
    total_E = zeros(Nsample, n_mcmc)
    EE = zeros(Nsample)
    prog_bar = ProgressMeter.Progress(Nsample, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i ∈ 1:Nsample
        # take init sample
        ρ0, u0 = o.ρ_sampler(o.d), rand()
        z0 = a.D.* o.q_sampler(o.d) .+ a.μ
        # compute err
        L, R, T, ee = flow_fwd_err_tr(o, a.leapfrog_stepsize, refresh, inv_ref, z0, ρ0, u0, n_mcmc) 
        leap_E[i, :] .= L 
        ref_E[i, :] .= R 
        total_E[i, :] .= T 
        EE[i] = ee
        ProgressMeter.next!(prog_bar)
    end
    
    return leap_E', ref_E', total_E', EE

end