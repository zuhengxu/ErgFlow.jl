function leapfrog_error(o::ErgodicFlow, a::HF_params, n_lfrg::Int; Nsample::Int = 100)
    ```
    compute ||(z, ρ, u) - lfrg∘lfrg^{-1}(z, ρ, u)||
    ```
    E = zeros(Nsample)
    prog_bar = ProgressMeter.Progress(Nsample, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i ∈ 1:Nsample
        # take initial momentum and position 
        ρ0= o.ρ_sampler(o.d)
        z0 = a.D.* o.q_sampler(o.d) .+ a.μ # q0 ∼ N(μ, D^2)
        # fwd and bwd leapfrog
        z1, ρ1 = leapfrog(o.logp, o.logp_mom, n_lfrg, a.leapfrog_stepsize, z0, ρ0)
        z, ρ= leapfrog(o.logp, o.logp_mom, n_lfrg, -a.leapfrog_stepsize, z1, ρ1)
        # compute err
        error =  sum(abs2, z .- z0) + sum(abs2, ρ.-ρ0)
        E[i] =  sqrt(error)
        ProgressMeter.next!(prog_bar)
    end
    return E
end

function error_checking_fwd(o::ErgodicFlow, a::HF_params, n_mcmc::Int; Nsample::Int = 100, refresh::Function = pseudo_refresh_coord, inv_ref::Function = inv_refresh_coord)
    ```
    T = ergflow
    compute ||(z, ρ, u) - T^{-1}∘T(z, ρ, u)||
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
    compute ||(z, ρ, u) - T∘T^{-1}(z, ρ, u)||
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

