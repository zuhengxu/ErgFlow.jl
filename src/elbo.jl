include("train.jl") # SGD using Zygote

#####################
# ELBO function: average of single_elbo (defined in each Hamflow file)
#####################

function ELBO(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
    if print
        prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    @threads for i in 1:elbo_size
        el_single = single_elbo_long(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
        Threads.atomic_add!(el, 1/elbo_size * el_single)
        ProgressMeter.next!(prog_bar)
    end
    return el[]
end


function ELBO_fast(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
    if print
        prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    @threads for i in 1:elbo_size
        el_single = single_elbo_fast(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
        Threads.atomic_add!(el, 1/elbo_size * el_single)
        ProgressMeter.next!(prog_bar)
    end
    return el[]
end

function ELBO_eff(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
    if print
        prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    @threads for i in 1:elbo_size
        el_single = single_elbo_eff(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
        Threads.atomic_add!(el, 1/elbo_size * el_single)
        ProgressMeter.next!(prog_bar)
    end
    return el[]
end
# function ELBO_long(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
#     el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
#     if print
#         prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
#     end
#     @threads for i in 1:elbo_size
#         el_single = single_elbo_long(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
#         Threads.atomic_add!(el, 1/elbo_size * el_single)
#         ProgressMeter.next!(prog_bar)
#     end
#     return el[]
# end


# # same ELBO function but without @threads---cannot use multithreads inside Zygote
# function ELBO_opt(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
#     el = 0.0    
#     for i in 1:elbo_size
#         el_single = single_elbo(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
#         el += 1/elbo_size * el_single
#     end
#     return el
# end

# # optimizing hyperparams (not working) 
# function HamErgFlow(o::ErgodicFlow, a::HF_params, refresh::Function, inv_ref::Function, n_mcmc::Int; niters::Int = 20000, 
#                     learn_init::Bool = false, elbo_size::Int = 1, nBurn::Int = 0, optimizer = Flux.ADAM(1e-3), kwargs...) 
    
#     logϵ, μ, D = log.(a.leapfrog_stepsize), a.μ, a.D
#     ps = learn_init ? Flux.params(logϵ, μ, D) : Flux.params(logϵ) 
    

#     #define loss
#     loss = () -> begin 
#         ϵ = exp.(logϵ)
#         elbo = ELBO_opt(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; elbo_size = elbo_size, nBurn = nBurn)
#         return -elbo
#     end

#     elbo_log, ps_log = vi_train!(niters, loss, ps, optimizer; kwargs...)
#     return [[copy(p) for p in ps]], -elbo_log, ps_log
# end
