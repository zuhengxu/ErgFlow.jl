#####################
# ELBO function: average of single_elbo (defined in each Hamflow file)
#####################

# estimating ELBO value for a fixed setting
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


# function ELBO_fast(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, n_mcmc::Int; elbo_size::Int = 1, nBurn::Int = 0, print = false)
#     el = Threads.Atomic{Float64}(0.0) # have to use atomic_add! to avoid racing 
#     if print
#         prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
#     end
#     @threads for i in 1:elbo_size
#         el_single = single_elbo_fast(o, ϵ, μ, D, refresh, inv_ref, n_mcmc; nBurn=nBurn)
#         Threads.atomic_add!(el, 1/elbo_size * el_single)
#         ProgressMeter.next!(prog_bar)
#     end
#     return el[]
# end

# estimating ELBO curve for various Ns
function ELBO_sweep(o::ErgodicFlow, ϵ, μ, D, refresh::Function, inv_ref::Function, Ns::Vector{Int}; elbo_size::Int = 1, nBurn::Int = 0, print = false)
    if print
        prog_bar = ProgressMeter.Progress(elbo_size, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    ELs = zeros(size(Ns,1), elbo_size)
    @threads for i in 1:elbo_size
        ELs[:, i] .= single_elbo_sweep(o, ϵ, μ, D, refresh, inv_ref, Ns; nBurn=nBurn)
        ProgressMeter.next!(prog_bar)
    end
    return vec(mean(ELs; dims = 2))
end
