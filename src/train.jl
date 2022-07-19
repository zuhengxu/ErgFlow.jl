using Flux, ProgressMeter
using Zygote: Params, pullback


######################
## VI training functions (modify from Flux.optimise)
########################
#=
Perform update steps of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).
As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
=#

# Callback niceties
call(f::Function, args...) = f(args...)
call(f::Function, args::Tuple) = f(args...)

# function trace the loss
function cb_loss!(logging_loss, ls_trace, ls, iter)
    if logging_loss
        ls_trace[iter] = ls
    else
        nothing
    end
end

# function that trace the updated params
function cb_ps!(logging_ps, ps_trace, ps::Params, iter::Int, verbose_freq::Int)
    if logging_ps
        if iter % verbose_freq === 0
            # @info "training step $iter / $niters"
            # println(ps)
            pp = [copy(p) for p in ps]
            push!(ps_trace,  pp)
        end
    else
        nothing
    end
end

function vi_train!(niters::Int, loss, ps::Params, optimizer;
                    logging_loss = true, logging_ps = true, verbose_freq = 100)

    # progress bar
    @info "VI training"
    prog_bar = ProgressMeter.Progress(niters, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    # initialize ls_trace if logging_loss = true
    ls_trace = logging_loss ? Vector{Float64}(undef, niters) : nothing
    # initialize ps_trace if logging_ps = true
    ps_trace = logging_ps ? [] : nothing
    # logging init ps
    call(cb_ps!, logging_ps, ps_trace, ps, 0, verbose_freq)

    # optimization
    for iter in 1:niters
        # compute loss, grad simultaneously
        ls, back = pullback(ps)do
            loss()
        end
        grads = back(1.0)
        # update parameters
        Flux.update!(optimizer, ps, grads)

        # logging and printing
        call(cb_loss!, logging_loss, ls_trace, ls, iter)
        call(cb_ps!, logging_ps, ps_trace, ps, iter, verbose_freq)

        # update progress bar
        ProgressMeter.next!(prog_bar)
    end

    # return logging info
    return ls_trace, ps_trace
end

