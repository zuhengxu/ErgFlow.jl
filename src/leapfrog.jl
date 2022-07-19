function leapfrog(o::Union{HamFlow,HamFlowRot}, ϵ, z, ρ)
    for i in 1:o.n_lfrg
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
        z -= ϵ .* o.∇logp_mom(ρ)
        ρ += 0.5 .* ϵ .* o.∇logp(z) 
    end
    return z, ρ
end

# function leapfrog(o::Union{HamFlow,HamFlowRot}, ϵ, z, ρ)
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

function leapfrog!(o::HamFlow, ϵ, z, ρ)
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
    for i in 1:o.n_lfrg-1
        z .-= ϵ .* o.∇logp_mom(ρ)
        ρ .+= ϵ .* o.∇logp(z) 
    end
    z .-= ϵ .* o.∇logp_mom(ρ)
    ρ .+= 0.5 .* ϵ .* o.∇logp(z) 
end