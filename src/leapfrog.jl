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