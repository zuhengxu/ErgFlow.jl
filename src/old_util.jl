
function erfcinv_finite(y)
    yfloat = Float64(y)
    xfloat = erfcinv(yfloat)
    if isfinite(xfloat)
        x = BigFloat(xfloat)
    else
        # Float64 overflowed, use asymptotic estimate instead
        # from erfc(x) ≈ exp(-x²)/x√π ≈ y  ⟹  -log(yπ) ≈ x² + log(x) ≈ x²
        if yfloat < 1
            x = sqrt(-log(y*sqrtπ))
        else # y must be close to 2
            x = -sqrt(-log((2-y)*sqrtπ))
        end
        # TODO: Newton convergence is slow near y=0 singularity; accelerate?
        isfinite(x) || return x
    end
    sqrtπhalf = sqrtπ * BigFloat(0.5)
    tol = 2eps(abs(x))
    n = 1
    while n < 100000 # Newton iterations
        Δx = sqrtπhalf * (erfc(x) - y) * exp(x^2)
        x += Δx
        abs(Δx) < tol && break
        n += 1
    end
    return x
end

############################
# Gaussian
############################

function cdf_normal(x, μ, σ) 
    z = (x- μ) / (sqrt(2.0)*σ)
    return 0.5 + 0.5*(1.0 - erfc(z))
end 

function invcdf_normal(x, μ, σ)
    z = 2.0*x - 1.0
    m1 = 1.0 - z
    s = sqrt(2.0)*σ
    shift = μ
    # m = erfcinv_finite(m1)
    m = erfcinv(m1)
    m = s*m + shift
    return m
end 

cdf_normal_std(x) = cdf_normal(x, 0.0, 1.0)
invcdf_normal_std(x) = invcdf_normal(x, 0.0, 1.0)


# invcdf_normal_std(cdf_normal_std(4.5))

function pdf_normal(x)
    return 1.0/sqrt(2.0π)*exp(-0.5*x^2.0)
end

function lpdf_normal(x)
    return -0.5*(x^2.0 + log(2.0π))
end

function ∇lpdf_normal(x)
    return -x 
end

############################
# Cauchy
############################

function cdf_cauchy_std(x) 
    return atan(x) / π + 0.5
end 

function invcdf_cauchy_std(x)
    return tan((x - 0.5) * pi)
end 

function pdf_cauchy_std(x)
    return 1.0 / (π * (1 + x^2))
end

function lpdf_cauchy_std(x)
    return -log(π) - log(1+x^2)
end

function ∇lpdf_cauchy_std(x)
    return -2*x / (x^2 + 1)  
end

function randc(size)
    return rand(Cauchy(), size)
end

randc()=rand(Cauchy())