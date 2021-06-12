

# Replication of Conesa et al "Taxing Capital"
using Parameters, QuantEcon, BasisMatrices,LinearAlgebra,Optim,DataFrames,Gadfly,SparseArrays,Arpack,Roots

#### Step 0: Create a struct for the model parameters  
@with_kw mutable struct TaxMod 
    # Demographics 
    jr::Int64 = 46 # Retirement age
    J::Int64 = 81 # Maximum age
    nn::Float64 = 0.011 # Population growth
    ψ::Vector{Float64} = zeros(0) # survival probabilities

    # Preferences 
    β::Float64 = 1.00093 # Discount Factor 
    σ::Float64 = 4.0 # Risk aversion 
    γ::Float64 = 0.377 # Consumption share 

    # Labor Productivity Process 
    var_α::Float64 = 0.14 # Variance types 
    ρ::Float64 = 0.98 # persistence 
    var_η::Float64 = 0.0289 # variance shock
    Nη::Int64 = 7 # Number of states
    η::Vector{Float64} = zeros(0) # productivity shock states
    Π::Matrix{Float64} = zeros(0,0) # Transition matrix
    ϵ::Matrix{Float64} = zeros(0,0) # ability x age efficiency matrix

    # Technology
    α::Float64 = 0.36 # capital share
    δ::Float64 = 0.0833 # depreciation
    TFP::Float64 = 1.0 # scale parameter Z
    w::Float64 = 1.0 # wages  
    r::Float64 = 0.05 # interest rate 

    # Government Policy  
    τc::Float64 = 0.05 # consumption tax 
    τk::Float64 = 0.36 # capital tax
    κ0::Float64 = 0.258 # marginal tax
    κ1::Float64 = 0.768 # tax progressivity
    κ2::Float64 = 0.0 # deduction
    τp::Float64 = 0.124 # payroll tax 
    b::Float64 = 0.5 # Social security replacement rate
    maxSSrat::Float64 = 87000.0/37748.0

    # Grid sizes 
    ns::Int64 = 7 #X Also number of states?
    na::Int64 = 101 # asset grid
    nl::Int64 = 66  # leisure grid
    nty::Int64 = 2 
    maxit::Int64 = 10000

    # Other
    a̲::Float64 = 0.0 # borrowing constraint
    umin::Float64 = -1.0E+2 # minimum utility 
    penscale::Float64 = 10000000 # penalty on utility for bad c or l
    l̅::Float64 = 0.99 # maximum value for labor supply 
    
    # Spending stuff
    G::Float64 = 6.670
    Tr::Float64 = 2.0
    SS::Float64 = 2.0

    #Solution
    agrid::Vector{Float64} = zeros(0)
    k::Int = 2 #type of interpolation
    Vf::Array{Interpoland} = Interpoland[]
    cf::Array{Interpoland} = Interpoland[]
    

end;

"""
    U(TM::TaxMod, c, l)

Calculates household utility given consumption, leisure, and preferences
"""
function U(TM::TaxMod, c, l)
    @unpack σ,γ,penscale,umin = TM
    
    if c <= 0.0 # Setting very low utility for negative consumption
		U = umin - penscale * c^2
	elseif l < 0.0 # Setting very low utility for negative labor supply
		U = umin - penscale * l^2
	elseif l >= 1.0 # Setting very low utility for labor supply over 1
		U = umin - penscale*(l-1.0)^2
	else # Real utility function
		U = (1/(1-σ)) * (((c^γ)*((1-l)^(1-γ)))^(1-σ))
	end

    return U
end

"""
    marginal_utility(TM::TaxMod, c, l)

Calculates marginal utility 
"""
function marginal_utility(TM::TaxMod, c, l)
    @unpack σ,γ = TM

    if c > 0.0
        mu = (γ*(c^γ*(1.0-l)^(1.0-γ))^(1.0-σ))/c
	else # Very large MU if c is zero or negative
		mu = 1000000.0+ abs(c)^2.0
	end 

end

"""
    tax_gs(y, k0, k1, k2)

Gouveia and Strauss tax function 
"""
function tax_gs(y, k0, k1, k2)

    tax_gs = 0.0

    if k1 >= 0.001 && y > 0.0
        tax_gs = k0*(y - (y^(-k1)+k2)^(-1/k1))
    elseif k1 < 0.001 && y > 0.0 #
        tax_gs = k0*y + k2
    else # No taxes if income is negative/zero
        tax_gs = 0.0
    end
    
    return tax_gs
end

"""
    Tax3()

This tax function is separable in capital and labor income
"""

function Tax3(l0,l1,l2,ca0,lear,caear)
    
    Tax3=0.0
    
    # Labor Income Tax     
    if l1 >= 0.000001 && lear >= 0.00000001
        Tax3 = l0*( lear - ( lear^(-l1)+l2 )^(-1.0/l1) )
    elseif l1 < 0.000001 && lear >= 0.00000001
        Tax3 = l0*lear+l2
    else 
        Tax3 = 0.0
    end
    
    # Capital Income Tax
    Tax3 = Tax3 + ca0*caear
    
end 

"""
    marginal_tax_gs(k0,k1,k2,cap_earn)
"""
function marginal_tax_gs(k0,k1,k2,cap_earn)

    mt = 0.0
    if k1 >= 0.001 && cap_earn > 0.0
        mt = k0*(1 - (1 + k2*cap_earn^k1)^(-1/k1 - 1))
    elseif k1 < 0.001 && cap_earn > 0.0
        mt = k0
    end 

    return mt
end 

"""
    marginal_tax3(ca0)
"""
function marginal_tax3(ca0)
    return ca0
end 


### SEE PARAMS for basefun subroutine
tm = TaxMod()

##### Step 1: Create a grid of individual asset holdings. define survival rates, age distribution, and labor efficiency grids 

"""
    l(c,j)

Static FOC relating consumption and leisure 
"""
function l(TM::TaxMod, c)
    @unpack jr,γ,τc,ϵ,w,η = TM

    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                if j >= jr
                    l[s,ty,j] = 0
                else
                    l[s,ty,j] = 1 - ((1-γ)/γ)*(1 + τc)/(w*)  

                cf[s]= Interpoland(abasis,c)
            end
        end
    end
      


end

"""
    setup_grid!()

This creates the grids for assets and labor. Also loads demographic 
characteristics: age efficiency units from Hansen 1993 and population
from Bell and Miller 2002.
"""
a′grid = zeros(tm.na)
lgrid = zeros(tm.nl)

function setup_grid!(TM::TaxMod, a̅ = 75, curve = 2.5)
    @unpack na,nl,a̲,l̅,J,nty,nn,ns,ρ,var_η,var_α,Vf,cf,k = tm

    # Asset grid: with curvature
    tm.agrid = (a̅ - a̲).*LinRange(0,1,na).^curve .+ a̲
    # Labor grid: evenly spaced 
    lgrid = LinRange(0,l̅,nl)

    # Loading demographic data from other script
    include("demographics.jl")
    #ephansen,pop,surv,Nu,mu,ep,measty,topop = age_eff_pop(J,nty,nn)
    TM.ψ = age_eff_pop(J,nty,nn).surv
    ephansen = age_eff_pop(J,nty,nn).ephansen

    # Labor Productivity Markov Chain 
    mc = rouwenhorst(ns,ρ,var_η)
    TM.Π = mc.p
    TM.η = exp.(mc.state_values)

    # Fixed effects 
    TM.ϵ = zeros(nty,J)
    TM.ϵ[1,1:J] = exp(-sqrt(var_α)) .* ephansen[1:J]
    TM.ϵ[2,1:J] = exp(sqrt(var_α)) .* ephansen[1:J]

    #First guess of interpolation functions
    abasis = Basis(SplineParams(a′grid,0,k))
    a = nodes(abasis)[1]

    Vf = TM.Vf = Array{Interpoland}(undef,ns,nty,J)
    cf = TM.cf = Array{Interpoland}(undef,ns,nty,J)
    lf = Array{Interpoland}(undef,ns,nty,J)

    for s in 1:ns
        for 
            c = @. r̄*a + w̄*HH.ϵ[s]
            V = U(HH,c)./(1-β)
            l = 

            Vf[s]= Interpoland(abasis,V)
            cf[s]= Interpoland(abasis,c)
    
        end
    end
    
    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                
                c[:,s,ty,j]= @. r*a + w*TM.ϵ[s]
                V = U.(c[:,s,ty,j],0.5)./(1-β)
    
                Vf[s]= Interpoland(abasis,V)
                cf[s]= Interpoland(abasis,c)
            end
        end
    end
    
end

setup_grid!(tm)
##### Step 2: Solving household problem

# Start with grid of next period's assets
# Guess consumption tommorrow
# Use to get consumption today w/ EE
# Use that to get labor using FOC 
# Use c, labor, a′ to get assets today 
#


"""
    iterate_endogenousgrid(HH,a′grid,cf′)

Iterates on Euler equation using endogenous grid method

c needs to be na x ns

"""
function iterate_endogenousgrid(tm::TaxMod,a′grid,cf′)
    @unpack ns,nty,J,jr,ψ,β,τk,τc,σ,κ0,κ1,κ2,w,ϵ,η,Π = tm

    # Finding initial guess for c′ at a′grid values for each ability type and period
    c′ = zeros(length(a′grid),ns,nty,J)
    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                c′[:,s,ty,j]= cf′[s,ty,j](a′grid)
            end
        end
    end

    c = zeros(length(a′grid),ns,nty,J)
    # Finding c from c′ using Euler Equation
    for i in 1:length(a′grid)
        for s in 1:ns
            for ty in 1:nty
                for j in 1:(J-1)
                    if j == J
                        c[i,s,ty,j] = 0
                    elseif j >= jr
                        c[i,s,ty,j] = 0
                    elseif j < jr
                        c[i,s,ty,j] = ((β * ψ[j] * ((1+r(1-τk))/(1+τc)) * c′[i,s,ty,j]^(-σ) *
                            ((1 - marginal_tax_gs(κ0,κ1,κ2,w*ϵ[ty,j]) - τss)*ϵ[ty,j]*η[s])/
                            ((1 - marginal_tax_gs(κ0,κ1,κ2,w*ϵ[ty,j+1]) - τss)*ϵ[ty,j+1]) ./ η) * Π[s,:])^(1/(σ))
                    end
                end
            end
        end
    
                    (1+r̄)*(c′).^(-γ)*Π' #RHS of Euler Equation
    c = EERHS.^(-1/γ)

    #compute implies assets
    a = ((c .+ a′grid) .- w̄ .*ϵ')./(1+r̄)

    cf = Vector{Interpoland}(undef,Nϵ)
    for s in 1:Nϵ
        if a[1,s]> a̲
            c̲ = r̄*a̲ + w̄*ϵ[s]
            cf[s]= Interpoland(Basis(SplineParams([a̲; a[:,s]],0,1)),[c̲;c[:,s]])
        else
            cf[s]= Interpoland(Basis(SplineParams(a[:,s],0,1)),c[:,s])
        end
    end
    return cf
end;
