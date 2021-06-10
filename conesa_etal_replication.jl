
# Replication of Conesa et al "Taxing Capital"
using Parameters, QuantEcon

#### Step 0: Create a struct for the model parameters  
@with_kw mutable struct TaxMod 
    # Demographics 
    jr::Int64 = 46 # Retirement age
    J::Int64 = 81 # Maximum age
    nn::Float64 = 0.011 # Population growth

    # Preferences 
    β::Float64 = 1.00093 # Discount Factor 
    σ::Float64 = 4.0 # Risk aversion 
    γ::Float64 = 0.377 # Consumption share 

    # Labor Productivity Process 
    var_α::Float64 = 0.14 # Variance types 
    ρ::Float64 = 0.98 # persistence 
    var_η::Float64 = 0.0289 # variance shock
    Nη::Int64 = 7 # Number of states
    η::Vector{Float64} = zeros(0) 
    Π::Matrix{Float64} = zeros(0,0) # Transition matrix

    # Technology
    α::Float64 = 0.36 # capital share
    δ::Float64 = 0.0833 # depreciation
    TFP::Float64 = 1.0 # scale parameterZ

    # Government Policy  
    τc::Float64 = 0.05 # consumption tax 
    κ0::Float64 = 0.258 # marginal tax
    κ1::Float64 = 0.768 # tax progressivity
    τp::Float64 = 0.124 # payroll tax 
    b::Float64 = 0.5 # Social security replacement rate
    maxSSrat::Float64 = 87000.0/37748.0

    # Grid sizes 
    ns::Int64 = 7   
    na::Int64 = 101 # asset grid
    nl::Int64 = 66  # leisure grid
    nty::Int64 = 2 # tax grid
    maxit::Int64 = 10000

    # Other
    blimit::Float64 = 0.0 # borrowing constraint
    umin::Float64 = -1.0E+2 # minimum utility 
    penscale::Float64 = 10000000 # penalty on utility for bad c or l
    maxl::Float64 = 0.99 # maximum value for labor supply

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
    setup_grid!()

This creates the grids for assets and labor. Also loads demographic 
characteristics: age efficiency units from Hansen 1993 and population
from Bell and Miller 2002.
"""
function setup_grid!(TM::TaxMod, scale = 75, curve = 2.5)
    @unpack na,nl,blimit,maxl,J,nty,nn,Nη,ρ,var_η = TM

    # Asset grid: with curvature
    grida = (scale - blimit).*LinRange(0,1,na).^curve .+ blimit
    # Labor grid: evenly spaced 
    gridl = LinRange(0,maxl,nl)

    # Loading demographic data from other script
    include("demographics.jl")
    ephansen,pop,surv,Nu,mu,ep,measty,topop = age_eff_pop(J,nty,nn)

    # Labor Productivity Markov Chain 
    mc = rouwenhorst(Nη,ρ,var_η)
    HH.Π = Π = mc.p
    HH.η = exp.(mc.state_values)


end


##### Step 2 


