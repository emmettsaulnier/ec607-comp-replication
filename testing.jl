

using BasisMatrices,LinearAlgebra,Parameters,Optim,QuantEcon,DataFrames,Gadfly,SparseArrays,Arpack
using Roots


@with_kw mutable struct HHModel
    #Preference Parameters
    γ::Float64 = 1. #Risk aversion
    β::Float64 = 0.985 #Discount Rate

    #Prices
    r̄::Float64 = .01 #quarterly
    w̄::Float64 = 1.

    #Asset Grid Parameters
    a̲::Float64 = 0. #Borrowing Constraint
    a̅::Float64 = 600. #Upper Bound on assets
    Na::Int64 = 100 #Number of grid points for splines

    #Income Process
    ρ_ϵ::Float64 = 0.9923 #calibrated to quarterly wage regressions
    σ_ϵ::Float64 = 0.0983
    Nϵ::Int64 = 7
    ϵ::Vector{Float64} = zeros(0)
    Π::Matrix{Float64} = zeros(0,0)

    #Solution
    k::Int = 2 #type of interpolation
    Vf::Vector{Interpoland} = Interpoland[]
    cf::Vector{Interpoland} = Interpoland[]

    #Extra
    EΦ′::SparseMatrixCSC{Float64,Int64} = spzeros(0,0)
    Φ::SparseMatrixCSC{Float64,Int64} = spzeros(0,0)
end;

"""
    U(HH::HHModel,c)
"""
function U(HH::HHModel,c)
    γ = HH.γ
    if γ == 1
        return log.(c)
    else
        return (c.^(1-γ))./(1-γ)
    end
end

"""
    setupgrids_shocks!(HH::HHModel, curv=1.7)

Set up non-linear grids for interpolation
"""
function setupgrids_shocks!(HH::HHModel, curv=1.7)
    @unpack a̲,a̅,Na,ρ_ϵ,σ_ϵ,Nϵ,k,r̄,w̄,β = HH
    #Compute grid on A
    agrid = (a̅-a̲).*LinRange(0,1,Na).^curv .+ a̲

    #Store markov chain
    mc = rouwenhorst(Nϵ,ρ_ϵ,σ_ϵ)
    HH.Π = Π = mc.p
    HH.ϵ = exp.(mc.state_values)

    #First guess of interpolation functions
    abasis = Basis(SplineParams(agrid,0,k))
    a = nodes(abasis)[1]

    Vf = HH.Vf = Vector{Interpoland}(undef,Nϵ)
    cf = HH.cf = Vector{Interpoland}(undef,Nϵ)
    for s in 1:Nϵ
        c = @. r̄*a + w̄*HH.ϵ[s]
        V = U(HH,c)./(1-β)

        Vf[s]= Interpoland(abasis,V)
        cf[s]= Interpoland(abasis,c)
    end

    #Expectations of 1st derivative of Basis functions
    HH.EΦ′ = kron(Π,BasisMatrix(abasis,Direct(),nodes(abasis)[1],[1]).vals[1])
    HH.Φ = kron(Matrix{Float64}(I,Nϵ,Nϵ),BasisMatrix(abasis,Direct()).vals[1])
end;

"""
computeoptimalconsumption(HH::HHModel,V)

Computes optimal savings using endogenous grid method.  
"""
function computeoptimalconsumption(TM::TaxModel,Vcoefs::Array{Float64},ty,j)::Vector{Interpoland}
    @unpack a̲,EΦ′,η,ϵ,r,w,γ,β,ns,σ,jr,lgrid,nl,k,κ0,κ1,κ2,τp,τc,ȳ,ψ,Π = TM

    a′grid = nodes(TM.Vf[1,1,1].basis)[1]
    na = length(a′grid)
    EV_a = reshape(EΦ′*Vcoefs,:,ns) #Compute expectations of V'(a',s') using matrix multiplication
    v_max = zeros(na,ns)
    c = zeros(na,ns)
    a = zeros(na,ns)
    l = zeros(na,ns)

    if j < jr
        cl = zeros(nl,na,ns)
        al = zeros(nl,na,ns)
        for l in 1:nl # Find optimal consumption for each l in lgrid
            earn = w*ϵ[ty,j]*η*lgrid[l]
            lab_tax = tax_gs(κ0,κ1,κ2,earn)
            ss_tax =  τp*min.(earn,ȳ)
            # Consumption from Euler Equation, Implied assets today from Budget Constraint
            cl[l,:,:] = (β*ψ[j]/(γ*(1-lgrid[l]))^((1-σ)*(1-γ)).*EV_a).^(1/(γ*(1-σ)-1)) 
            al[l,:,:] = ((a′grid .+ (1+τc)*c[l] .+ ss_tax' .+ lab_tax' .- earn')/(1+r*(1-τk))) - Tr
            for apr in 1:na, s in 1:ns
                val[l,apr,s] = U(TM,c[l,apr,s],lgrid[l]) + β*ψ[j]*sum(map(eta -> TM.Vf[eta,ty,j+1](apr) * Π[s,eta],1:ns))
            end 
        end 
        for apr in 1:na, s in 1:ns
           v_max[apr,s],ind_max[apr,s] = findmax(val[:,apr,s])
           c[apr,s] = cl[ind_max[apr,s],apr,s]
           a[apr,s] = al[ind_max[apr,s],apr,s] 
           l[apr,s] = lgrid[ind_max[apr,s]] 
        end 
    end

    cf = Vector{Interpoland}(undef,ns)#implied policy rules for each productivity
    for s in 1:ns
        #if a[1,s]> a̲
        #    c̲ = (earn - ss_tax + (1+r*(1-τk))*(a + Tr) - lab_tax - a̲)/(1+τc)
        #    cf[s]= Interpoland(Basis(SplineParams([a̲; a[:,s]],0,k)),[c̲;c[:,s]]) 
        #else
            cf[s]= Interpoland(Basis(SplineParams(a[:,s],0,k)),c[:,s])
        #end
    end
    return cf
end;
            

"""
iteratebellman_newton!(AM::AiyagriModel,Vcoefs)

Updates the coefficients of the value function using newton's method
"""
function iteratebellman_newton!(TM::TaxMod)
    @unpack β,ϵ,Π,r̄,w̄,Nϵ,Vf,Φ  ,ns,ty,j = TM

    agrid = nodes(Vf[1,1,1].basis)[1]
    Na = length(agrid)
    Vcoefs = zeros(ns)
    cf = Array{Interpoland}(undef,ns,nty,J);

    for ty in 1:nty, j in 1:J
        Vcoefs = vcat([Vf[s,ty,j].coefs for s in 1:ns]...)::Vector{Float64}
        cf[:,ty,j] = computeoptimalconsumption(TM,Vcoefs,ty,j) #Compute optimal consumption function
        
        c = zeros(Na*Nϵ) 
        EΦ = spzeros(Na*Nϵ,Na*Nϵ)
        for s in 1:Nϵ
            for s′ in 1:Nϵ
                c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
                a′ = (1+r̄)*agrid .+ ϵ[s]*w̄ .- c[(s-1)*Na+1:s*Na] #asset choice
                #Compute expectation of basis functions at a′
                EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = Π[s,s′]*BasisMatrix(Vf[s].basis,Direct(),a′).vals[1][:]
            end
        end


        Jac = β.*EΦ .- Φ
        res = U(HH,c) .+ Jac*Vcoefs 
        Vcoefs′ = Vcoefs - Jac\res #newtons method
        for s in 1:Nϵ
            Vf[s].coefs .= Vcoefs′[1+(s-1)*Na:s*Na]
        end
    end 
    return norm(res,Inf)
end;

"""
    iteratebellman_time!(AM::AiyagriModel,Vcoefs)

Updates the coefficients of the value function using time iteration of the bellman equation
"""
function iteratebellman_time!(HH::HHModel)
    @unpack β,ϵ,Π,r̄,w̄,Nϵ,Vf,Φ = HH
    Vcoefs = vcat([Vf[s].coefs for s in 1:Nϵ]...)::Vector{Float64}
    agrid = nodes(Vf[1].basis)[1]
    Na = length(agrid)

    cf = computeoptimalconsumption(HH,Vcoefs) #Compute optimal consumption function
    c = zeros(Na*Nϵ) 
    EΦ = spzeros(Na*Nϵ,Na*Nϵ)
    for s in 1:Nϵ
        for s′ in 1:Nϵ
            c[(s-1)*Na+1:s*Na] = cf[s](agrid) #compute consumption at gridpoints
            a′ = (1+r̄)*agrid .+ ϵ[s]*w̄ .- c[(s-1)*Na+1:s*Na] #asset choice
            #Compute expectation of basis functions at a′
            EΦ[(s-1)*Na+1:s*Na,(s′-1)*Na+1:s′*Na] = Π[s,s′]*BasisMatrix(Vf[s].basis,Direct(),a′).vals[1][:]
        end
    end

    res = U(HH,c) .+ β.*EΦ*Vcoefs - Φ*Vcoefs
    Vcoefs′ = Φ\(U(HH,c) .+ β.*EΦ*Vcoefs)
    for s in 1:Nϵ
        Vf[s].coefs .= Vcoefs′[1+(s-1)*Na:s*Na]
    end
    return norm(res,Inf)
end;

"""
    solvebellman!(HH::HHModel)

Solves the bellman equation for given some initial 
value function V.
"""
function solvebellman!(HH::HHModel,tol=1e-8)
    #increases stability to iterate on the time dimension a few times
    diff = 1.
    for _ in 1:5
        iteratebellman_time!(HH)
    end
    while diff > tol
        #then use newtons method
        diff = iteratebellman_newton!(HH)
    end
    Vcoefs = vcat([HH.Vf[s].coefs for s in 1:HH.Nϵ]...)::Vector{Float64}
    HH.cf = computeoptimalconsumption(HH,Vcoefs)
end;


"""
    setupgrids_shocks!(AM::AiyagariModel)

Setup the grids and shocks for the aiyagari model
"""
function setupgrids_shocks!(AM::AiyagariModel,curv=2.)
    @unpack HH,Ia,N̄= AM
    @unpack a̲,a̅,Nϵ = HH
    setupgrids_shocks!(HH)
    #Normalize so that average labor supply is 1
    πstat = real(eigs(HH.Π',nev=1)[2])
    πstat ./= sum(πstat)
    HH.ϵ = HH.ϵ./dot(πstat,HH.ϵ)*N̄
    #Grid for distribution
    agrid = (a̅-a̲).*LinRange(0,1,Ia).^curv .+ a̲
    AM.z̄ = hcat(kron(ones(Nϵ),agrid),kron(1:Nϵ,ones(Ia)))
    AM.ω̄ = ones(Ia*Nϵ)/(Ia*Nϵ)
end;



@with_kw mutable struct AiyagariModel
    HH::HHModel = HHModel()

    #Production Parameters
    α::Float64 = 0.3
    δ::Float64 = 0.025
    Θ̄::Float64 = 1.

    #Moments to match/prices
    W̄::Float64 = 1.
    R̄::Float64 = 1.01
    K2Y::Float64 = 10.2 #capital to output ratio
    N̄::Float64 = 1.

    #Distribution Parameters
    Ia::Int = 1000 #Number of gridpoints for distribution
    z̄::Matrix{Float64} = zeros(0,0) #Gridpoints for the state variables
    ω̄::Vector{Float64} = zeros(0) #Fraction of agents at each grid level
    H::SparseMatrixCSC{Float64,Int64} = spzeros(Ia,Ia) #Transition matrix
end;

"""
    find_stationarydistribution!(AM::AiyagariModel,V)

Computes the stationary distribution 
"""
function find_stationarydistribution!(AM::AiyagariModel)
    @unpack Ia,z̄,HH,W̄,R̄ = AM
    @unpack ϵ,Π,Nϵ,cf,a̲,a̅ = HH

    a = z̄[1:Ia,1] #grids are all the same for all shocks
    c = hcat([cf[s](a) for s in 1:Nϵ]...) #consumption policy IaxNϵ
    a′ = R̄.*a .+ W̄.*ϵ' .- c #create a IaxNϵ grid for the policy rules
    
    #make sure we don't go beyond bounds.  Shouldn't bind if bmax is correct
    a′ = max.(min.(a′,a̅),a̲)
    
    Qa = BasisMatrix(Basis(SplineParams(a,0,1)),Direct(),reshape(a′,Ia*Nϵ)).vals[1]
    Q = spzeros(Ia*Nϵ,Ia*Nϵ)
    for s in 1:Nϵ
        Q[1+(s-1)*Ia:s*Ia,:] = kron(reshape(Π[s,:],1,:),Qa[1+(s-1)*Ia:s*Ia,:]) 
    end
    
    AM.H = Q'
    AM.ω̄ .= real(eigs(AM.H;nev=1)[2])[:]
    AM.ω̄ ./= sum(AM.ω̄) #normalize eigenvector
end;

"""
    capital_supply_demand(AM,R)

Compute the supply and demand for capital which prevails in
the stationary distribution at a given interest rate.
"""
function capital_supply_demand(AM,R)
    @unpack Θ̄,α,N̄,δ = AM
    AM.R̄ = R
    AM.HH.r̄ = R-1

    Y2K = (R-1+δ)/α
    K2N = (Y2K/Θ̄)^(1/(α-1))
    AM.W̄ = AM.HH.w̄ = (1-α)*Θ̄*K2N^α
    KD = K2N * N̄ 

    solvebellman!(AM.HH)
    find_stationarydistribution!(AM)
    KS = dot(AM.ω̄,AM.z̄[:,1])

    return [KS,KD]
end;

AM = AiyagariModel()
AM.HH.β = 0.99
setupgrids_shocks!(AM)

setupgrids_shocks!(HH,3.)
solvebellman!(HH)
setupgrids_shocks!(HH,3.)
@time solvebellman!(HH);


Rgrid = LinRange(1.,1.007,10)
KSKD = hcat([capital_supply_demand(AM,R) for R in Rgrid]...)
plot(layer(y=Rgrid,x=KSKD[1,:],color=["Capital Supplied"],Geom.line),
     layer(y=Rgrid,x=KSKD[2,:],color=["Capital Demanded"],Geom.line),
     Guide.ylabel("Gross Interest Rate"), Guide.xlabel("Capital"))

AM.R̄ = 1.01 #target a quarterly interest rate of 1%

function calibratesteadystate!(AM)
    @unpack Θ̄,α,N̄,K2Y,R̄ = AM
    AM.HH.r̄ = R̄ - 1
    Y2K = 1/K2Y
    AM.δ = α*Y2K + 1 - R̄ #matching capital to output ratio and interest rate gives depreciation rate
    K2N = (Y2K/Θ̄)^(1/(α-1)) #relationship between capital to output and capital to labor
    K̄ = K2N*N̄
    AM.W̄ = AM.HH.w̄ = (1-α)*Θ̄*K2N^α

    setupgrids_shocks!(AM)
    function βres(β)
        AM.HH.β=β
        solvebellman!(AM.HH)
        find_stationarydistribution!(AM)
        return dot(AM.ω̄,AM.z̄[:,1]) -K̄
    end

    Q̄ = 1/R̄
    fzero(βres,Q̄^2,Q̄^1.2)
end

AM = AiyagariModel()
calibratesteadystate!(AM)
plot(x=AM.z̄[AM.z̄[:,1].>1.,1],y=AM.ω̄[AM.z̄[:,1].>1.],Geom.bar,
    Guide.xlabel("Capital"), Guide.ylabel("Density"))