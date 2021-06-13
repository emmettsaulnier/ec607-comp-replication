
# Replication of Conesa et al "Taxing Capital"
using Parameters, QuantEcon, BasisMatrices,LinearAlgebra,Optim,DataFrames,Gadfly,SparseArrays,Arpack,Roots,Optim, BasisMatrices

####################################################################
#### Create a struct for the model parameters and setup utility #### 
####################################################################

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
    ȳ::Float64 = 87000.0/37748.0

    # Grid sizes 
    ns::Int64 = 7 #X Also number of states?
    na::Int64 = 10 # asset grid
    nl::Int64 = 4  # leisure grid
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
    lgrid::Vector{Float64} = zeros(0)
    k::Int = 2 #type of interpolation
    Vf::Array{Interpoland} = Interpoland[] # value function given a
    Vlf::Array{Interpoland} = Interpoland[] # Value function given a,l
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

####################################################################
################ Interpolating the Value function  ################# 
####################################################################

"""
    optimalpolicy(para::NCParameters,Vprime,k)

Computes optimal policy for aprime when retired given continuation value and 
current period assets. Same for all types and states.
"""
function optimalPolicyRetired(TM::TaxMod,Vf′,a,j)
    @unpack τk,τc,SS,Tr,β,ψ,a̲,r = TM

    # Budget Constraint when retired
    c_ret(a′) = (SS+(1.0+r*(1.0-τk))*(a + Tr) - a′)/(1.0+τc)

    # Value function when retired, plugging in budget constraint for c
    Vfun(a′) = U(TM,c_ret(a′),0.) + β*ψ[j]*Vf′[1,1,j+1](a′) 
    
    # Optimizing over a′∈[a̲,a(c=0)] 
    ā = (SS+(1+r*(1-τk))*(a + Tr)) - 0.001
    result = optimize(Vfun,a̲,ā)
    aprime = result.minimizer
    V = -result.minimum
    
    return (aprime = aprime,V=V) 
end;

function optimalPolicyWork(TM::TaxMod,Vf′,a,s,ty,j,l)
    @unpack τk,τc,τp,Tr,β,ψ,a̲,r,w,ϵ,η,κ0,κ1,κ2,ȳ,Π = TM

    earn = w*ϵ[ty,j]*η[s]*l
    #earncap = r*(a + Tr)
    lab_tax = tax_gs(κ0,κ1,κ2,earn)
    #cap_tax = τk*earncap # DIFF THAN REAL CODE
    ss_tax =  τp*min(earn,ȳ)
    #mtaxes = marginal_tax_gs(κ0,κ1,κ2,earn) + τk*earncap 

    # Budget Constraint when working
    c_unret(a′) = (earn - ss_tax + (1+r*(1-τk))*(a + Tr) - lab_tax - a′)/(1+τc)
    
    # Objective function 
    Vfun(a′) = U(TM,c_unret(a′),l) + β*ψ[j]*sum(map(eta -> Vf′[eta,ty,j+1](a′) * Π[s,eta],1:tm.ns))
    
    # Maximize choosing a′
    ā = (earn - ss_tax + (1+r*(1-τk))*(a + Tr) - lab_tax)*(1+τc) - 0.001
    result = optimize(Vfun,a̲,ā)
    aprime = result.minimizer
    V = -result.minimum

    return (aprime = aprime,V=V) 
end;


"""
    iterateBellman(TM::TaxMod,Vf′)

Apply the bellman map given continuation value function Vprime
"""
function iterateBellman(TM::TaxMod,Vf′)
    @unpack ns,nty,J,SS,r,τk,Tr,τc,nl,lgrid,jr = TM
    
    # Setting up
    Vf = Array{Interpoland}(undef,ns,nty,J)
    anodes = nodes(Vf′[1,1,1].basis)
    len_node = length(anodes[1])
    V = zeros(len_node,ns,nty,J)
    #lf = zeros(len_node,ns,nty,J)
    Vr = zeros(len_node)
    aprime = zeros(len_node,ns,nty,J)
    Vl = zeros(len_node,nl)
    
    # Finding value function for given aprime and l
    for j in 1:J
        if j == J # Last period consume everything
            aprime[:,:,:,J] .= 0.
            for a in 1:len_node
                V[a,:,:,J] .= U(TM,(SS+(1.0+r*(1.0-τk))*(anodes[1][a] + Tr))/(1.0+τc),0.)
            end
            for s in 1:ns, ty in 1:nty
                basis = Vf′[s,ty,j].basis
                Vf[s,ty,j]= Interpoland(basis,V[:,s,ty,J])
            end
            #println("J done")
        elseif j >= jr # Retired folk
            basis = Vf′[1,1,j].basis
            Vr = [optimalPolicyRetired(TM,Vf′,a,j).V for a in nodes(basis)[1]]
            for s in 1:ns, ty in 1:nty 
                V[:,s,ty,J] = Vr
                Vf[s,ty,j]= Interpoland(basis,V[:,s,ty,J])
            end
            #println("Retired $j done")
        else # Working folk
            for s in 1:ns, ty in 1:nty
                basis = Vf′[s,ty,j].basis
                for l in 1:nl # find optimal policy given a and l 
                    Vl[:,l] = [optimalPolicyWork(TM,Vf′,a,s,ty,j,lgrid[l]).V for a in nodes(basis)[1]]
                end
                # for each a, choose l to maximize V
                for ap in 1:len_node
                    V[ap,s,ty,j] = findmax(Vl[ap,:])[1]
                end
                Vf[s,ty,j]= Interpoland(basis,V[:,s,ty,J])
            end
            #println("Working $j done")
        end
    end
    return Vf
end;

"""
    solvebellman!(TM::TaxMod)

Solves the bellman equation by iteration
"""
function solvebellman!(TM::TaxMod)
    @unpack ns,nty,J = TM

    diff = 1.
    counter = 1
    Vf′ = copy(TM.Vf)
    while diff > 1e-8 
        TM.Vf = iterateBellman(TM,Vf′);
        diff = maximum([norm(Vf′[s,ty,j].coefs - TM.Vf[s,ty,j].coefs,Inf) for s in 1:ns, ty in 1:nty, j in 1:J]);
        counter += 1;
        println("Iteration $counter, Diff: $diff")
        Vf′ = TM.Vf;
    end

end;


"""
    age_eff_pop(J, nty, nn)

Generates the age efficieny and population vectors
"""
function age_eff_pop(J, nty, nn)

    # Age-Efficiency Units from Hansen (1993)
    ephansen = zeros(J)

    ephansen[1]=1.0000
    ephansen[2]=1.0719
    ephansen[3]=1.1438
    ephansen[4]=1.2158
    ephansen[5]=1.2842
    ephansen[6]=1.3527
    ephansen[7]=1.4212
    ephansen[8]=1.4897
    ephansen[9]=1.5582
    ephansen[10]=1.6267
    ephansen[11]=1.6952
    ephansen[12]=1.7217
    ephansen[13]=1.7438
    ephansen[14]=1.7748
    ephansen[15]=1.8014
    ephansen[16]=1.8279
    ephansen[17]=1.8545
    ephansen[18]=1.8810
    ephansen[19]=1.9075
    ephansen[20]=1.9341
    ephansen[21]=1.9606
    ephansen[22]=1.9623
    ephansen[23]=1.9640
    ephansen[24]=1.9658
    ephansen[25]=1.9675
    ephansen[26]=1.9692
    ephansen[27]=1.9709
    ephansen[28]=1.9726
    ephansen[29]=1.9743
    ephansen[30]=1.9760
    ephansen[31]=1.9777
    ephansen[32]=1.9700
    ephansen[33]=1.9623
    ephansen[34]=1.9546
    ephansen[35]=1.9469
    ephansen[36]=1.9392
    ephansen[37]=1.9315
    ephansen[38]=1.9238
    ephansen[39]=1.9161
    ephansen[40]=1.9084
    ephansen[41]=1.9007
    ephansen[42]=1.8354
    ephansen[43]=1.7701
    ephansen[44]=1.7048 
    ephansen[45]=1.6396

    # Population Numbers from Bell and Miller (2002)
    pop = zeros(J)
    pop[1]=	197316
    pop[2]=	197141
    pop[3]=	196959
    pop[4]=	196770
    pop[5]=	196580
    pop[6]=	196392
    pop[7]=	196205
    pop[8]=	196019
    pop[9]=	195830
    pop[10]=195634
    pop[11]=195429
    pop[12]=195211
    pop[13]=194982
    pop[14]=194739
    pop[15]=194482
    pop[16]=194211
    pop[17]=193924
    pop[18]=193619
    pop[19]=193294
    pop[20]=192945
    pop[21]=192571
    pop[22]=192169
    pop[23]=191736
    pop[24]=191271
    pop[25]=190774
    pop[26]=190243
    pop[27]=189673
    pop[28]=189060
    pop[29]=188402
    pop[30]=187699
    pop[31]=186944
    pop[32]=186133
    pop[33]=185258
    pop[34]=184313
    pop[35]=183290
    pop[36]=182181
    pop[37]=180976
    pop[38]=179665
    pop[39]=178238
    pop[40]=176689
    pop[41]=175009
    pop[42]=173187
    pop[43]=171214
    pop[44]=169064
    pop[45]=166714
    pop[46]=164147
    pop[47]=161343
    pop[48]=158304
    pop[49]=155048
    pop[50]=151604
    pop[51]=147990
    pop[52]=144189
    pop[53]=140180
    pop[54]=135960
    pop[55]=131532
    pop[56]=126888
    pop[57]=122012
    pop[58]=116888
    pop[59]=111506
    pop[60]=105861
    pop[61]=99957
    pop[62]=93806
    pop[63]=87434
    pop[64]=80882
    pop[65]=74204
    pop[66]=67462
    pop[67]=60721
    pop[68]=54053
    pop[69]=47533
    pop[70]=41241
    pop[71]=35259
    pop[72]=29663
    pop[73]=24522
    pop[74]=19890
    pop[75]=15805
    pop[76]=12284
    pop[77]=9331
    pop[78]=6924
    pop[79]=5016
    pop[80]=3550
    pop[81]=2454
    
    # Labor efficiency 
    ep = zeros(nty,J)
    for i in 1:nty
        ep[i,1:J] = ephansen[1:J]
    end
    
    # WHat is measty? Probability of high and low ability
    measty = zeros(nty)
    measty[1:nty] .= 1.0/nty

    # Survival probabilities: surv(i)=prob(alive in i+1|alive in i)
    surv = zeros(J)
    for i in  1:(J-1)
    surv[i] = pop[i+1]/pop[i]
    end
    surv[J] = 0.0

    # Number of Agents in population
    Nu = zeros(J)
    Nu[1] = 1.0
    for i in 2:J
        Nu[i] = surv[i-1]*Nu[i-1]/(1.0+nn)	  
    end 

    # Fraction of agents in population
    mu = zeros(J)
    for i in 1:J
        mu[i] = Nu[i]/sum(Nu)
    end 

    # Total population
    topop=sum(Nu)

    return (ephansen = ephansen, pop = pop, surv = surv, Nu = Nu, mu = mu, ep = ep, measty = measty, topop = topop)
end

"""
    setup_grid!(TM::TaxMod, a̅ = 75, curve = 2.5)

Sets up grids, shocks, demographics 
"""
function setup_grid!(TM::TaxMod, a̅ = 75, curve = 2.5)
    @unpack na,nl,a̲,l̅,J,nty,nn,ns,ρ,var_η,var_α,Vf,cf,k,β = TM

    # Asset grid: with curvature
    agrid = TM.agrid = (a̅ - a̲).*LinRange(0,1,na).^curve .+ a̲;
    # Labor grid: evenly spaced 
    TM.lgrid = LinRange(0,l̅,nl);

    # Loading survival probs and age efficiency
    #ephansen,pop,surv,Nu,mu,ep,measty,topop = age_eff_pop(J,nty,nn)
    TM.ψ = age_eff_pop(J,nty,nn).surv;
    ephansen = age_eff_pop(J,nty,nn).ephansen;

    # Labor Productivity Markov Chain 
    mc = rouwenhorst(ns,ρ,var_η);
    TM.Π = mc.p;
    TM.η = exp.(mc.state_values);

    # Fixed effects 
    TM.ϵ = zeros(nty,J);
    TM.ϵ[1,1:J] = exp(-sqrt(var_α)) .* ephansen[1:J];
    TM.ϵ[2,1:J] = exp(sqrt(var_α)) .* ephansen[1:J];

    # Basis functions and interpolation
    abasis = Basis(SplineParams(agrid,0,k));
    anodes = nodes(abasis);
    Vf = Array{Interpoland}(undef,ns,nty,J);
    cons = zeros(length(anodes[1]),ns,nty,J);
    V = zeros(length(anodes[1]),ns,nty,J);
    

    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                cons[:,s,ty,j] .=  anodes[1].*(1-((J-j)/J));
                V[:,s,ty,j] = map(x->U(TM,x,0.5)./(1-β),cons[:,s,ty,j]);
                Vf[s,ty,j]= Interpoland(abasis,V[:,s,ty,j]);
            end
        end
    end

    TM.Vf  = Vf;
      #, c0 = cf)
end

tm = TaxMod();
setup_grid!(tm);
solvebellman!(tm)


plot(layer(x->tm.Vf[3,2,60].(x),0,1000,color=["s=1,ty=1,j=1 Vf"]),
    Guide.xlabel("x"),Guide.ylabel("f(x)"),Guide.colorkey(title=""))


# Tami's guess code
function guesses!(param::TTparam,N,SS)
    @unpack α,TFP,r̄, δ, ν,N̄,K̄,Ȳ,w̄,SSM = param 
    TT.N̄ = N
    TT.K̄ = K̄ = N*( α / (r̄+δ) )^(1.0/(1.0-α))        
    TT.Ȳ = Ȳ =  TFP*(K̄^α)*(N̄^(1.0-α))					     
    TT.w̄ =  w̄ = (1.0-α)*Ȳ/N̄
    TT.SSM = SSM = 2.3047578679665146* Ȳ / sum(ν)
    TT.SS = SS 
end



