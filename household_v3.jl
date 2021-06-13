using Base: Float64


using Optim, BasisMatrices



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
    f_objective(a′) = U(TM,c_ret(a′),0.) + β*ψ[j]*Vf′[1,1,j+1](a′) 
    
    # Optimizing over a′∈[a̲,a(c=0)] 
    result = optimize(f_objective,a̲,(SS+(1+r*(1-τk))*(a + Tr)) - 0.001)
    aprime = result.minimizer
    V = -result.minimum
    
    return (aprime = aprime,V=V) 
end;


function optimalPolicyWork(TM::TaxMod,Vf′,a,s,ty,j)
    @unpack τk,τc,τss,SS,Tr,β,ψ,a̲,r,w,ϵ,η,κ0,κ1,κ2,ȳ,na,ns,nty,J = TM

    earn(l) = w*ϵ[ty,j]*η[s]*l
    earncap = r*(a + Tr)
    lab_tax(l) = tax_gs(κ0,κ1,κ2,earn(l))
    cap_tax = τk*earncap # DIFF THAN REAL CODE
    ss_tax(l) =  τss*min(earn(l),ȳ)
    mtaxes = marginal_tax_gs(κ0,κ1,κ2,earn(l)) + τk*earncap 

    # Budget Constraint when working
    c_unret(a′,l) = (earn(l) - ss_tax(l) + (1+r*(1-τk))(a + Tr) - lab_tax(l) - a′)/(1+τc)
    
    # Objective function 
    #--------------------- STOPPED HERE

    return (aprime = aprime,V=V) 
end;
"""
    bellmanmap(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function iterateBellman(TM::TaxMod,Vf′)
    @unpack ns,nty,J = TM
    
    # Setting up
    Vf = Array{Interpoland}(undef,ns,nty,J)
    len_node = length(nodes(Vf′[1,1,1].basis))
    V = zeros(len_node,ns,nty,J)
    aprime = zeros(len_node,ns,nty,J)
    
    # Finding value function for given aprime and l
    for j in 1:J
        if j == J # Last period consume everything
            aprime[:,:,:,J] = 0.
            V[:,:,:,J] = c_ret(aprime)
        elseif j >= jr # Retired folk
            basis = Vf′[1,1,j].basis
            Vr = [optimalPolicyRetired(TM,Vf′,a,j).V for a in nodes(basis)[1]]
            for s in 1:ns, ty in 1:nty 
                V[:,s,ty,J] = Vr
            end
        else 
            for s in 1:ns, ty in 1:nty
                basis = Vf′[s,ty,j].basis
                #------------------------ STOPPED HERE 
                #V[:,s,ty,J] = [optimalPolicyWork(TM,Vf′,a,j).V for a in nodes(basis)[1]]
            end
        end
        # Interpoland 
        basis = Vf′[s,ty,j].basis
        V = [optimalPolicy(TM,Vf′,a,s,ty,j).V for a in nodes(basis)[1]]
        Vf[s,ty,j]= Interpoland(basis,V)
    end

    return Vf
end;

"""
    solvebellman(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given Vjr 
"""
function solvebellman!(TM::TaxMod)
    @unpack ns,nty,J = TM

    diff = 1.
    Vf′ = copy(TM.Vf)
    for i in 1:10 #while diff > 1e-8 
        TM.Vf = iterateBellman(TM,Vf′)
        diff = maximum([norm(Vf′[s,ty,j].coefs - TM.Vf[s,ty,j].coefs,Inf) for s in 1:ns, ty in 1:nty, j in 1:J])
        println(diff)
        Vf′ = TM.Vf
    end

end;

"""
    getV0(para::NCParameters)

Initializes V0(k) = 0 using the kgrid of para
"""
function setup_grid!(TM::TaxMod, a̅ = 75, curve = 2.5)
    @unpack na,nl,a̲,l̅,J,nty,nn,ns,ρ,var_η,var_α,Vf,cf,k,β = TM

    # Asset grid: with curvature
    agrid = TM.agrid = (a̅ - a̲).*LinRange(0,1,na).^curve .+ a̲;
    # Labor grid: evenly spaced 
    lgrid = TM.lgrid = LinRange(0,l̅,nl);

    # Loading demographic data from other script
    include("demographics.jl");
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
    lbasis = Basis(SplineParams(lgrid,0,k));
    basis = Basis(abasis,lbasis)
    Vlf = Array{Interpoland}(undef,undef,ns,nty,J);
    Vf = Array{Interpoland}(undef,,ns,nty,J);
    #cf = Array{Interpoland}(undef,ns,nty,J);
    X = nodes(basis)[1];
    N = size(X,1)
    cons = zeros(na+1,ns,nty,J);
    V = zeros(na+1,ns,nty,J);
    

    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                cons[:,s,ty,j] .=  anodes.*(1-((J-j)/J));
                V[:,s,ty,j] = map(x->U(TM,x,0.5)./(1-β),cons[:,s,ty,j]);
                Vlf[s,ty,j]= Interpoland(basis,V[:,s,ty,j]);
                Vf[s,ty,j]= Interpoland(abasis,V[:,s,ty,j]);
                cf[s,ty,j]= Interpoland(basis,cons[:,s,ty,j]);
            end
        end
    end

     TM.Vf  = Vf;
     TM.Vlf  = Vlf;
      #, c0 = cf)
end

tm = TaxMod();
setup_grid!(tm);

aprime,V = solvebellman!(tm);




for l in 1:nl 
    V[:,l] = [optimalPolicyRetired(TM,Vf′,a,j).V for a in nodes(basis)[1]]
end 


function get_l(lgrid, Vf, a) # Vf(a,l)

lgrid
abasis = Basis(SplineParams(agrid,0,k));

for a in 1:length(agrid)
    val[a,:] = Vf.(a,lgrid)
    maxV[a] = findmax(val)
    maximizedL[a] = maximizer(val)
end 

V = Interpoland(abasis,maxV)


# Guess V(a)
# find optimal policy for grida and gridl
# choose l that maximizes for each a using findmax
# interpolate V(a)


