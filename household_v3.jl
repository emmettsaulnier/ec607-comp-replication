using Base: Float64


using Optim
"""
    optimalpolicy(para::NCParameters,Vprime,k)

Computes optimal policy for aprime when retired given continuation value and 
current period assets. Same for all types and states.
"""
function optimalPolicy(TM::TaxMod,Vf′,a,s,ty,j)
    @unpack τk,τc,SS,Tr,β,ψ,a̲,r,na,ns,nty,J = TM

    # Budget Constraint when retired
    c_ret(aprm::Float64) = (SS+(1.0+r*(1.0-τk))*(a + Tr) - aprm)/(1.0+τc)
    # Budget Constraint when working
    #c_unret(a) = a
    # Objective function 
    if j == J
        aprime = 0.
        V = c_ret(aprime)
    else#if j >= jr    
        f_objective(apr) = U(TM,c_ret(apr),0.) + β*ψ[j]*Vf′[1,1,j](apr) 
        result = optimize(f_objective,a̲,(SS+(1+r*(1-τk))*(a + Tr)))
        aprime = result.minimizer
        V = -result.minimum
    #else 
    #  aprime[:,:,:,j] .= 1.
    #  V[:,:,:,j] .= 1.
    end

    return (aprime = aprime,V=V) 
end;

"""
    bellmanmap(Vprime,para::NCParameters)

Apply the bellman map given continuation value function Vprime
"""
function iterateBellman(TM::TaxMod,Vf′)
    @unpack ns,nty,J = TM
    Vf = Array{Interpoland}(undef,ns,nty,J)

    for s in 1:ns, ty in 1:nty, j in 1:J
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
    while diff > 1e-8 
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
    lgrid = LinRange(0,l̅,nl);

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
    Vf = Array{Interpoland}(undef,ns,nty,J);
    cf = Array{Interpoland}(undef,ns,nty,J);
    cons = zeros(na+1,ns,nty,J);
    V = zeros(na+1,ns,nty,J);
    anodes = nodes(abasis)[1];

    for s in 1:ns
        for ty in 1:nty
            for j in 1:J
                cons[:,s,ty,j] .=  anodes.*(1-((J-j)/J));
                V[:,s,ty,j] = map(x->U(TM,x,0.5)./(1-β),cons[:,s,ty,j]);
                Vf[s,ty,j]= Interpoland(abasis,V[:,s,ty,j]);
                cf[s,ty,j]= Interpoland(abasis,cons[:,s,ty,j]);
            end
        end
    end

     TM.Vf  = Vf; #, c0 = cf)
end

tm = TaxMod();
setup_grid!(tm);

aprime,V = solvebellman!(tm);