

function household(TM::TaxModel)
    @unpack na,ns,nty,J,κ0,κ1,κ2,τk,a̲  = TM

#INPUTS NEEDED: agrid,Tr
Vfun = zeros(na,ns,nty,J)
lfun = zeros(na,ns,nty,J)
cfun = zeros(na,ns,nty,J)
afun = zeros(na,ns,nty,J)
vpfun = zeros(na,ns,nty,J)

# Last generation
for a in 1:na 
    # Calculating earnings and taxes
    earnl = 0.0
    earncap = r*(agrid[a] + Tr)
    lab_tax = tax_gs(κ0,κ1,κ2,earnl)
    cap_tax = τk*earncap # DIFF THAN REAL CODE
    mtaxes = marginal_tax_gs(κ0,κ1,κ2,earnl) + τk*earncap 

    # Consume all assets and supply no labor
    l = 0.0 
    c = (SS+(1.0+r(1-τk))*(agrid[a]+Tr))/(1.0+τc) 
    
    # Setting function values
	Vfun[a,1:ns,1:nty,J] = U(c,l)
	lfun[a,1:ns,1:nty,J] = l
	cfun[a,1:ns,1:nty,J] = c
	afun[a,1:ns,1:nty,J] = 0.0
	vpfun[a,1:ns,1:nty,J]=(1.0+r*(1.0-τk))*marginal_utility(con,lab)
end

# Rest of time period 
for j in 1:(J-1)
    for ty in 1:nty
        for s in 1:ns
            for a in 1:na
                # Retirement periods
                if j >= jr
                    # Don't work in retirement
                    lfun[a,s,ty,j] = 0.0
                    # Bounds on capital
                    lbound = a̲
                    ubound = agrid[na] # assets will be decreasing 
                    # earnings
                    earnl = 0.0
                    earncap = r*(agrid[a]+Tr)
                    # Taxes
                    lab_tax = tax_gs(κ0,κ1,κ2,earnl)
                    cap_tax = τk*earncap # DIFF THAN REAL CODE
                    mtaxes = marginal_tax_gs(κ0,κ1,κ2,earnl) + τk*earncap 

                    lab=0.0
			        conl = (SS+(1.0+r(1-τk))*(agrid[a]+Tr) - lbound)/(1.0+τc) 
			        conh = (SS+(1.0+r(1-τk))*(agrid[a]+Tr) - ubound)/(1.0+τc) 

                    