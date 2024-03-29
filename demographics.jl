
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