# Kalman-Filter-and-State-Space-Model


#Basic equition

yt = xtβt + εt  

βt = μ + Fβt-1 + vt

εt∼ iid N(0,R); vt~ iid N(0,Qt); E(εtvt) = 0

#Filtering 

Prediction:
βt|t-1 = μ + Fβt-1|t-1

Pt|t-1 = FPt-1|t-1 F' + Q

ηt|t-1 = yt - xtβt|t-1

ft|t-1 = xtPt|t-1x't + R

Updating
βt|t = βt|t-1 + Ktηt|t-1

Pt|t = Pt|t-a - KtxtPt|t-1

