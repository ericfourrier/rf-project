# Reinforcement Learning project
#Recurent Reinforcement Learning part

tab=read.table("returnO.txt",header=T)
X=as.matrix(tab[,2])
X=as.numeric(X)
featureNormalize<-function(X)
{
  mu=mean(X)
  sigma=sd(X)
  m=length(X)
  Xn=0*X
  for (i in 1:m){
    Xn[i]=(X[i]-mu)/sigma
  }
  return (Xn)
}
Xn=featureNormalize(X)
#Initialisation des entrées
#Initialisation de eta qui est le pas de la methode Gradient Descent
M=7
T=1000
Ne=100
sigma=0.05
eta=0.01

#Initialisation de F qui est la fonction du trader (décision)
#Initialisation de w
#Initialisation du Gradient
#Initialisation du Derivative recurent DFw
#Initialisation du Rendement R(t)

F=c(rep(0,T+1))
#w=c(rep(1/50,M+2))
w=c(rep(1,M+2));
DFw=c(rep(1,M+2))
Gradient=c(rep(0,M+2))
R=c(rep(0,T))
Shapiro=c(rep(0,Ne))
for (i in 1:Ne){
  for (t in 2:(T+1)){
    #x est le vecteur (1,r(t-M),....,r(t),F(t-1))
    x=c(1,Xn[(t-1):(t+M-2)],F[t-1])
    F[t]=tanh(w%*%x)
    R[t-1]=(X[t-1+M]*F[t-1]+1)*(1-sigma*abs(F[t]-F[t-1]))-1
  }
  #Calcul de Shapiro
  #Calcul de A et B
  A=sum(R)/T
  B=sum(R*R)/T 
  Shapiro[i]=A/sqrt(B-A^2)
  dSt_dA=-(A/(B-A^2)^(3/2))+(1/sqrt(B-A^2))
  dSt_dB=-A/(B-A^2)^(3/2)
  for (t in 1:T) {
    x=c(1,Xn[t:(t+M-1)],F[t])
    DFw_1=DFw
    DFw=(1-F[t+1]^2)*(x+w[M+2]*DFw_1)
    dB_dRt=2*R[t]/T
    dRt_dFt=-(1+F[t]*X[t+M])*sigma*sign(F[t+1]-F[t])
    dRt_dFt_1=X[t+M]*(1-sigma*abs(F[t+1]-F[t]))+(1+F[t]*X[t+M])*sigma*sign(F[t+1]-F[t])
    Gradient=Gradient+(dSt_dA*1/T+dSt_dB*dB_dRt)*(dRt_dFt*DFw+dRt_dFt_1*DFw_1)
  }
  w=w+eta*Gradient
  Gradient=c(rep(0,M+2))
}

# Test Set 

Ntest=50
Test_Set=c(rep(0,Ntest+M))
Test_Set=X[(T):(T+Ntest+M)]
Test_Set_n=Xn[(T):(T+Ntest+M)]
R=c(rep(0,Ntest))
F=c(rep(0,Ntest+1))
for (t in 2:(Ntest+1)){
  #x est le vecteur (1,r(t-M),....,r(t),F(t-1))
  x=c(1,Test_Set_n[(t):(t+M-1)],F[t-1])
  F[t]=tanh(w%*%x)
  R[t-1]=(Test_Set[t-1+M]*F[t-1]+1)*(1-sigma*abs(F[t]-F[t-1]))-1
}
W0=1000
Wealth=c(rep(0,Ntest+1))
Wealth[1]=W0
for ( t in 2:(Ntest+1)){
  Wealth[t]=Wealth[t-1]*(1+F[t-1]*Test_Set[t-1+M])*(1-sigma*abs(F[t]-F[t-1]))
}
plot(x=c(seq(1:Ne)),Shapiro)  
plot(x=c(seq(1:(Ntest+1))),Wealth)


     