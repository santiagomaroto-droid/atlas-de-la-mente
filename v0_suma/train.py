import numpy as np, json, sys
N=int(sys.argv[1]); rng=np.random.default_rng(0)
p,d,H=43,32,128
A,B=np.meshgrid(np.arange(p),np.arange(p),indexing='ij'); A=A.ravel(); B=B.ravel(); Y=(A+B)%p
idx=rng.permutation(p*p); ntr=int(.55*p*p); tr,te=idx[:ntr],idx[ntr:]
P={'E':rng.normal(0,.5,(p,d)),'W1':rng.normal(0,1,(2*d,H))/np.sqrt(2*d),'b1':np.zeros(H),'W2':rng.normal(0,1,(H,p))/np.sqrt(H),'b2':np.zeros(p)}
m={k:np.zeros_like(v) for k,v in P.items()}; s={k:np.zeros_like(v) for k,v in P.items()}
def fwd(ix):
    h0=np.concatenate([P['E'][A[ix]],P['E'][B[ix]]],1); z=h0@P['W1']+P['b1']; r=np.maximum(z,0)
    return h0,z,r,r@P['W2']+P['b2']
def acc(ix): return float((fwd(ix)[3].argmax(1)==Y[ix]).mean())
lr,wd,be1,be2=2e-3,1.0,.9,.98; curve=[]
for t in range(1,N+1):
    h0,z,r,lg=fwd(tr); g=np.exp(lg-lg.max(1,keepdims=True)); g/=g.sum(1,keepdims=True)
    g[np.arange(len(tr)),Y[tr]]-=1; g/=len(tr)
    dz=(g@P['W2'].T)*(z>0); dh=dz@P['W1'].T; gE=np.zeros_like(P['E'])
    np.add.at(gE,A[tr],dh[:,:d]); np.add.at(gE,B[tr],dh[:,d:])
    G={'W2':r.T@g,'b2':g.sum(0),'W1':h0.T@dz,'b1':dz.sum(0),'E':gE}
    for k in P:
        m[k]=be1*m[k]+(1-be1)*G[k]; s[k]=be2*s[k]+(1-be2)*G[k]**2
        P[k]-=lr*((m[k]/(1-be1**t))/(np.sqrt(s[k]/(1-be2**t))+1e-8)+wd*P[k])
    if t%200==0:
        curve.append([t,round(acc(tr),4),round(acc(te),4)])
        if t%2000==0: print(curve[-1],flush=True)
F=(np.abs(np.fft.fft(P['E'],axis=0))**2).sum(1); sp=2*F[1:p//2+1]; sp=sp/sp.sum()
order=np.argsort(-sp); keys=sorted(int(k)+1 for k in order if sp[k]>=0.08)[:6]
allx=np.arange(p*p); pred=fwd(allx)[3].argmax(1)
c=np.arange(p); sc=sum(np.cos(2*np.pi*k*((A+B)[:,None]-c[None,:])/p) for k in keys); alg=sc.argmax(1)
mask=np.zeros(p*p,int); mask[tr]=1
out=dict(p=p,curve=curve,spec=[round(float(x),4) for x in sp],keys=keys,
 pred=''.join(chr(48+int(x)) for x in pred),mask=''.join(map(str,mask)),
 net_acc=float((pred==Y).mean()),net_test=acc(te),alg_acc=float((alg==Y).mean()),agree=float((alg==pred).mean()),
 share=float(sp[[k-1 for k in keys]].sum()))
json.dump(out,open('atlas.json','w'))
print({k:v for k,v in out.items() if k not in('curve','spec','pred','mask')})
