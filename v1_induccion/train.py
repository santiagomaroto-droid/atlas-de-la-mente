import jax, jax.numpy as jnp, numpy as np, json, sys
V,T,d,H,NL=20,25,48,4,2; dh=d//H; N=int(sys.argv[1])
ks=jax.random.split(jax.random.PRNGKey(0),8)
n=lambda k,s,c: jax.random.normal(k,s)*c
P={'E':n(ks[0],(V,d),.1),'P':n(ks[1],(T,d),.1),'U':n(ks[2],(d,V),d**-.5)}
for i,nm in enumerate('QKVO'): P[nm]=n(ks[3+i],(NL,d,d),d**-.5)
def fwd(P,tok,abl=-1):
    B,t=tok.shape; x=P['E'][tok]+P['P'][:t]; mk=jnp.tril(jnp.ones((t,t),bool)); att=[]
    for l in range(NL):
        sp=lambda M:(x@M).reshape(B,t,H,dh).transpose(0,2,1,3)
        q,k,v=sp(P['Q'][l]),sp(P['K'][l]),sp(P['V'][l])
        a=jax.nn.softmax(jnp.where(mk,q@k.transpose(0,1,3,2)/dh**.5,-1e9),-1); att.append(a); o=a@v
        if abl>=0 and abl//H==l: o=o.at[:,abl%H].set(0)
        x=x+o.transpose(0,2,1,3).reshape(B,t,d)@P['O'][l]
    return x@P['U'],att
def batch(key,B,Lb=None):
    k1,k2=jax.random.split(key); R=jax.random.randint(k1,(B,T),0,V)
    Lb=jax.random.randint(k2,(B,),5,13) if Lb is None else jnp.full((B,),Lb)
    return jnp.take_along_axis(R,jnp.arange(T)[None,:]%Lb[:,None],1),Lb
def la(P,s,L,abl=-1):
    lg,_=fwd(P,s[:,:-1],abl); tg=s[:,1:]; m=(jnp.arange(T-1)[None,:]>=(L[:,None]-1))
    nll=-jnp.take_along_axis(jax.nn.log_softmax(lg),tg[...,None],-1)[...,0]
    return (nll*m).sum()/m.sum(),(((lg.argmax(-1)==tg)*m).sum()/m.sum())
tm=jax.tree_util.tree_map
@jax.jit
def step(P,m,v,key,t):
    s,L=batch(key,128); l,g=jax.value_and_grad(lambda P:la(P,s,L)[0])(P)
    m=tm(lambda m,g:.9*m+.1*g,m,g); v=tm(lambda v,g:.98*v+.02*g*g,v,g)
    return tm(lambda p,m,v:p-2e-3*(m/(1-.9**t))/(jnp.sqrt(v/(1-.98**t))+1e-8),P,m,v),m,v,l
ev=jax.jit(lambda P,key:la(P,*batch(key,512)))
m=tm(jnp.zeros_like,P); v=tm(jnp.zeros_like,P); key=jax.random.PRNGKey(1); curve=[]
for t in range(1,N+1):
    key,k=jax.random.split(key); P,m,v,l=step(P,m,v,k,float(t))
    if t%100==0:
        e=ev(P,jax.random.PRNGKey(99)); curve.append([t,round(float(e[0]),3),round(float(e[1]),3)])
        if t%500==0: print(curve[-1],flush=True)
s,L=batch(jax.random.PRNGKey(5),512,10); _,att=fwd(P,s[:,:-1]); att=[np.array(a) for a in att]
prev=[float(att[h//H][:,h%H,np.arange(1,24),np.arange(0,23)].mean()) for h in range(8)]
ind=[float(att[h//H][:,h%H,np.arange(10,24),np.arange(1,15)].mean()) for h in range(8)]
s2,L2=batch(jax.random.PRNGKey(7),2048); ab=[round(float(la(P,s2,L2,a)[1]),3) for a in range(-1,8)]
r=lambda a:np.round(np.array(a),3).tolist()
W={k:r(P[k]) for k in P}; W.update(nv=V,d=d,H=H,NL=NL)
tok=[3,7,1,9,4,3,7,1,9,4,3,7,1]; lg,_=fwd(P,jnp.array([tok])); pr=lg[0].argmax(-1).tolist()
json.dump(dict(W=W,curve=curve,prev=r(prev),ind=r(ind),base=ab[0],abl=ab[1:],tok=tok,pr=pr),open('mind2.json','w'))
print('prev',r(prev),'ind',r(ind),'base',ab[0],'abl',ab[1:],flush=True)
