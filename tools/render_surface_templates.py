#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, pathlib, re
import numpy as np

ENTITY_RE=re.compile(r"^#(\d+)=(.*);$")
POINT_RE=re.compile(r"CARTESIAN_POINT\('[^']*',\(([^)]*)\)\)")
SURF_RE=re.compile(r"B_SPLINE_SURFACE_WITH_KNOTS\('[^']*',(\d+),(\d+),")

def balanced(s,pos):
    d=0; quoted=False; i=pos
    while i<len(s):
        ch=s[i]
        if ch=="'":
            if quoted and i+1<len(s) and s[i+1]=="'":
                i+=2; continue
            quoted=not quoted
        elif not quoted:
            if ch=="(": d+=1
            elif ch==")":
                d-=1
                if d==0: return s[pos:i+1],i+1
        i+=1
    raise ValueError("unbalanced")

def split_top(s):
    out=[]; start=0; d=0; quoted=False; i=0
    while i<len(s):
        ch=s[i]
        if ch=="'":
            if quoted and i+1<len(s) and s[i+1]=="'":
                i+=2; continue
            quoted=not quoted
        elif not quoted:
            if ch=="(": d+=1
            elif ch==")": d-=1
            elif ch=="," and d==0:
                out.append(s[start:i]); start=i+1
        i+=1
    out.append(s[start:])
    return out

def nums(s, as_int=False):
    s=s.strip()
    assert s[0]=="(" and s[-1]==")"
    vals=[x for x in s[1:-1].split(",") if x]
    return [int(x) for x in vals] if as_int else [float(x) for x in vals]

def parse_surface(body, points):
    m=SURF_RE.match(body)
    if not m: return None
    du,dv=map(int,m.groups())
    net,end=balanced(body,m.end())
    rows=re.findall(r"\((#[0-9]+(?:,#[0-9]+)*)\)",net)
    ids=[[int(x[1:]) for x in row.split(",")] for row in rows]
    cp=np.array([[points[i] for i in row] for row in ids],dtype=float)
    tail=body[end:]
    if tail.startswith(","): tail=tail[1:]
    args=split_top(tail)
    # form,u_closed,v_closed,self_intersect,u_mults,v_mults,u_knots,v_knots,knot_spec
    um=nums(args[4],True); vm=nums(args[5],True)
    uk=nums(args[6]); vk=nums(args[7])
    U=np.array([v for v,mul in zip(uk,um) for _ in range(mul)],dtype=float)
    V=np.array([v for v,mul in zip(vk,vm) for _ in range(mul)],dtype=float)
    return dict(du=du,dv=dv,cp=cp,U=U,V=V,um=um,vm=vm,uk=uk,vk=vk)

def basis_all(u,p,U,n):
    # valid domain is [U[p], U[n]]
    if abs(u-U[n]) <= max(1.0,abs(U[n]))*1e-14:
        N=np.zeros(n); N[-1]=1.0; return N
    N=np.zeros(n)
    for i in range(n):
        if U[i] <= u < U[i+1]: N[i]=1.0
    for k in range(1,p+1):
        nxt=np.zeros(n)
        for i in range(n):
            a=0.0;b=0.0
            da=U[i+k]-U[i]
            if da!=0: a=(u-U[i])/da*N[i]
            if i+1<n:
                db=U[i+k+1]-U[i+1]
                if db!=0: b=(U[i+k+1]-u)/db*N[i+1]
            nxt[i]=a+b
        N=nxt
    return N

def sample(s,nu=45,nv=35):
    cp=s["cp"]; n,m=cp.shape[:2]
    us=np.linspace(s["U"][s["du"]],s["U"][n],nu)
    vs=np.linspace(s["V"][s["dv"]],s["V"][m],nv)
    pts=np.empty((nu,nv,3))
    for a,u in enumerate(us):
        Nu=basis_all(u,s["du"],s["U"],n)
        for b,v in enumerate(vs):
            Nv=basis_all(v,s["dv"],s["V"],m)
            pts[a,b]=np.einsum("i,j,ijc->c",Nu,Nv,cp)
    return pts

def equal_axes(ax, pts, pad=1.05):
    q=pts.reshape(-1,3)
    lo=q.min(0);hi=q.max(0); mid=(lo+hi)/2; span=max(float((hi-lo).max()),1e-9)*pad
    ax.set_xlim(mid[0]-span/2,mid[0]+span/2)
    ax.set_ylim(mid[1]-span/2,mid[1]+span/2)
    ax.set_zlim(mid[2]-span/2,mid[2]+span/2)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("stats")
    ap.add_argument("out")
    args=ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw={};points={}
    for line in pathlib.Path(args.step).open(errors="replace"):
        m=ENTITY_RE.match(line.strip())
        if not m:continue
        i=int(m.group(1)); b=m.group(2); raw[i]=b
        q=POINT_RE.fullmatch(b)
        if q:points[i]=np.array([float(x) for x in q.group(1).split(",")])

    st=json.load(open(args.stats))
    reps=[x["canonical"] for x in st["families"] if x["family_size"]==82]
    surfaces={i:parse_surface(raw[i],points) for i in reps}

    fig=plt.figure(figsize=(16,16),dpi=150)
    for k,sid in enumerate(reps,1):
        ax=fig.add_subplot(4,4,k,projection="3d")
        s=surfaces[sid]; P=sample(s)
        ctr=P.mean((0,1))
        Q=P-ctr; C=s["cp"]-ctr
        ax.plot_surface(Q[:,:,0],Q[:,:,1],Q[:,:,2],rstride=2,cstride=2,linewidth=0,alpha=0.8)
        for i in range(C.shape[0]):
            ax.plot(C[i,:,0],C[i,:,1],C[i,:,2],marker=".",markersize=2,linewidth=0.5)
        for j in range(C.shape[1]):
            ax.plot(C[:,j,0],C[:,j,1],C[:,j,2],marker=".",markersize=2,linewidth=0.5)
        equal_axes(ax,Q)
        ax.view_init(elev=28,azim=-55)
        ax.set_title(f"#{sid}  {C.shape[0]}×{C.shape[1]}")
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    fig.suptitle("PCIe 164-pin: 16 strict B-spline surface templates (centered)")
    fig.tight_layout()
    fig.savefig(args.out,bbox_inches="tight")
    plt.close(fig)

    # second view chosen to expose mirror/side differences
    out2=str(pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem+"-side.png"))
    fig=plt.figure(figsize=(16,16),dpi=150)
    for k,sid in enumerate(reps,1):
        ax=fig.add_subplot(4,4,k,projection="3d")
        s=surfaces[sid]; P=sample(s); ctr=P.mean((0,1));Q=P-ctr
        ax.plot_surface(Q[:,:,0],Q[:,:,1],Q[:,:,2],rstride=2,cstride=2,linewidth=0,alpha=0.85)
        equal_axes(ax,Q)
        ax.view_init(elev=5,azim=-90)
        ax.set_title(f"#{sid}")
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    fig.suptitle("Same templates, near-side view")
    fig.tight_layout()
    fig.savefig(out2,bbox_inches="tight")
    print(json.dumps({"reps":reps,"out":args.out,"side":out2},indent=2))

if __name__=="__main__": main()
