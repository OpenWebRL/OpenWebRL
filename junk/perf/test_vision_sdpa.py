"""sdpa_thd must equal per-image full attention, forward and backward, in both layouts."""
import ast, math, types, sys
import torch
torch.manual_seed(0)

SRC="slime/backends/megatron_utils/vision_sdpa_patch.py"
tree=ast.parse(open(SRC).read())
keep=[n for n in tree.body if getattr(n,"name",None) in {"_offsets","sdpa_thd"} or (isinstance(n,(ast.Assign,ast.AnnAssign)) and any("_offsets_cache" in ast.dump(t) for t in ([n.target] if isinstance(n,ast.AnnAssign) else n.targets)))]
ns={"torch":torch,"F":torch.nn.functional,"math":math,"logging":types.SimpleNamespace(),"os":None,"dict":dict,"list":list,"tuple":tuple,"int":int}
ns["_MAX_HEAD_DIM"]=128
exec(compile(ast.Module(body=keep,type_ignores=[]),SRC,"exec"),ns)
sdpa_thd=ns["sdpa_thd"]

def reference(q,k,v,cu):
    t,h,d=q.shape; out=torch.zeros_like(q); offs=cu.tolist()
    for s,e in zip(offs[:-1],offs[1:]):
        qi,ki,vi=(x[s:e].transpose(0,1) for x in (q,k,v))          # [h,n,d]
        a=torch.softmax(qi@ki.transpose(1,2)/math.sqrt(d),-1)      # [h,n,n]
        out[s:e]=(a@vi).transpose(0,1)
    return out.reshape(t,h*d)

fails=0
for trial in range(20):
    n_img=torch.randint(1,9,(1,)).item()
    lens=torch.randint(5,90,(n_img,))
    cu=torch.cat([torch.zeros(1,dtype=torch.int32),lens.cumsum(0).int()])
    t,h,d=int(cu[-1]),4,24
    for layout in ("thd","t1hd"):
        q=torch.randn(t,h,d,dtype=torch.float32,requires_grad=True)
        k=torch.randn(t,h,d,dtype=torch.float32,requires_grad=True)
        v=torch.randn(t,h,d,dtype=torch.float32,requires_grad=True)
        qq,kk,vv=((x.unsqueeze(1) if layout=="t1hd" else x) for x in (q,k,v))
        out=sdpa_thd(qq,kk,vv,cu); ref=reference(q.detach(),k.detach(),v.detach(),cu)
        if out.shape!=(t,h*d) or not torch.allclose(out,ref,atol=1e-5):
            print(f"trial {trial} {layout}: forward mismatch, max err {(out-ref).abs().max():.2e}"); fails+=1; continue
        g=torch.randn_like(out)
        (out*g).sum().backward()
        q2,k2,v2=(x.detach().clone().requires_grad_(True) for x in (q,k,v))
        (reference(q2,k2,v2,cu)*g).sum().backward()
        for name,a,b in (("dq",q.grad,q2.grad),("dk",k.grad,k2.grad),("dv",v.grad,v2.grad)):
            if not torch.allclose(a,b,atol=1e-4):
                print(f"trial {trial} {layout}: {name} mismatch {(a-b).abs().max():.2e}"); fails+=1
# an image must not see another image
q=torch.randn(30,2,8); k=torch.randn(30,2,8); v=torch.randn(30,2,8); cu=torch.tensor([0,10,30],dtype=torch.int32)
o1=sdpa_thd(q,k,v,cu); k2=k.clone(); k2[10:]+=100; v2=v.clone(); v2[10:]+=100
o2=sdpa_thd(q,k2,v2,cu)
if not torch.allclose(o1[:10],o2[:10]): print("cross-image leakage!"); fails+=1
print(f"{fails} failures over 20 random packings x 2 layouts, forward+backward, plus isolation")
