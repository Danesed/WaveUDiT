import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wavedit.models.direct_unet import DirectInpaintModel
from wavedit.data.brats_inpaint import (BraTSInpaintingDataset, _load_nii, _normalise_t1,
                                        _to_axial_first, _pad_volume)
from wavedit.evaluation.visualization import _mask_centroid_axis, _take_plane_rot, _to01

DEV="cuda"; TS=(160,256,256); SEED=42
TRN=os.environ.get("BRATS_DATA", "data/BraTS-Local-Synthesis-Training")
CACHE=os.environ.get("BRATS_CACHE", "cache/inpaint")
CKPT=os.environ.get("WUDIT_CKPT","checkpoints_inpaint/WaveDiT_inpaint_contra_wudit_base128_clean/unet_best_aggregate.pth")   # WaveUDiT (wavelet)
DST=os.environ.get("FIG_DST","PAPER/overleaf_WaveUDiT/figures/results_qualitative")  # basename; .png + .pdf
# NOTE: wudit is still training -- re-render from the final checkpoint before submission.

# ---- model ----
c=torch.load(CKPT,map_location=DEV,weights_only=False); cfg=c.get("config",{}); g=lambda k,d:cfg.get(k,d)
m=DirectInpaintModel(base=g("base",32),levels=g("levels",4),dropout=g("dropout",0.2),arch=g("arch","udit"),
    udit_blocks=g("udit_blocks",2),udit_downsample=g("udit_downsample",2),udit_d_head=g("udit_d_head",64),
    in_contra=g("in_contra",False),in_sdf=g("in_sdf",False),sharpen_head=g("sharpen_head",False),
    sharpen_sigma=g("sharpen_sigma",1.0),nonlocal_healthy=g("nonlocal_healthy",False),
    sdf_attn_bias=g("sdf_attn_bias",False),deco_head=g("deco_head",False),deco_channels=g("deco_channels",64),
    deco_blocks=g("deco_blocks",3),udit_tokenrep=g("udit_tokenrep",False),
        hdit_patch=g("hdit_patch",8),contra_attn=g("contra_attn",False)).to(DEV)
sd=next((c[k] for k in ["model","ema","state_dict"] if isinstance(c,dict) and k in c and isinstance(c[k],dict)),c)
m.load_state_dict(sd,strict=False); m.eval()

# ---- SAME case as the viz: cases[val_indices[0]] with seed-42 shuffle ----
ds=BraTSInpaintingDataset(root=TRN,target_shape=TS,train_mode=True,augment=False,seed=SEED,cache_dir=CACHE)
n_total=len(ds.cases); rng=np.random.default_rng(SEED); perm=np.arange(n_total); rng.shuffle(perm)
case_dir=str(ds.cases[perm[0]]); cid=os.path.basename(case_dir)
print("first-val case:",cid,"(index",int(perm[0]),"of",n_total,")")

vx=_load_nii(f"{case_dir}/{cid}-t1n-voided.nii.gz")[0]
mx=(_load_nii(f"{case_dir}/{cid}-mask.nii.gz")[0]>0).astype(np.float32)
gx=_load_nii(f"{case_dir}/{cid}-t1n.nii.gz")[0]
vn,maxv=_normalise_t1(vx); gn=_normalise_t1(gx,max_v=maxv)[0]
v =_pad_volume(_to_axial_first(vn),TS,mode="reflect",value=0.0)
mk=(_pad_volume(_to_axial_first(mx),TS,mode="constant",value=0.0)>0.5).astype(np.float32)
gt=_pad_volume(_to_axial_first(gn),TS,mode="reflect",value=0.0)
with torch.no_grad():
    vt=torch.from_numpy(v)[None,None].to(DEV); mt=torch.from_numpy(mk)[None,None].to(DEV)
    pr=m.predict_mirror_consistent(vt,mt).squeeze().cpu().numpy()   # composited, mirror-TTA

# ---- same planes/levels/orientation as the visualizer ----
planes=[("axial",0),("coronal",1),("sagittal",2)]
cols=["Ground truth","Voided",r"Prediction","$|\\mathrm{Pred}-\\mathrm{GT}|$"]
plt.rcParams.update({"text.color":"white","pdf.fonttype":42})
fig,axes=plt.subplots(3,4,figsize=(11,7.6),facecolor="black")
fig.subplots_adjust(left=0.055,right=0.93,top=0.94,bottom=0.01,wspace=0.02,hspace=0.02)
im_err=None
def pad_to(a,H,W):
    h,w=a.shape; ph,pw=H-h,W-w
    return np.pad(a,((ph//2,ph-ph//2),(pw//2,pw-pw//2)),constant_values=0.0)
# pass 1: crop each plane to its brain bbox; track a common canvas so all 3 views share one scale
prep=[]; Hm=Wm=0
for plane,axis in planes:
    idx=_mask_centroid_axis(mk,axis=axis)
    gt_s=_to01(_take_plane_rot(gt,plane,idx)); v_s=_to01(_take_plane_rot(v,plane,idx))
    pr_s=_to01(_take_plane_rot(pr,plane,idx)); mk_s=(_take_plane_rot(mk,plane,idx)>0.5).astype(float)
    ys,xs=np.where(gt_s>0.05); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    cr=lambda a:a[y0:y1,x0:x1]
    t=[cr(gt_s),cr(v_s),cr(pr_s),cr(np.abs(pr_s-gt_s)),cr(mk_s)]
    Hm=max(Hm,t[0].shape[0]); Wm=max(Wm,t[0].shape[1]); prep.append((plane,idx,t))
# pass 2: pad every plane to (Hm,Wm) -> uniform brain scale, then render
for i,(plane,idx,t) in enumerate(prep):
    gt_c,v_c,pr_c,err_c,mk_c=[pad_to(a,Hm,Wm) for a in t]
    vm=float(np.percentile(gt_c[gt_c>0.02],99)) if (gt_c>0.02).any() else 1.0
    panels=[(gt_c,"gray",vm),(v_c,"gray",vm),(pr_c,"gray",vm),
            (err_c,"magma",max(np.percentile(t[3],99.5),1e-3))]
    for j,(img,cmap,hi) in enumerate(panels):
        ax=axes[i,j]; ax.set_facecolor("black")
        h=ax.imshow(img,cmap=cmap,vmin=0,vmax=hi,aspect="equal")
        if j==3: im_err=h
        if j in (1,2,3): ax.contour(mk_c,levels=[0.5],colors="red",linewidths=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if i==0: ax.set_title(cols[j],fontsize=18,pad=7)
    axes[i,0].set_ylabel(f"{plane} @{idx}",fontsize=16,rotation=90,labelpad=5)
cax=fig.add_axes([0.945,0.08,0.014,0.80]); cb=fig.colorbar(im_err,cax=cax)
cb.set_label("abs. error",color="white",fontsize=15); cb.ax.yaxis.set_tick_params(color="white",labelsize=13)
plt.setp(cb.ax.get_yticklabels(),color="white")
for _ext in ("png","pdf"):
    fig.savefig(f"{DST}.{_ext}",facecolor="black",bbox_inches="tight",dpi=150)
print("saved",DST+".png/.pdf")
