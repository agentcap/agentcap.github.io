"""Publication-ready figures generated exclusively from measured summary data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .report import LABELS

COLORS = {"bm25":"#a8a29e","dense":"#78716c","hybrid":"#292524",
          "ce_tinybert":"#7dd3fc","ce_minilm":"#0284c7","llm":"#7c3aed",
          "jev":"#059669","decisions":"#ea580c"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":13,
    "axes.labelsize":10,"figure.facecolor":"#fcfaf6","axes.facecolor":"#fcfaf6",
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#d6d3d1",
    "grid.color":"#e7e5e4","savefig.facecolor":"#fcfaf6","text.color":"#292524",
    "axes.labelcolor":"#292524","xtick.color":"#57534e","ytick.color":"#57534e"})

def save(fig,name):
    folder=Path("blog/assets")
    folder.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","svg"):
        fig.savefig(folder/f"{name}.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)

def main():
    s=json.loads(Path("results/summary.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True)
    for ax,(name,d) in zip(axes,s["datasets"].items()):
        methods=list(d["paired"])
        ys=np.arange(len(methods))
        scores=[d["paired"][m]["ndcg10"] for m in methods]
        errors=np.array([[v-d["paired"][m]["ndcg10_ci"]["low"] for m,v in zip(methods,scores)],
                         [d["paired"][m]["ndcg10_ci"]["high"]-v for m,v in zip(methods,scores)]])
        ax.barh(ys,scores,color=[COLORS[m] for m in methods],height=.65)
        ax.errorbar(scores,ys,xerr=errors,fmt="none",ecolor="#44403c",capsize=2,lw=1)
        ax.set_yticks(ys,[LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_title(f"{name.upper()} · {d['paired_test_queries']} paired queries",loc="left",fontweight="bold")
        ax.set_xlabel("nDCG@10 · higher is better")
        ax.set_xlim(0,1)
        ax.xaxis.grid(True,alpha=.6)
        ax.set_axisbelow(True)
        for y,v in zip(ys,scores):
            ax.text(min(.95,v+.09),y,f"{v:.3f}",va="center",fontsize=9)
    fig.suptitle("Hybrid retrieval and five ways to rerank it",fontsize=18,fontweight="bold",x=.03,ha="left",y=1.04)
    fig.text(.03,-.01,"Full corpora. Rerank top 50 hybrid candidates. Error bars: 95% query-bootstrap intervals.",color="#57534e")
    fig.tight_layout(w_pad=3)
    save(fig,"ranking-quality")

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(name,d) in zip(axes,s["datasets"].items()):
        for i,(m,r) in enumerate(d["paired"].items()):
            ax.scatter(r["latency_p50_ms"],r["ndcg10"],s=95,color=COLORS[m],edgecolor="white",zorder=3)
            offset={"bm25":(-6,-14),"dense":(6,11) if name=="scifact" else (6,-13),
                    "hybrid":(-6,10),"ce_tinybert":(5,-18),"ce_minilm":(5,8),
                    "llm":(5,8),"jev":(5,-12),"decisions":(5,8)}[m]
            label={"ce_tinybert":"TinyBERT","ce_minilm":"MiniLM"}.get(m,LABELS[m])
            ax.annotate(label,(r["latency_p50_ms"],r["ndcg10"]),xytext=offset,
                        textcoords="offset points",fontsize=8,ha="right" if m in ("bm25","hybrid") else "left")
        ax.set_xscale("log")
        ax.set_title(name.upper(),loc="left",fontweight="bold")
        ax.set_xlabel("Median retrieval + reranking latency (ms, log scale)")
        ax.set_ylabel("nDCG@10")
        ax.grid(alpha=.5)
        lo,hi=ax.get_ylim()
        ax.set_ylim(lo-.025,hi+.025)
        latencies=[r["latency_p50_ms"] for r in d["paired"].values()]
        ax.set_xlim(min(latencies)*.45,max(latencies)*5)
    fig.suptitle("Quality and latency are separate decisions",fontsize=18,fontweight="bold",x=.03,ha="left",y=1.03)
    fig.text(.03,-.02,"Local: RTX 4070, warm execution. APIs: measured request wall time, concurrency 2. No startup/indexing.",color="#57534e")
    fig.tight_layout(w_pad=3)
    save(fig,"quality-latency")

    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,(name,d) in enumerate(s["datasets"].items()):
        for row,backend in enumerate(("cross_encoder","llm")):
            ax=axes[row,col]
            policies=d["gates"].get(backend,{})
            keys=[k for k in policies if k.endswith("_calibrated")]
            target=d["paired"]["ce_minilm" if backend=="cross_encoder" else "llm"]
            labels=["Always rerank","Hybrid only"]+[k.split("_")[0].title() for k in keys]
            values=[target["mean_latency_ms"],d["paired"]["hybrid"]["mean_latency_ms"]]+[policies[k]["mean_latency_ms"] for k in keys]
            quality=[target["ndcg10"],d["paired"]["hybrid"]["ndcg10"]]+[policies[k]["ndcg10"] for k in keys]
            ax.barh(np.arange(len(labels)),np.array(values)/target["mean_latency_ms"],
                    color=["#0284c7","#292524"]+[COLORS.get(k.split("_")[0],"#a8a29e") for k in keys],height=.62)
            ax.axvline(1,color="#78716c",ls="--",lw=1)
            ax.set_yticks(np.arange(len(labels)),labels)
            ax.invert_yaxis()
            ax.set_xlabel("Mean latency / always-rerank · lower is better")
            ax.set_title(f"{name.upper()} · {'MiniLM' if backend=='cross_encoder' else 'GPT-6 Luna'}",loc="left",fontweight="bold")
            ax.set_xlim(0,max(1.5,max(values)/target["mean_latency_ms"]*1.35))
            for y,v,q in zip(range(len(labels)),values,quality):
                ax.text(v/target["mean_latency_ms"]+.03,y,f"nDCG {q:.3f}",va="center",fontsize=9)
            ax.xaxis.grid(alpha=.5)
            ax.set_axisbelow(True)
    fig.suptitle("A gate must pay for itself",fontsize=20,fontweight="bold",x=.03,ha="left",y=1.01)
    fig.text(.03,-.01,"Thresholds chosen on disjoint calibration queries. Gate time included. Policy timing is a serial replay estimate.",color="#57534e")
    fig.tight_layout(h_pad=3,w_pad=3)
    save(fig,"gating-latency")
    print("Saved 3 figures as PNG and SVG.")

if __name__=="__main__":
    main()
