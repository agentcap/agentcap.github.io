"""Package source, article and numeric results; exclude credentials, corpora and models."""
from pathlib import Path
import argparse, zipfile
from .data import dump
import hashlib

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--credential-file",action="append",default=[])
    args=parser.parse_args()
    root=Path(".").resolve()
    chosen=[]
    chosen += [root/p for p in ("pyproject.toml","uv.lock","models.lock.json","config.json","README.md",
                                 "EXPERIMENT_PROTOCOL.md",".gitignore",".env.example")]
    chosen += list((root/"searchbench").glob("*.py"))
    chosen += list((root/"tests").glob("*.py"))
    chosen += [p for p in (root/"blog").rglob("*") if p.is_file()]
    for p in (root/"results").glob("*"):
        if p.is_file():
            chosen.append(p)
    for folder in (root/"results").iterdir():
        if folder.is_dir():
            chosen += [p for p in folder.glob("*.json") if p.name not in ("passages.json","case_studies.json")]
    chosen=sorted(set(chosen))
    secrets=[]
    for filename in args.credential_file:
        path=Path(filename)
        if path.exists():
            key=path.read_text().strip().strip('"').strip("'").encode()
            if key:
                secrets.append(key)
    hashes={}
    for path in chosen:
        data=path.read_bytes()
        assert not any(key in data for key in secrets),f"Credential detected in deliverable {path.name}"
        hashes[str(path.relative_to(root))]=hashlib.sha256(data).hexdigest()
    dump(root/"deliverables.sha256.json",hashes)
    archive=root/"hybrid-search-benchmark.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in chosen:
            z.write(path,path.relative_to(root))
        z.write(root/"deliverables.sha256.json","deliverables.sha256.json")
    print(f"Packaged {len(chosen)} files; credential scan passed; {archive.stat().st_size/1e6:.2f} MB.")

if __name__=="__main__":
    main()
