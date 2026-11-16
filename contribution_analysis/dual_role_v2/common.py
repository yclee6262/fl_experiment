import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def save(p,obj):
    with Path(p).open('x') as f:json.dump(plain(obj),f,indent=2,allow_nan=False);f.write('\n')
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def check_sources(root):
    cfg=read(root/'registration.json')
    for n,h in cfg['source_sha256'].items():
        if sha(root/'source_snapshot'/n)!=h:raise ValueError('frozen_source:'+n)
    for n,h in read(root/'input_sha256.json').items():
        if sha(n)!=h:raise ValueError('input_hash:'+n)
    for n,h in cfg['document_sha256'].items():
        if sha(root/n)!=h:raise ValueError('document_hash:'+n)
    return cfg
