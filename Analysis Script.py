#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import importlib.util

REQUIRED = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "requests": "requests", "tabulate": "tabulate"
}
missing = [pip_name for mod, pip_name in REQUIRED.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("[SETUP] Installing missing packages:", ", ".join(missing), flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import csv
import gzip
import hashlib
import io
import json
import math
import platform
import re
import shutil
import tarfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

try:
    from google.colab import files as colab_files
    COLAB = True
except ImportError:
    colab_files = None
    COLAB = False

SEED = 20271106
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_REPLICATES = 3000
MASK_LEVELS = (0.0, 0.10, 0.25, 0.50)
MASK_REPLICATES = 5
META_FIELD_WEIGHTS = {"subject": 0.55, "from": 0.225, "to": 0.225}
ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
BETA_GRID = (0.0, 0.025, 0.05, 0.10, 0.15)
BM25F_SUBJECT_BOOSTS = (1.0, 2.0, 4.0)   # tuned on training groups only
BM25F_BOOSTS_BASE = {"body": 1.0, "subject": 2.0, "from": 1.0, "to": 1.0}
SYSTEMS = (
    "BM25-body", "TFIDF-body", "BM25F-fielded", "Headers-only",
    "BM25+headers", "PAMR-availability"
)
METRICS = ("NDCG@10", "MAP", "BPref", "MRR", "P@10", "Recall@100")
PRIMARY = "NDCG@10"

URLS = {
    "email_archive": (
        "https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz",
        "https://bebop.berkeley.edu/enron/enron_with_categories.tar.gz",
    ),
    "human_qrels": (
        "https://zenodo.org/records/8006820/files/repeated_qrels.txt?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/repeated_qrels.txt",
    ),
    "queries": (
        "https://zenodo.org/records/8006820/files/repeated_queries.tsv?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/repeated_queries.tsv",
    ),
    "needs": (
        "https://zenodo.org/records/8006820/files/information_needs.tsv?download=1",
        "https://raw.githubusercontent.com/JackMcKechnie/SARA/main/information_needs.tsv",
    ),
}
# Observed in a successful, document-ID-matched original-release execution.
# Different archive recompression is allowed ONLY after rigorous semantic checks.
SOURCE_SHA256_REFERENCE = {
    "email_archive": "08625500ab4c032f99acfc1f8eed125b2cd5e9c4280c7ce2d7ce0bd28a190625",
    "human_qrels": "11abd7b0c465bd07c16848042dea8c160a3398c69e549fcf641f46d3b890c67b",
    "queries": "fdadebeba8f2c5b097636f4910ae83ffe2bad2e83b2fc891479eec8fbcdfdb7e",
    "needs": "0d0522fb77b46472e19326e550aeff855f53ecf41a5086b5f31a23c76e618f02",
}

TEST_MODE = "--self-test" in sys.argv
BASE = Path("/content") if COLAB else (Path("/mnt/data") if TEST_MODE else Path.cwd())
OUTPUT_NAME = "PAMR_SARA_ICDE2027_SOFTWARE_SELF_TEST" if TEST_MODE else "PAMR_SARA_ICDE2027_RESULTS"
OUT = BASE / OUTPUT_NAME
ZIP_PATH = BASE / (OUTPUT_NAME + ".zip")
CACHE = BASE / "PAMR_SARA_ORIGINAL_SOURCE_CACHE"


def log(s):
    print("[PAMR]", s, flush=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download_asset(key, name, min_bytes):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / name
    errors = []
    if cache_path.exists() and cache_path.stat().st_size >= min_bytes:
        buf = cache_path.read_bytes()
        if key != "email_archive" or buf[:2] == b"\x1f\x8b":
            return buf, "local_source_cache"
    for uri in URLS[key]:
        for attempt in range(2):
            try:
                log(f"Retrieving {key}: {uri.split('/')[2]}")
                response = requests.get(uri, timeout=(20, 150),
                                        headers={"User-Agent": "Academic-SARA-Original-Evaluation/1.0"})
                response.raise_for_status()
                buf = response.content
                if len(buf) < min_bytes:
                    raise ValueError("File smaller than expected: " + str(len(buf)))
                if key == "email_archive" and buf[:2] != b"\x1f\x8b":
                    raise ValueError("Archive is not gzip-compressed; perhaps an HTML error page")
                cache_path.write_bytes(buf)
                return buf, uri
            except Exception as exc:
                errors.append(f"{uri}: {type(exc).__name__}: {exc}")
                time.sleep(0.5 + attempt)
    if COLAB:
        log(f"Cannot automatically retrieve {key}. Please upload original file: {name}")
        uploaded = colab_files.upload()
        valid = [v for k,v in uploaded.items() if k.lower().endswith((".tar.gz", ".txt", ".tsv"))]
        if valid:
            buf = valid[0]
            if len(buf) >= min_bytes and (key != "email_archive" or buf[:2] == b"\x1f\x8b"):
                cache_path.write_bytes(buf)
                return buf, "uploaded_by_user"
    raise RuntimeError(f"Could not obtain original {key}; refused to substitute another release:\n" + "\n".join(errors))


def get_header(msg, key):
    return " ".join(str(x).strip() for x in msg.get_all(key, []) if str(x).strip())


def date_parseable(txt):
    try:
        d = parsedate_to_datetime(txt)
        if d is None:
            return False
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return 1970 <= d.year <= 2026
    except Exception:
        return False


def email_body(msg, raw):
    try:
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    parts.append(str(part.get_content() or ""))
            result = "\n".join(parts).strip()
        else:
            result = str(msg.get_content() or "").strip()
        if result:
            return result
    except Exception:
        pass
    text = raw.decode("utf-8", errors="replace")
    sep = re.search(r"\r?\n\r?\n", text)
    return text[sep.end():].strip() if sep else ""


def original_email_dataframe(blob):
    rows = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            if not raw or b"\x00" in raw[:256]:
                continue
            msg = BytesParser(policy=policy.default).parsebytes(raw)
            values = {f: get_header(msg, h) for f,h in
                      (("from","From"),("to","To"),("subject","Subject"),
                       ("date","Date"),("message_id","Message-ID"))}
            if sum(bool(x) for x in values.values()) < 2:
                continue
            docno = re.sub(r"\.txt$", "", PurePosixPath(member.name).name, flags=re.I)
            if not docno:
                continue
            if docno in rows:
                raise ValueError("Duplicate email document identifier: " + docno)
            values["docno"] = docno
            values["body"] = email_body(msg, raw)
            values["has_thread"] = int(bool(get_header(msg,"In-Reply-To") or get_header(msg,"References")))
            for field in ("from","to","subject","message_id"):
                values["has_"+field] = int(bool(values[field]))
            values["valid_date"] = int(date_parseable(values["date"]))
            values["completeness"] = float(np.mean([
                values["has_from"], values["has_to"], values["has_subject"],
                values["has_message_id"], values["valid_date"]]))
            rows[docno] = values
    if not rows:
        raise ValueError("No email document records found in Berkeley tar archive")
    return pd.DataFrame(rows.values()).sort_values("docno").reset_index(drop=True)


def read_benchmark(qbytes, rbytes, nbytes):
    queries = pd.read_csv(io.StringIO(qbytes.decode("utf-8-sig")), sep="\t", header=None,
                          names=["qid","query"], dtype=str)
    needs = pd.read_csv(io.StringIO(nbytes.decode("utf-8-sig")), sep="\t", header=None,
                        names=["need_id","need_description"], dtype=str)
    qrels = pd.read_csv(io.StringIO(rbytes.decode("utf-8-sig")), sep=r"\s+", header=None,
                        names=["qid","iteration","docno","grade"], engine="python",
                        dtype={"qid":str,"iteration":str,"docno":str})
    queries["qid"] = queries.qid.astype(str)
    qrels["qid"] = qrels.qid.astype(str)
    qrels["docno"] = qrels.docno.astype(str)
    qrels["grade"] = pd.to_numeric(qrels.grade, errors="raise").astype(int)
    queries["qid_num"] = pd.to_numeric(queries.qid, errors="raise").astype(int)
    if sorted(queries.qid_num.tolist()) != list(range(1,151)):
        raise ValueError("Expected original 150 SARA qids numbered 1...150")
    queries["need_id"] = ((queries.qid_num-1)//3+1).astype(int)
    if len(needs) != 50 or queries.groupby("need_id").size().tolist() != [3]*50:
        raise ValueError("Expected exactly 50 original information needs, 3 formulations each")
    if len(qrels) != 34413 or not qrels.grade.isin([0,1,2]).all():
        raise ValueError("Wrong original HUMAN qrels; expected 34,413 entries, grades 0/1/2")
    if qrels.duplicated(["qid","docno"]).any() or set(qrels.qid) != set(queries.qid):
        raise ValueError("Duplicate qrels or misaligned qids")
    group_map = dict(zip(queries.qid, queries.need_id))
    qrels["need_id"] = qrels.qid.map(group_map).astype(int)
    counts = qrels.groupby(["need_id","docno"]).agg(n=("grade","size"), u=("grade","nunique"))
    if not (counts.n == 3).all() or not (counts.u == 1).all() or len(counts) != 11471:
        raise ValueError("Qrels do not match 11,471 human assessments repeated exactly 3 times")
    return queries, qrels, needs


def validate_corpus(docs, queries, qrels, strict=True):
    if strict and (len(docs)!=1702 or len(queries)!=150 or len(qrels)!=34413):
        raise ValueError(f"Expected original SARA 1702 docs/150 queries/34413 qrels, got {len(docs)}/{len(queries)}/{len(qrels)}")
    if docs.docno.isna().any() or docs.docno.duplicated().any():
        raise ValueError("Missing or duplicate document identifiers")
    missing_docs = set(qrels.docno) - set(docs.docno)
    if missing_docs:
        raise ValueError(f"Qrel documents do not match corpus: {len(missing_docs)} unmatched IDs")
    if docs.body.map(len).eq(0).all():
        raise ValueError("All parsed email bodies are empty")
    if not docs.completeness.between(0,1).all():
        raise ValueError("Metadata completeness outside [0,1]")
    if qrels.duplicated(["qid","docno"]).any():
        raise ValueError("Duplicate query-document pair")
    return {"email_documents":int(len(docs)), "queries":int(len(queries)),
            "information_needs":int(queries.need_id.nunique()),
            "repeated_human_qrels":int(len(qrels)),
            "unique_need_document_assessments":int(qrels[["need_id","docno"]].drop_duplicates().shape[0]),
            "judged_documents":int(qrels.docno.nunique()),
            "empty_email_bodies":int(docs.body.map(len).eq(0).sum()),
            "qrel_doc_ids_all_match":True,"human_qrels_only":True,
            "simulated_provenance_fields":False,
            "metadata_completeness_not_authenticated_provenance":True}


WORD = re.compile(r"[a-z0-9]+")
STOP = set(ENGLISH_STOP_WORDS)

def tok(text):
    return [word for word in WORD.findall(str(text).lower()) if word not in STOP]


def norm(scores):
    x = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("NaN or infinite retrieval score")
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi-lo < 1e-12 else (x-lo)/(hi-lo)


def rank(scores):
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")


class BM25Index:
    """Standard Okapi BM25 with positive Robertson/Lucene-smoothed IDF.

    Both the body baseline and BM25F use the same IDF convention:
    log(1 + (N-df+0.5)/(df+0.5)). No extra external BM25 package required.
    """
    def __init__(self, tokenized, k1=1.5, b=0.75):
        self.n=len(tokenized)
        self.k1=float(k1)
        self.b=float(b)
        self.doc_len=np.asarray([len(t) for t in tokenized],dtype=float)
        self.avgdl=max(float(self.doc_len.mean()),1e-12)
        self.length_norm=(1-self.b+self.b*self.doc_len/self.avgdl)
        counts=defaultdict(list)
        for docidx,tokens in enumerate(tokenized):
            for term,tf in Counter(tokens).items():
                counts[term].append((docidx,tf))
        self.post={term:(np.asarray([idx for idx,_ in vals],dtype=np.int32),
                         np.asarray([v for _,v in vals],dtype=float))
                   for term,vals in counts.items()}
        self.idf={term:float(np.log1p((self.n-len(vals)+0.5)/(len(vals)+0.5)))
                  for term,vals in counts.items()}

    def get_scores(self,query_terms):
        result=np.zeros(self.n,dtype=float)
        for term, qtf in Counter(query_terms).items():
            if term not in self.post:continue
            idx,tf=self.post[term]
            val=self.idf[term]*(self.k1+1)*tf/(tf+self.k1*self.length_norm[idx])
            result[idx]+=qtf*val
        return result


class BM25FIndex:
    """BM25F: per-field length normalization, aggregated term frequency, one saturation.

    IDF for a term is based on the union of documents containing it in any field.
    Header weights are actual field boosts, not a weighted average of BM25 ranks.
    """
    def __init__(self, docs, fields=("body","subject","from","to"), k1=1.5, b=0.75):
        self.n = len(docs)
        self.k1 = float(k1)
        self.b = float(b)
        self.fields = tuple(fields)
        self.lengths = {}
        self.avglens = {}
        self.postings = {}
        df_sets = defaultdict(set)
        for field in self.fields:
            terms_per_doc = [tok(x) for x in docs[field].fillna("").astype(str)]
            length = np.asarray([len(x) for x in terms_per_doc], dtype=float)
            self.lengths[field] = length
            self.avglens[field] = max(float(length.mean()), 1e-12)
            index = defaultdict(list)
            for docidx, tokens in enumerate(terms_per_doc):
                for term, count in Counter(tokens).items():
                    index[term].append((docidx, count))
                    df_sets[term].add(docidx)
            self.postings[field] = {
                term: (np.fromiter((x for x,_ in data),dtype=np.int32),
                       np.fromiter((tf for _,tf in data),dtype=np.float64))
                for term,data in index.items()
            }
        self.idf = {term:float(np.log1p((self.n-len(ds)+0.5)/(len(ds)+0.5)))
                    for term,ds in df_sets.items()}
        self.denoms = {f:1-self.b+self.b*self.lengths[f]/self.avglens[f] for f in self.fields}

    def score(self, query_terms, subject_boost=2.0):
        weights = {"body":1.0, "subject":float(subject_boost), "from":1.0, "to":1.0}
        result = np.zeros(self.n, dtype=np.float64)
        for term, qtf in Counter(query_terms).items():
            if term not in self.idf:
                continue
            aggregate = np.zeros(self.n, dtype=np.float64)
            for field in self.fields:
                pairs = self.postings[field].get(term)
                if pairs is None:
                    continue
                indices, frequencies = pairs
                aggregate[indices] += weights[field]*frequencies/self.denoms[field][indices]
            result += qtf*self.idf[term]*(self.k1+1)*aggregate/(self.k1+aggregate)
        return result


class Indices:
    def __init__(self, docs):
        start=time.perf_counter()
        self.docnos = docs.docno.tolist()
        self.n=len(docs)
        self.bm={}
        for f in ("body","subject","from","to"):
            tokens=[tok(s) for s in docs[f].fillna("").astype(str)]
            if not any(tokens):
                raise ValueError("Empty token vocabulary: " + f)
            self.bm[f]=BM25Index(tokens, k1=1.5, b=0.75)
        self.tfidf=TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2),
                                   sublinear_tf=True, dtype=np.float64)
        self.tfidf_corpus=self.tfidf.fit_transform(docs.body.fillna("").astype(str))
        self.bm25f=BM25FIndex(docs)
        self.build_seconds=time.perf_counter()-start

    def features(self, query):
        terms=tok(query)
        raw={f:np.asarray(m.get_scores(terms),dtype=float) if terms else np.zeros(self.n)
             for f,m in self.bm.items()}
        base={f:norm(v) for f,v in raw.items()}
        tfscore=(self.tfidf.transform([str(query)]) @ self.tfidf_corpus.T).toarray().ravel()
        base["body_raw"] = raw["body"]
        base["tfidf_raw"] = tfscore
        base["meta"] = sum(META_FIELD_WEIGHTS[f]*base[f] for f in META_FIELD_WEIGHTS)
        base["bm25f"] = {boost:self.bm25f.score(terms, subject_boost=boost)
                         for boost in BM25F_SUBJECT_BOOSTS}
        return base


def combine(feat, alpha, beta, completeness, drop=None):
    if alpha<0 or beta<0 or alpha+beta>0.75+1e-12:
        raise ValueError("Invalid fusion weight combination")
    meta=feat["meta"]
    if drop in META_FIELD_WEIGHTS:
        meta=meta - META_FIELD_WEIGHTS[drop]*feat[drop]
    score=(1-alpha-beta)*feat["body"] + alpha*meta
    if drop != "completeness":
        score=score + beta*completeness
    return score


def measurement(order, docnos, qtruth):
    labels=np.fromiter((qtruth.get(docnos[int(k)],-1) for k in order),dtype=np.int8,count=len(order))
    true_grades=list(qtruth.values())
    nrel=sum(v>=1 for v in true_grades)
    nneg=sum(v==0 for v in true_grades)
    gains=np.power(2.0, np.maximum(labels[:10],0))-1
    discounts=np.log2(np.arange(2,len(gains)+2))
    dcg=float(np.sum(gains/discounts))
    ideal=np.sort(np.asarray(true_grades,dtype=np.int8))[::-1][:10]
    idcg=float(np.sum((np.power(2.,ideal)-1)/np.log2(np.arange(2,len(ideal)+2))))
    ndcg=dcg/idcg if idcg>0 else 0.0
    binary=labels>=1
    positions=np.flatnonzero(binary)
    ap=float((np.cumsum(binary)[binary]/(positions+1)).sum()/nrel) if nrel else 0.0
    mrr=float(1/(positions[0]+1)) if len(positions) else 0.0
    p10=float(binary[:10].sum()/10)
    rec100=float(binary[:100].sum()/nrel) if nrel else 0.0
    if nrel and nneg:
        cap=min(nrel,nneg)
        neg_before=0
        total=0.0
        for g in labels:
            if g==0:
                neg_before+=1
            elif g>=1:
                total+=1-min(cap,neg_before)/cap
        bpref=total/nrel
    else:
        bpref=float("nan")
    return {"NDCG@10":float(ndcg),"MAP":ap,"BPref":bpref,"MRR":mrr,
            "P@10":p10,"Recall@100":rec100}


def cheap_ndcg(scores, rel_array, idcg):
    if idcg <= 0:
        return 0.0
    ix=rank(scores)[:10]
    grades=rel_array[ix]
    return float(np.sum((np.power(2.,grades)-1)/np.log2(np.arange(2,len(ix)+2)))/idcg)


def grouped_mean(values, groups):
    by=defaultdict(list)
    for v,g in zip(values, groups):
        by[g].append(float(v))
    return float(np.mean([np.mean(x) for x in by.values()]))


def train_grid(train_qids, features, qtruth, docnos, qgroups, completeness, mode):
    """Training-only grouped CV to pick a parameter; no outer-test qrels involved."""
    group_values=np.array([qgroups[q] for q in train_qids],dtype=int)
    split_indices=list(GroupKFold(n_splits=INNER_FOLDS).split(np.arange(len(train_qids)),groups=group_values))
    label_arrays={q:np.asarray([max(qtruth[q].get(d,-1),0) for d in docnos],dtype=np.int8)
                  for q in train_qids}
    idcg={}
    discounts=np.log2(np.arange(2,12))
    for q,y in label_arrays.items():
        ideal=np.sort(y)[::-1][:10]
        idcg[q]=float(np.sum((np.power(2.,ideal)-1)/discounts))
    configs = ([{"alpha":a,"beta":b} for a in ALPHA_GRID for b in (BETA_GRID if mode=="pamr" else (0.0,))
                if a+b<=0.75+1e-12] if mode in ("pamr","hybrid")
               else [{"subject_boost":b} for b in BM25F_SUBJECT_BOOSTS])
    rows=[]
    for cfg in configs:
        fold_vals=[]
        for _,valid in split_indices:
            vals=[]; gs=[]
            for j in valid:
                q=train_qids[int(j)]
                if mode == "bm25f":
                    scores=features[q]["bm25f"][cfg["subject_boost"]]
                else:
                    scores=combine(features[q],cfg["alpha"],cfg["beta"],completeness)
                vals.append(cheap_ndcg(scores,label_arrays[q],idcg[q]))
                gs.append(qgroups[q])
            fold_vals.append(grouped_mean(vals,gs))
        rows.append({**cfg,"inner_NDCG@10":float(np.mean(fold_vals)),
                     "inner_sd":float(np.std(fold_vals,ddof=1)),"mode":mode})
    df=pd.DataFrame(rows)
    # Favor simpler/smaller metadata weights when performance ties within floating error.
    if mode=="bm25f":
        df=df.sort_values(["inner_NDCG@10","subject_boost"],ascending=[False,True],kind="mergesort")
    else:
        df=df.sort_values(["inner_NDCG@10","beta","alpha"],ascending=[False,True,True],kind="mergesort")
    return df.iloc[0].to_dict(),df.reset_index(drop=True)


def make_bootstrap_groups(seed=SEED, n=50, reps=BOOTSTRAP_REPLICATES):
    return np.random.default_rng(seed).integers(0,n,size=(reps,n))


def bootstrap_ci(values, idx):
    vals=np.asarray(values,dtype=float)
    if len(vals)!=idx.shape[1]:
        raise ValueError("Bootstrap group array length mismatch")
    draws=np.nanmean(vals[idx],axis=1)
    return tuple(float(x) for x in np.nanquantile(draws,[0.025,0.975]))


def holm(pvals):
    p=np.asarray(pvals,dtype=float)
    order=np.argsort(p)
    n=len(p)
    adjusted=np.empty(n,dtype=float)
    largest=0.0
    for j,idx in enumerate(order):
        largest=max(largest, min(1.0,(n-j)*p[idx]))
        adjusted[idx]=largest
    return adjusted


def safe_wilcoxon(x):
    x=np.asarray(x,dtype=float)
    if np.all(np.abs(x)<1e-14):
        return 1.0
    try:
        return float(wilcoxon(x,zero_method="wilcox",alternative="two-sided",method="auto").pvalue)
    except ValueError:
        return float("nan")


def save_table(folder,stem,df,caption):
    (folder/(stem+".csv")).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/(stem+".csv"),index=False,float_format="%.12g")
    (folder/(stem+".md")).write_text("# "+caption+"\n\n"+df.to_markdown(index=False,floatfmt=".6g")+"\n",encoding="utf-8")


def fig_save(folder,stem,caption):
    plt.tight_layout()
    plt.savefig(folder/(stem+".png"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.savefig(folder/(stem+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close()
    return f"{stem}: {caption}"


def set_figstyle():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"savefig.dpi":300,"figure.dpi":100})


def dataset_tables(docs, queries, qrels, folder):
    uq=qrels.drop_duplicates(["need_id","docno"])
    counts=uq.grade.value_counts()
    profile=pd.DataFrame([
        ("Email documents",len(docs)),("Information needs (independent units)",queries.need_id.nunique()),
        ("Query formulations (three per need)",len(queries)),("Repeated human qrel rows",len(qrels)),
        ("Unique need-document human assessments",len(uq)),("Unique judged emails",qrels.docno.nunique()),
        ("Emails never judged for any need",len(docs)-qrels.docno.nunique()),
        ("Unique grade 0 (nonrelevant) assessments",int(counts.get(0,0))),
        ("Unique grade 1 (partially relevant) assessments",int(counts.get(1,0))),
        ("Unique grade 2 (highly relevant) assessments",int(counts.get(2,0)))
    ],columns=["characteristic","value"])
    save_table(folder,"Table_01_dataset_profile",profile,"Original human benchmark composition; repeated rows are not independent")
    fields=[("From (asserted sender)","has_from"),("To (recipient)","has_to"),
            ("Subject","has_subject"),("Date (parseable)","valid_date"),
            ("Message-ID","has_message_id"),("In-Reply-To/References","has_thread")]
    coverage=pd.DataFrame([{"field":k,"present_documents":int(docs[v].sum()),
                            "coverage_percent":100*float(docs[v].mean())} for k,v in fields])
    save_table(folder,"Table_02_header_coverage",coverage,"Actual header coverage; not authenticated provenance")
    comp=docs.completeness.round(6).value_counts().sort_index()
    comp_df=pd.DataFrame({"completeness_score":comp.index.to_numpy(),"documents":comp.to_numpy(),
                          "percent":100*comp.to_numpy()/len(docs)})
    save_table(folder,"Table_03_completeness",comp_df,"Observed availability of five email metadata fields")
    return coverage


def audit_cross_check(results, ranks, truth, docnos):
    sample=results.sample(n=min(24,len(results)),random_state=SEED)
    checks=[]
    for r in sample.itertuples(index=False):
        order=ranks[(r.system,r.qid)]
        truth_labels=np.asarray([max(truth[r.qid].get(docnos[int(i)],-1),0) for i in order],dtype=float)
        ndcg=float(ndcg_score((np.power(2.0,truth_labels)-1)[None,:],
                              np.arange(len(order),0,-1,dtype=float)[None,:],k=10))
        # pandas .itertuples() sanitizes metric field names; use explicit row lookup.
        reported=float(results.loc[(results.system==r.system)&(results.qid==r.qid),"NDCG@10"].iloc[0])
        diff=abs(ndcg-reported)
        checks.append({"system":r.system,"qid":r.qid,"ndcg_saved":reported,
                       "ndcg_sklearn":ndcg,"abs_diff":diff})
        if diff>1e-10:
            raise AssertionError(f"Independent sklearn nDCG mismatch, qid={r.qid}: {reported} vs {ndcg}")
    return pd.DataFrame(checks)


def make_source_mock():
    """SOFTWARE-ONLY fixture with 48 docs and 12 needs; never for publication."""
    random=np.random.default_rng(SEED)
    rows=[]
    for i in range(48):
        k=i%12
        rows.append({"docno":f"D{i:04d}",
                     "body":f"project{k} report compliance audit meeting project{k} schedule budget team updates",
                     "subject":f"project{k} monthly report", "from":f"person{i%7}@example.org",
                     "to":f"team{i%4}@example.org" if i%5 else "",
                     "date":"Mon, 03 Jan 2005 12:00:00 -0800",
                     "message_id":f"<mock{i}@example.org>","has_thread":0,
                     "has_from":1,"has_to":int(i%5!=0),"has_subject":1,"has_message_id":1,
                     "valid_date":1,"completeness":float((4+int(i%5!=0))/5)})
    docs=pd.DataFrame(rows)
    queries=[]; qrels=[]
    for g in range(1,13):
        for v in range(3):
            q=str(3*(g-1)+v+1)
            queries.append({"qid":q,"query":f"project{g-1} report", "qid_num":int(q),"need_id":g})
            for d in rows:
                grade=2 if d["docno"].endswith(f"{g-1:02d}") else (1 if f"project{g-1}" in d["subject"] else 0)
                qrels.append({"qid":q,"docno":d["docno"],"grade":grade,"need_id":g})
    return docs,pd.DataFrame(queries),pd.DataFrame(qrels)


def query_class(feat):
    s=np.count_nonzero(feat["subject"])
    f=np.count_nonzero(feat["from"])
    t=np.count_nonzero(feat["to"])
    if s and (f or t): return "subject + address overlap"
    if s: return "subject overlap only"
    if f or t: return "address overlap only"
    return "no header-token overlap"


def perform(docs, queries, qrels, source_audit, strict=True):
    set_figstyle()
    start_all=time.perf_counter()
    validation=validate_corpus(docs, queries, qrels, strict=strict)
    if OUT.exists(): shutil.rmtree(OUT)
    tables=OUT/"tables"
    figs=OUT/"figures"
    runs=OUT/"runs"
    audit=OUT/"audit"
    for p in (tables,figs,runs,audit):p.mkdir(parents=True,exist_ok=True)
    coverage=dataset_tables(docs,queries,qrels,tables)
    docnos=docs.docno.tolist()
    num_docs=len(docnos)
    c=docs.completeness.to_numpy(dtype=float)
    flags={f:docs["has_"+f].to_numpy(dtype=float) for f in ("subject","from","to")}
    flags["date"]=docs.valid_date.to_numpy(dtype=float)
    flags["message_id"]=docs.has_message_id.to_numpy(dtype=float)
    qids=queries.qid.astype(str).tolist()
    qtexts=dict(zip(qids,queries["query"].astype(str)))
    qgroup=dict(zip(queries.qid.astype(str),queries.need_id.astype(int)))
    truth={q:{} for q in qids}
    for r in qrels.itertuples(index=False):truth[str(r.qid)][str(r.docno)]=int(r.grade)
    log("Validated source IDs; indexing body, headers, TF-IDF and true BM25F")
    indices=Indices(docs)
    log(f"All index structures built in {indices.build_seconds:.2f} seconds")
    features={}
    pre_ms=[]
    for q in qids:
        tic=time.perf_counter()
        features[q]=indices.features(qtexts[q])
        pre_ms.append({"qid":q,"feature_generation_ms":1000*(time.perf_counter()-tic)})
    pd.DataFrame(pre_ms).to_csv(runs/"per_query_feature_generation_ms.csv",index=False)
    docflag_cols=["docno","has_from","has_to","has_subject","valid_date","has_message_id","has_thread","completeness"]
    docs[docflag_cols].to_csv(audit/"observed_email_header_flags.csv",index=False)
    queries[["qid","query","need_id"]].to_csv(audit/"original_queries.csv",index=False)
    qrels[["qid","docno","grade","need_id"]].to_csv(audit/"original_human_qrels.csv",index=False)
    (audit/"validation.json").write_text(json.dumps({**validation,"source_audit":source_audit},indent=2),encoding="utf-8")

    outer_splits=list(GroupKFold(n_splits=OUTER_FOLDS).split(np.arange(len(qids)), groups=queries.need_id.to_numpy()))
    fold_rows=[]; grids=[]; primary_rows=[]; ablations=[]; evidence=[]; rankings={}
    selected={}; q_classes={q:query_class(features[q]) for q in qids}
    tic_tune=time.perf_counter()
    for fold,(tr_ix,te_ix) in enumerate(outer_splits,start=1):
        train=[qids[i] for i in tr_ix]
        test=[qids[i] for i in te_ix]
        if set(qgroup[q] for q in train)&set(qgroup[q] for q in test):
            raise AssertionError("Outer-fold group leakage")
        best_bm, gb=train_grid(train,features,truth,docnos,qgroup,c,"bm25f")
        best_h, gh=train_grid(train,features,truth,docnos,qgroup,c,"hybrid")
        best_p, gp=train_grid(train,features,truth,docnos,qgroup,c,"pamr")
        bmf=float(best_bm["subject_boost"])
        a0=float(best_h["alpha"])
        a=float(best_p["alpha"]); b=float(best_p["beta"])
        fold_rows.append({"fold":fold,"training_needs":len(set(qgroup[q] for q in train)),
                          "heldout_needs":len(set(qgroup[q] for q in test)),
                          "BM25F_subject_boost":bmf,"hybrid_alpha":a0,
                          "PAMR_alpha":a,"PAMR_beta":b,
                          "BM25F_train_NDCG@10":best_bm["inner_NDCG@10"],
                          "PAMR_train_NDCG@10":best_p["inner_NDCG@10"]})
        for grid in (gb,gh,gp):
            v=grid.copy();v["outer_fold"]=fold;grids.append(v)
        log(f"Held-out fold {fold}/{OUTER_FOLDS}: BM25F Subject={bmf:g}, hybrid alpha={a0:g}, PAMR alpha={a:g} beta={b:g}")
        for q in test:
            feat=features[q]
            selected[q]={"fold":fold,"bm25f_subject_boost":bmf,"hybrid_alpha":a0,"pamr_alpha":a,"pamr_beta":b}
            orders={
                "BM25-body":rank(feat["body_raw"]),
                "TFIDF-body":rank(feat["tfidf_raw"]),
                "BM25F-fielded":rank(feat["bm25f"][bmf]),
                "Headers-only":rank(feat["meta"]),
                "BM25+headers":rank(combine(feat,a0,0.0,c)),
                "PAMR-availability":rank(combine(feat,a,b,c)),
            }
            for system,order in orders.items():
                metrics=measurement(order,docnos,truth[q])
                primary_rows.append({"system":system,"qid":q,"need_id":qgroup[q],"fold":fold,
                                     "query_type":q_classes[q],**metrics})
                rankings[(system,q)]=order
                top=docs.iloc[order[:10]]
                evidence.append({"system":system,"qid":q,"need_id":qgroup[q],
                                 "completeness_at10":float(top.completeness.mean()),
                                 "five_fields_at10":float((top.completeness==1).mean()),
                                 "thread_pointer_at10":float(top.has_thread.mean())})
            for dropped in ("subject","from","to","completeness"):
                ab_order=rank(combine(feat,a,b,c,drop=dropped))
                ablations.append({"variant":"without "+dropped,"qid":q,"need_id":qgroup[q],
                                  **measurement(ab_order,docnos,truth[q])})
    tune_seconds=time.perf_counter()-tic_tune
    pd.DataFrame(fold_rows).to_csv(audit/"nested_cv_selected_parameters.csv",index=False)
    pd.concat(grids,ignore_index=True).to_csv(audit/"all_inner_cv_parameter_scores.csv",index=False)
    pq=pd.DataFrame(primary_rows)
    pq.to_csv(runs/"heldout_metrics_by_query.csv",index=False)
    pq["need_id"]=pq.need_id.astype(int)
    gm=pq.groupby(["system","need_id"],as_index=False)[list(METRICS)].mean()
    gm.to_csv(runs/"heldout_metrics_by_50_needs.csv",index=False)
    pd.DataFrame(ablations).to_csv(runs/"ablation_metrics_by_query.csv",index=False)
    pd.DataFrame(evidence).to_csv(runs/"metadata_evidence_at10_by_query.csv",index=False)
    if not all(pq[pq.system==system].qid.nunique()==len(qids) for system in SYSTEMS):
        raise AssertionError("A held-out query is missing from one of the systems")
    if len(rankings)!=len(SYSTEMS)*len(qids):
        raise AssertionError("Missing out-of-fold full ranking")
    log("Held-out rankings generated; exporting full rankings and audit metrics")
    with gzip.open(runs/"full_rankings_all_systems.csv.gz","wt",encoding="utf-8",newline="") as fh:
        writer=csv.writer(fh);writer.writerow(["system","qid","need_id","rank","docno"])
        for system in SYSTEMS:
            for q in qids:
                order=rankings[(system,q)]
                if len(order)!=num_docs or len(set(order.tolist()))!=num_docs:
                    raise AssertionError("Duplicate document in a full ranking")
                writer.writerows((system,q,qgroup[q],i+1,docnos[int(j)]) for i,j in enumerate(order))
    independent=audit_cross_check(pq,rankings,truth,docnos)
    independent.to_csv(audit/"independent_sklearn_ndcg_validation.csv",index=False)
    validation["independent_sklearn_ndcg_test_pairs"]=len(independent)
    log(f"Independently verified {len(independent)} nDCG computations against scikit-learn")

    # All confidence intervals use the SAME seeded need-resampling indices.
    need_order=sorted(queries.need_id.unique().tolist())
    boot=make_bootstrap_groups(n=len(need_order))
    summary=[]
    for sysname in SYSTEMS:
        part=gm[gm.system==sysname].set_index("need_id").reindex(need_order)
        row={"system":sysname}
        for metric in METRICS:
            values=part[metric].to_numpy(dtype=float)
            lo,hi=bootstrap_ci(values,boot)
            row[metric]=float(np.nanmean(values));row[metric+"_CI_low"]=lo;row[metric+"_CI_high"]=hi
        summary.append(row)
    performance=pd.DataFrame(summary)
    save_table(tables,"Table_04_retrieval_performance",performance,
               "Held-out mean metrics across 50 independent information needs, common clustered bootstrap 95% CIs")
    comparison=[]
    base=gm[gm.system=="BM25-body"].set_index("need_id").reindex(need_order)
    for sysname in SYSTEMS:
        if sysname=="BM25-body":continue
        test=gm[gm.system==sysname].set_index("need_id").reindex(need_order)
        diff=test[PRIMARY].to_numpy()-base[PRIMARY].to_numpy()
        lo,hi=bootstrap_ci(diff,boot)
        comparison.append({"system":sysname,"versus":"BM25-body","mean_delta_NDCG@10":float(diff.mean()),
                           "CI_low":lo,"CI_high":hi,"p_wilcoxon":safe_wilcoxon(diff),
                           "positive_needs":int((diff>1e-12).sum()),
                           "equal_needs":int((np.abs(diff)<=1e-12).sum()),
                           "negative_needs":int((diff< -1e-12).sum())})
    diff_table=pd.DataFrame(comparison)
    diff_table["p_Holm_adj"]=holm(diff_table.p_wilcoxon.fillna(1).values)
    save_table(tables,"Table_05_paired_vs_BM25",diff_table,
               "Paired need-level NDCG@10 differences against BM25 and Holm-adjusted p-values")

    # Ablations use FULL PAMR's chosen OOF weights without retuning.
    abr=pd.DataFrame(ablations)
    abg=abr.groupby(["variant","need_id"],as_index=False)[list(METRICS)].mean()
    ab_summ=[]
    pamr=gm[gm.system=="PAMR-availability"].set_index("need_id").reindex(need_order)
    for label in ("without subject","without from","without to","without completeness"):
        part=abg[abg.variant==label].set_index("need_id").reindex(need_order)
        delta=part[PRIMARY].to_numpy()-pamr[PRIMARY].to_numpy()
        lo,hi=bootstrap_ci(delta,boot)
        ab_summ.append({"variant":label,"NDCG@10":float(part[PRIMARY].mean()),
                        "delta_vs_full_PAMR":float(delta.mean()),"delta_CI_low":lo,"delta_CI_high":hi,
                        "p_unadjusted_exploratory":safe_wilcoxon(delta)})
    ab_table=pd.DataFrame(ab_summ)
    save_table(tables,"Table_06_fixed_weight_ablation",ab_table,
               "Frozen-parameter component removal; exploratory unadjusted p-values")
    fold_df=pd.DataFrame(fold_rows)
    save_table(tables,"Table_07_selected_outer_fold_weights",fold_df,
               "Nested selection based solely on training information needs")

    # Query analysis is descriptive; header overlap labels are computed without qrels.
    query_types=pq.groupby(["system","query_type"],as_index=False)[PRIMARY].mean()
    query_types.to_csv(runs/"descriptive_query_type_performance.csv",index=False)
    def per_need_delta(sysname):
        a=gm[gm.system==sysname].set_index("need_id").reindex(need_order)[PRIMARY]
        return a.to_numpy()-base[PRIMARY].to_numpy()
    pamr_delta=per_need_delta("PAMR-availability")
    per_need=pd.DataFrame({"need_id":need_order,"delta_PAMR_minus_BM25":pamr_delta})
    per_need["outcome"]=np.where(per_need.delta_PAMR_minus_BM25>1e-12,"improved",
                           np.where(per_need.delta_PAMR_minus_BM25< -1e-12,"worsened","unchanged"))
    per_need.to_csv(runs/"per_need_PAMR_effects.csv",index=False)
    save_table(tables,"Table_08_query_effect_summary",
               per_need.outcome.value_counts().rename_axis("outcome").reset_index(name="information_needs"),
               "PAMR versus BM25, one comparison per independent information need")

    # Header-overlap subgroups are defined from query/field tokens WITHOUT qrels.
    # Descriptive only: subgroups overlap at need level, and are not confirmatory tests.
    subgroup_rows=[]
    for label in sorted(set(q_classes.values())):
        q_in=[q for q in qids if q_classes[q]==label]
        common=pq[pq.qid.isin(q_in)]
        joined=(common[common.system=="PAMR-availability"][["qid","need_id",PRIMARY]]
                .merge(common[common.system=="BM25-body"][["qid",PRIMARY]],on="qid",suffixes=("_pamr","_bm25")))
        deltas=joined[PRIMARY+"_pamr"]-joined[PRIMARY+"_bm25"]
        subgroup_rows.append({"header_token_overlap_category":label,
                              "formulations":len(q_in),"distinct_needs":len(set(qgroup[q] for q in q_in)),
                              "mean_query_level_delta_NDCG@10_descriptive":float(deltas.mean())})
    save_table(tables,"Table_12_header_overlap_subgroups",pd.DataFrame(subgroup_rows),
               "Exploratory predefined query-header overlap groups, not independent confirmatory testing")

    # Test whether observed completeness is actually discriminatory for relevance.
    # Use unique need-document human judgments (not repeated three times).
    judged_unique=(qrels.drop_duplicates(["need_id","docno"])[["need_id","docno","grade"]]
                   .merge(docs[["docno","completeness"]],on="docno",how="left",validate="many_to_one"))
    judged_unique["binary_relevant"]=judged_unique.grade>=1
    relevance_coverage=(judged_unique.groupby("binary_relevant")["completeness"]
                        .agg(["size","mean","median"]).reset_index())
    relevance_coverage["class"] = np.where(relevance_coverage.binary_relevant,"grade >= 1","grade = 0")
    save_table(tables,"Table_13_completeness_by_human_relevance",relevance_coverage,
               "Observed header completeness among unique human need-document assessments; NOT independent provenance quality")
    rel_g=(judged_unique.groupby(["need_id","binary_relevant"])["completeness"].mean()
           .unstack("binary_relevant"))
    if False in rel_g.columns and True in rel_g.columns:
        matched=rel_g.dropna(subset=[False,True])
        rel_diff=(matched[True]-matched[False]).to_numpy(dtype=float)
        boots=make_bootstrap_groups(seed=SEED+77,n=len(rel_diff))
        lo,hi=bootstrap_ci(rel_diff,boots)
        completeness_effect=pd.DataFrame([{
            "matched_information_needs":len(rel_diff),
            "mean_complete_relevant_minus_nonrelevant":float(rel_diff.mean()),
            "paired_group_bootstrap_CI_low":lo,"paired_group_bootstrap_CI_high":hi,
            "p_wilcoxon_exploratory":safe_wilcoxon(rel_diff)}])
    else:
        completeness_effect=pd.DataFrame([{"matched_information_needs":0,
            "mean_complete_relevant_minus_nonrelevant":float("nan"),
            "paired_group_bootstrap_CI_low":float("nan"),
            "paired_group_bootstrap_CI_high":float("nan"),
            "p_wilcoxon_exploratory":float("nan")}])
    save_table(tables,"Table_14_completeness_relevance_association",completeness_effect,
               "Need-level relationship of header completeness to judged relevance (exploratory)")

    # Reproducible extreme case illustrations (not representative sample selection).
    extreme=[]
    extreme_cases = [
        ("most_improved", int(per_need.loc[per_need["delta_PAMR_minus_BM25"].idxmax(), "need_id"])),
        ("most_worsened", int(per_need.loc[per_need["delta_PAMR_minus_BM25"].idxmin(), "need_id"])),
    ]
    for outcome,need in extreme_cases:
        q=next(q for q in qids if qgroup[q]==need)
        bm=rankings[("BM25-body",q)];pm=rankings[("PAMR-availability",q)]
        for i in range(min(10,num_docs)):
            a_doc=docnos[int(bm[i])];b_doc=docnos[int(pm[i])]
            extreme.append({"example_selection":outcome,"need_id":need,"qid":q,"rank":i+1,
                            "BM25_docno":a_doc,"BM25_grade_or_minus1_unjudged":truth[q].get(a_doc,-1),
                            "PAMR_docno":b_doc,"PAMR_grade_or_minus1_unjudged":truth[q].get(b_doc,-1)})
    pd.DataFrame(extreme).to_csv(runs/"illustrative_extreme_rank_changes.csv",index=False)

    # Controlled post-index metadata masking with SAME doc-level masks for every query.
    # Original corpus IDF is kept fixed: evaluates field unavailability at retrieval time.
    rng=np.random.default_rng(SEED+123)
    masking=[]
    mask_base=[(q,measurement(rank(combine(features[q],selected[q]["pamr_alpha"],
                                              selected[q]["pamr_beta"],c)),docnos,truth[q])[PRIMARY]) for q in qids]
    unmasked=dict(mask_base)
    for p_mask in MASK_LEVELS:
        for replicate in range(MASK_REPLICATES):
            masks={name:(rng.random(num_docs)>=p_mask).astype(float) for name in flags}
            c_mask=(flags["from"]*masks["from"] + flags["to"]*masks["to"] +
                    flags["subject"]*masks["subject"] + flags["date"]*masks["date"] +
                    flags["message_id"]*masks["message_id"])/5
            for q in qids:
                f=features[q]
                attenuated=sum(META_FIELD_WEIGHTS[x]*norm(f[x]*masks[x]) for x in META_FIELD_WEIGHTS)
                alpha=selected[q]["pamr_alpha"]
                beta=selected[q]["pamr_beta"]
                score=(1-alpha-beta)*f["body"] + alpha*attenuated + beta*c_mask
                measure=measurement(rank(score),docnos,truth[q])[PRIMARY]
                masking.append({"missingness_fraction":p_mask,"replicate":replicate,
                                "qid":q,"need_id":qgroup[q],"NDCG@10":measure,
                                "delta_vs_unmasked":measure-unmasked[q],
                                "remaining_metadata_completeness":float(c_mask.mean())})
    maskdf=pd.DataFrame(masking)
    maskdf.to_csv(runs/"controlled_metadata_masking_by_query.csv",index=False)
    # Nested dependence: average the 5 mask replications within each information need,
    # then bootstrap independent needs; do not treat all runs as independent.
    mask_g=(maskdf.groupby(["missingness_fraction","need_id"],as_index=False)
            [["NDCG@10","delta_vs_unmasked","remaining_metadata_completeness"]].mean())
    mask_summary=[]
    for p_mask in MASK_LEVELS:
        part=mask_g[mask_g.missingness_fraction==p_mask].set_index("need_id").reindex(need_order)
        d=part.delta_vs_unmasked.to_numpy()
        nd=part[PRIMARY].to_numpy()
        dlo,dhi=bootstrap_ci(d,boot)
        lo,hi=bootstrap_ci(nd,boot)
        mask_summary.append({"missingness_fraction":p_mask,
                             "NDCG@10":float(np.mean(nd)),"NDCG_CI_low":lo,"NDCG_CI_high":hi,
                             "delta_vs_unmasked":float(np.mean(d)),
                             "delta_CI_low":dlo,"delta_CI_high":dhi,
                             "remaining_completeness":float(part.remaining_metadata_completeness.mean())})
    mask_table=pd.DataFrame(mask_summary)
    save_table(tables,"Table_09_masking_robustness",mask_table,
               "Frozen-weight post-index header withholding; independent unit = information need")

    # Provenance header availability as a separate descriptive outcome, not authenticity.
    ev=pd.DataFrame(evidence)
    ev_group=ev.groupby(["system","need_id"],as_index=False)[["completeness_at10","five_fields_at10","thread_pointer_at10"]].mean()
    ev_summary=ev_group.groupby("system",as_index=False)[["completeness_at10","five_fields_at10","thread_pointer_at10"]].mean()
    ev_summary["system"]=pd.Categorical(ev_summary.system,SYSTEMS,ordered=True)
    ev_summary=ev_summary.sort_values("system").reset_index(drop=True)
    ev_summary["system"]=ev_summary.system.astype(str)
    save_table(tables,"Table_10_observable_evidence_at10",ev_summary,
               "Top-10 observed metadata availability; NOT source trust, verified provenance or document lineage")

    # Fixed reproducible timing protocol, not large-scale scalability evidence.
    # A query samples one formulation per independent need, then three repetitions.
    sample_queries=[str(queries.loc[queries.need_id==g,"qid"].iloc[0]) for g in need_order]
    timing=[]
    def score_latency(system,q):
        terms=tok(qtexts[q]); alpha=selected[q]["pamr_alpha"]
        beta=selected[q]["pamr_beta"]
        if system=="BM25-body":
            values=indices.bm["body"].get_scores(terms)
        elif system=="TFIDF-body":
            values=(indices.tfidf.transform([qtexts[q]])@indices.tfidf_corpus.T).toarray().ravel()
        elif system=="BM25F-fielded":
            values=indices.bm25f.score(terms,selected[q]["bm25f_subject_boost"])
        elif system=="Headers-only":
            pieces={x:norm(indices.bm[x].get_scores(terms)) for x in META_FIELD_WEIGHTS}
            values=sum(META_FIELD_WEIGHTS[x]*pieces[x] for x in META_FIELD_WEIGHTS)
        else:
            body=norm(indices.bm["body"].get_scores(terms))
            head={x:norm(indices.bm[x].get_scores(terms)) for x in META_FIELD_WEIGHTS}
            meta=sum(META_FIELD_WEIGHTS[x]*head[x] for x in META_FIELD_WEIGHTS)
            if system=="BM25+headers":
                a=selected[q]["hybrid_alpha"]
                values=(1-a)*body+a*meta
            else:
                values=(1-alpha-beta)*body + alpha*meta + beta*c
        # Full sort matches the complete-ranking evaluation, not just a fast top-k cutoff.
        return rank(values)
    for system in SYSTEMS:
        for q in sample_queries[:min(5,len(sample_queries))]:
            score_latency(system,q)  # warm-up
        for rep in range(3):
            for q in sample_queries:
                t0=time.perf_counter()
                order=score_latency(system,q)
                elapsed=(time.perf_counter()-t0)*1000
                if len(order)!=num_docs:
                    raise AssertionError("Query timing omitted ranked documents")
                timing.append({"system":system,"qid":q,"need_id":qgroup[q],
                               "repeat":rep,"latency_ms":elapsed})
    tim=pd.DataFrame(timing)
    tim.to_csv(runs/"query_latency_repeated_measurements.csv",index=False)
    runtime=[]
    for system in SYSTEMS:
        vals=tim[tim.system==system].latency_ms.to_numpy()
        runtime.append({"system":system,"median_ms":float(np.median(vals)),
                        "p95_ms":float(np.percentile(vals,95)),"mean_ms":float(np.mean(vals)),
                        "trials":len(vals),"index_build_seconds_all_models":indices.build_seconds})
    runtime_df=pd.DataFrame(runtime)
    save_table(tables,"Table_11_query_latency",runtime_df,
               "Single-thread query scoring and full-ranking latency, common hardware, 50 queries x 3 repeats")

    # ------------------- FIGURES -------------------
    captions=[]
    fig,ax=plt.subplots(figsize=(7.6,3.9))
    y=np.arange(len(coverage))[::-1]
    ax.barh(y,coverage.coverage_percent)
    ax.set_yticks(y,coverage.field.tolist())
    ax.set_xlim(0,112)
    for yy,value in zip(y,coverage.coverage_percent):
        ax.text(value+1,yy,f"{value:.1f}%",va="center",fontsize=9)
    ax.set_xlabel("Documents with observed header field (%)")
    ax.set_title("Original Enron email-header availability")
    captions.append(fig_save(figs,"Figure_01_metadata_coverage","Observed metadata; not verified provenance"))

    main_metrics=("NDCG@10","MAP","BPref")
    fig,axes=plt.subplots(1,3,figsize=(12.8,4),sharey=True)
    yy=np.arange(len(SYSTEMS))[::-1]
    for ax,m in zip(axes,main_metrics):
        mean=performance.set_index("system").loc[list(SYSTEMS),m].values
        low=performance.set_index("system").loc[list(SYSTEMS),m+"_CI_low"].values
        high=performance.set_index("system").loc[list(SYSTEMS),m+"_CI_high"].values
        ax.errorbar(mean,yy,xerr=np.vstack([mean-low,high-mean]),fmt="o",capsize=3)
        ax.set_yticks(yy,list(SYSTEMS))
        ax.set_xlabel(m)
        ax.grid(axis="x",alpha=.2)
        ax.set_xlim(max(0,float(np.min(low)-.03)), min(1,float(np.max(high)+.04)))
    fig.suptitle("Held-out retrieval effectiveness (95% information-need bootstrap CIs)",fontsize=12)
    captions.append(fig_save(figs,"Figure_02_retrieval_comparison","Group-level means and common clustered CIs; not independent-query intervals"))

    fig,ax=plt.subplots(figsize=(8.1,3.7))
    labels=diff_table.system.tolist()
    y=np.arange(len(labels))[::-1]
    m=diff_table["mean_delta_NDCG@10"].to_numpy()
    lo=diff_table.CI_low.to_numpy();hi=diff_table.CI_high.to_numpy()
    ax.errorbar(m,y,xerr=np.vstack([m-lo,hi-m]),fmt="o",capsize=4)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(y,labels)
    ax.set_xlabel("Paired change in NDCG@10 versus BM25-body")
    ax.set_title("Held-out paired differences by system (95% group-bootstrap CIs)")
    ax.grid(axis="x",alpha=.15)
    captions.append(fig_save(figs,"Figure_03_paired_vs_BM25","Negative values indicate lower effectiveness"))

    fig,ax=plt.subplots(figsize=(7.5,3.5))
    y=np.arange(len(ab_table))[::-1]
    mean=ab_table.delta_vs_full_PAMR.to_numpy()
    lo=ab_table.delta_CI_low.to_numpy();hi=ab_table.delta_CI_high.to_numpy()
    ax.errorbar(mean,y,xerr=np.vstack([mean-lo,hi-mean]),fmt="o",capsize=3)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(y,ab_table.variant.tolist())
    ax.set_xlabel("Change in NDCG@10 after component removal")
    ax.set_title("Frozen-weight ablation relative to full PAMR")
    captions.append(fig_save(figs,"Figure_04_component_ablation","Fixed weights; not a refitted model"))

    grids_df=pd.concat(grids,ignore_index=True)
    pam_grid=grids_df[grids_df["mode"]=="pamr"]
    heat=(pam_grid.groupby(["beta","alpha"])["inner_NDCG@10"].mean().unstack("alpha").sort_index())
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    im=ax.imshow(heat.to_numpy(),aspect="auto",origin="lower",cmap="viridis")
    ax.set_xticks(range(len(heat.columns)),[f"{x:g}" for x in heat.columns]);ax.set_xlabel("Metadata weight alpha")
    ax.set_yticks(range(len(heat.index)),[f"{x:g}" for x in heat.index]);ax.set_ylabel("Availability weight beta")
    ax.set_title("Training-only inner-CV parameter sensitivity (not held-out performance)")
    fig.colorbar(im,ax=ax,label="Inner-validation NDCG@10")
    captions.append(fig_save(figs,"Figure_05_inner_cv_sensitivity","Mean over outer folds of training-only grid values"))

    fig,ax=plt.subplots(figsize=(7.3,3.4))
    order=np.argsort(pamr_delta)
    ypos=np.arange(1,len(need_order)+1)
    ax.scatter(pamr_delta[order],ypos,s=25)
    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_xlabel("NDCG@10: PAMR minus BM25 (per information need)")
    ax.set_ylabel("Information needs ordered by paired effect")
    ax.set_title(f"PAMR vs BM25: {(pamr_delta>1e-12).sum()} improved, {(np.abs(pamr_delta)<=1e-12).sum()} equal, {(pamr_delta< -1e-12).sum()} worsened")
    captions.append(fig_save(figs,"Figure_06_per_need_effects","Each of the 50 dots is one independent information need"))

    fig,ax=plt.subplots(figsize=(7.4,3.6))
    x=100*mask_table.missingness_fraction.to_numpy()
    d=mask_table.delta_vs_unmasked.to_numpy()
    dl=mask_table.delta_CI_low.to_numpy();dh=mask_table.delta_CI_high.to_numpy()
    ax.errorbar(x,d,yerr=np.vstack([d-dl,dh-d]),fmt="o-",capsize=4)
    ax.axhline(0,linestyle="--",linewidth=1)
    ax.set_xticks(x,[f"{v:g}%" for v in x]);ax.set_ylabel("Paired change in NDCG@10")
    ax.set_xlabel("Actual email-header fields withheld at query time")
    ax.set_title("PAMR under controlled post-index metadata withholding")
    captions.append(fig_save(figs,"Figure_07_controlled_missingness","Fixed out-of-fold parameters, 5 independent masks per level"))

    fig,ax=plt.subplots(figsize=(7.7,3.5))
    syspos=np.arange(len(SYSTEMS))[::-1]
    med=runtime_df.set_index("system").loc[list(SYSTEMS),"median_ms"].to_numpy()
    p95=runtime_df.set_index("system").loc[list(SYSTEMS),"p95_ms"].to_numpy()
    ax.barh(syspos,med,label="Median latency")
    ax.plot(p95,syspos,"o",label="95th percentile")
    ax.set_yticks(syspos,list(SYSTEMS));ax.set_xlabel("Single-thread query scoring + full ranking (ms)")
    ax.set_title("Exploratory runtime; fixed 1,702-document collection")
    ax.legend()
    captions.append(fig_save(figs,"Figure_08_query_latency","Same runtime, repeated queries; not a scalability proof"))
    (OUT/"FIGURE_CAPTIONS.txt").write_text("\n".join(captions)+"\n",encoding="utf-8")

    # Generate an automatically populated, non-promotional summary for manuscript drafting.
    perf_lookup=performance.set_index("system")
    pamr_comp=diff_table.set_index("system").loc["PAMR-availability"]
    beta_zeros=int((fold_df.PAMR_beta.abs()<=1e-12).sum())
    stat=("was below" if float(pamr_comp["mean_delta_NDCG@10"])<-1e-12 else
          "was above" if float(pamr_comp["mean_delta_NDCG@10"])>1e-12 else "equaled")
    result_summary=f"""# Primary held-out findings — {'SOFTWARE FIXTURE ONLY' if not strict else 'ORIGINAL SARA'}

- Independent evaluation units: {len(need_order)} information needs; three formulations per need.
- BM25-body mean NDCG@10: {perf_lookup.loc['BM25-body',PRIMARY]:.6f}.
- BM25F-fielded mean NDCG@10: {perf_lookup.loc['BM25F-fielded',PRIMARY]:.6f}.
- BM25+headers mean NDCG@10: {perf_lookup.loc['BM25+headers',PRIMARY]:.6f}.
- PAMR-availability mean NDCG@10: {perf_lookup.loc['PAMR-availability',PRIMARY]:.6f}.
- PAMR {stat} BM25 by {abs(float(pamr_comp["mean_delta_NDCG@10"])):.6f} NDCG@10 on average.
- Paired 95% need-clustered bootstrap CI for PAMR minus BM25: [{float(pamr_comp.CI_low):.6f}, {float(pamr_comp.CI_high):.6f}].
- Holm-adjusted Wilcoxon p-value for PAMR versus BM25: {float(pamr_comp.p_Holm_adj):.6g}.
- Provenance-availability beta selected as exactly zero in {beta_zeros} of {len(fold_df)} outer folds.
- These observations DO NOT establish verified provenance quality, document authenticity, or general enterprise-scale efficiency.

Reference tables: `Table_04_retrieval_performance.csv`, `Table_05_paired_vs_BM25.csv`,
`Table_07_selected_outer_fold_weights.csv`, `Table_14_completeness_relevance_association.csv`.

Use the numerical outcomes as measured, including negative results. Correlations between
header completeness and relevance are exploratory and not independent provenance labels.
"""
    (OUT/"CORE_RESULTS_SUMMARY.md").write_text(result_summary,encoding="utf-8")

    # Core paper selection suggestions are descriptive, NOT claims of positive results.
    report={
        "dataset_validation":validation,
        "systems":list(SYSTEMS),"evaluation_metric_primary":PRIMARY,
        "number_of_outer_folds":OUTER_FOLDS,"inner_folds":INNER_FOLDS,
        "BM25F": "True per-field length normalization plus aggregated TF before single saturation; boosted Subject from training-only CV",
        "completeness": "Five observed email fields, not authenticated provenance, trustworthiness, or independent lineage",
        "bootstrap_random_seed":SEED,
        "masking": "Fixed out-of-fold parameters; suppress actual header scores post-index, retain original IDF; 5 random masks per level",
        "timing": "Single CPU process; 50 one-per-need queries, 3 repeated measurements per system; not large scale",
        "weights":fold_rows,
        "source_audit":source_audit,
        "index_build_seconds_all_models":indices.build_seconds,
        "training_weight_selection_seconds":tune_seconds,
        "total_runtime_seconds":time.perf_counter()-start_all,
        "independent_ndcg_checks_passed":len(independent),
        "software_self_test":not strict,
        "data_sensitivity": "Public research corpus may contain personal email content; no raw email bodies or header addresses exported in results package",
        "library_versions":{
            "python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,
            "scikit_learn":__import__("sklearn").__version__,"scipy":__import__("scipy").__version__,
            "matplotlib":matplotlib.__version__,"platform":platform.platform()
        },
    }
    (audit/"full_experiment_manifest.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    readme=f"""# PAMR / SARA original human-judged retrieval experiment

**Status:** {'SOFTWARE FIXTURE ONLY — NOT RESEARCH RESULTS' if not strict else 'ORIGINAL SARA HUMAN RELEVANCE DATASET'}.

- Corpus: {len(docs)} original labelled Berkeley email documents.
- Queries: {len(qids)} formulations grouped into {len(need_order)} independent information needs.
- Primary effectiveness metric: NDCG@10, grades 0/1/2, gain 2^grade-1.
- MAP/MRR/Recall/P@10: grade >=1 relevant; unjudged counted as nonrelevant.
- BPref: unjudged skipped, grade >=1 relevant.
- All reported model scores are held-out nested-grouped predictions.
- Reference model: BM25-body. No tuning of fusion weights on test folds.
- BM25F is genuine term-frequency aggregation before saturation, with field-length normalization.
- Common bootstrap sampled 50 information needs, not 150 pseudo-independent queries.
- Holm correction applies to primary NDCG@10 comparisons vs BM25.
- Sensitivity heatmap uses training-only inner folds, **not test performance**.
- Masking is a synthetic **post-index field-availability intervention** on real observed headers.
- Timing is single-threaded small-corpus exploratory query time, **not scalability**.
- Email headers reflect asserted origin metadata, NOT source authenticity, verification or information lineage.
- Public corpus and judgments remain subject to their source licenses and ethics requirements.

"""
    (OUT/"README_FIRST.md").write_text(readme,encoding="utf-8")
    if Path(__file__).exists():
        shutil.copy2(Path(__file__),OUT/"PAMR_SARA_ICDE2027_One_Click_Colab.py")
    if ZIP_PATH.exists():ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for path in OUT.rglob("*"):
            if path.is_file():
                z.write(path,path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH) as z:
        failed=z.testzip()
        if failed:
            raise IOError("ZIP archive verification failed: " + str(failed))
    log(f"SUCCESS: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1024**2:.2f} MiB)")
    return ZIP_PATH


def main():
    if TEST_MODE:
        log("Running SOFTWARE SELF-TEST with synthetic fixture. NOT a research result.")
        global OUTER_FOLDS, INNER_FOLDS, BOOTSTRAP_REPLICATES, MASK_REPLICATES
        OUTER_FOLDS=3;INNER_FOLDS=2;BOOTSTRAP_REPLICATES=300;MASK_REPLICATES=2
        docs,queries,qrels=make_source_mock()
        path=perform(docs,queries,qrels,{"mode":"synthetic software selftest"},strict=False)
    else:
        data={};audit={}
        for key,filename,minsize in (
            ("email_archive","enron_with_categories.tar.gz",100_000),
            ("human_qrels","repeated_qrels.txt",100_000),
            ("queries","repeated_queries.tsv",1_000),
            ("needs","information_needs.tsv",1_000)
        ):
            buf,where=download_asset(key,filename,minsize)
            digest=sha256(buf)
            audit[key]={"source":where,"sha256":digest,
                        "matches_previous_original_release":digest==SOURCE_SHA256_REFERENCE[key],
                        "size_bytes":len(buf)}
            if digest!=SOURCE_SHA256_REFERENCE[key]:
                log(f"NOTE: {key} byte hash differs from previously verified release; strict semantic validation still applies.")
            data[key]=buf
        docs=original_email_dataframe(data["email_archive"])
        queries,qrels,needs=read_benchmark(data["queries"],data["human_qrels"],data["needs"])
        path=perform(docs,queries,qrels,audit,strict=True)
    if COLAB and not TEST_MODE:
        try:
            colab_files.download(str(path))
        except Exception as exc:
            log(f"Automatic browser download blocked ({exc}). ZIP remains in Colab Files: {path}")
    else:
        log("Finished: " + str(path))


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print("\n[PAMR] VALIDATION OR EXECUTION FAILED — no results claimed.\n" +
              f"{type(exc).__name__}: {exc}",flush=True)
        raise
