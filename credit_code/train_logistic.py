# -*- coding: utf-8 -*-
"""
ExtraTrees with WOE/IV + undersampling + CV OOF
Outputs kept the same:
- submission_extratrees.csv to OUT
"""
import os, json, math, warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline as SkPipeline

try:
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    Pipeline = SkPipeline
    RandomUnderSampler = None

import matplotlib.pyplot as plt

# Paths
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.exists(BASE_DIR) else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

TRAIN_XLSX = os.path.join(DATA_DIR, "训练数据集.xlsx")
TEST_XLSX  = os.path.join(DATA_DIR, "测试集.xlsx")
SAMPLE_CSV = os.path.join(DATA_DIR, "提交样例.csv")

RANDOM_STATE = 42
N_SPLITS = 5

def ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

@dataclass
class BinSpec:
    edges: np.ndarray
    is_categorical: bool = False
    categories_: Dict[str, float] = None

class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins: int = 10, min_bin_frac: float = 0.03, rare_frac: float = 0.01, alpha: float = 0.5):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.rare_frac = rare_frac
        self.alpha = alpha
        self.bin_specs_: Dict[str, BinSpec] = {}
        self.iv_table_: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy(); y = pd.Series(y).reset_index(drop=True)
        n = len(y); total_bad = float(np.sum(y==1)); total_good = float(np.sum(y==0)); eps=1e-9
        iv_records = []
        for col in X.columns:
            s = X[col]
            if s.dtype == "O":
                vc = s.fillna("__MISSING__").astype(str).value_counts(normalize=True)
                rare = set(vc[vc < self.rare_frac].index.tolist())
                s2 = s.fillna("__MISSING__").astype(str).map(lambda v: v if v not in rare else "__RARE__")
                woe_map = {}; ivc = 0.0
                for cat,_ in s2.value_counts().items():
                    m = (s2==cat)
                    bad=float(np.sum(y[m]==1)); good=float(np.sum(y[m]==0))
                    woe = np.log(((bad+self.alpha)/(total_bad+self.alpha))/((good+self.alpha)/(total_good+self.alpha)+eps)+eps)
                    woe_map[cat]=woe
                    ivc += ((bad/(total_bad+eps))-(good/(total_good+eps)))*woe
                self.bin_specs_[col]=BinSpec(edges=np.array([]), is_categorical=True, categories_=woe_map)
                iv_records.append({"feature": col, "IV": ivc})
            else:
                s1 = s.astype(float)
                qs = np.unique(np.nanquantile(s1, q=np.linspace(0,1,self.max_bins+1), interpolation="linear"))
                edges = np.unique(np.concatenate(([-np.inf], qs[1:-1], [np.inf])))
                if len(edges)<=2: edges=np.array([-np.inf, np.inf])
                b = pd.cut(s1, bins=edges, right=True, include_lowest=True)
                bc = b.value_counts(dropna=False).sort_index()
                while (bc/n).min()<self.min_bin_frac and len(bc)>2:
                    idx=np.argmin(bc.values)
                    if idx==0: edges=np.delete(edges,1)
                    else: edges=np.delete(edges,idx+1)
                    b = pd.cut(s1, bins=edges, right=True, include_lowest=True)
                    bc = b.value_counts(dropna=False).sort_index()
                ivc=0.0; wmap={}
                b2 = pd.cut(s1, bins=edges, right=True, include_lowest=True)
                for inter,_ in b2.value_counts().sort_index().items():
                    m=(b2==inter)
                    bad=float(np.sum(y[m]==1)); good=float(np.sum(y[m]==0))
                    woe=np.log(((bad+self.alpha)/(total_bad+self.alpha))/((good+self.alpha)/(total_good+self.alpha)+eps)+eps)
                    wmap[str(inter)]=woe
                    ivc += ((bad/(total_bad+eps))-(good/(total_good+eps)))*woe
                self.bin_specs_[col]=BinSpec(edges=np.array(edges), is_categorical=False, categories_=wmap)
                iv_records.append({"feature": col, "IV": ivc})
        self.iv_table_=pd.DataFrame(iv_records).sort_values("IV", ascending=False).reset_index(drop=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col, spec in self.bin_specs_.items():
            s = X[col]
            if spec.is_categorical:
                s2 = s.fillna("__MISSING__").astype(str)
                mapped = s2.map(lambda v: v if v in spec.categories_ else "__RARE__")
                out[col] = mapped.map(spec.categories_).fillna(0.0).astype(float)
            else:
                s1 = s.astype(float)
                b = pd.cut(s1, bins=spec.edges, right=True, include_lowest=True)
                out[col] = b.astype(str).map(spec.categories_).fillna(0.0).astype(float)
        return out

def load_data():
    train = pd.read_excel(TRAIN_XLSX)
    test  = pd.read_excel(TEST_XLSX)
    sample = pd.read_csv(SAMPLE_CSV)
    return train, test, sample

def main():
    print(">>> ExtraTrees with WOE + Undersampling")
    train, test, sample = load_data()
    target="target"; id_col="id"
    feats=[c for c in train.columns if c not in [target,id_col]]
    cat_cols=[c for c in feats if train[c].dtype=="O"]
    num_cols=[c for c in feats if c not in cat_cols]

    imp_num=SimpleImputer(strategy="median")
    imp_cat=SimpleImputer(strategy="most_frequent")

    X_num=pd.DataFrame(imp_num.fit_transform(train[num_cols]), columns=num_cols, index=train.index)
    X_cat=pd.DataFrame(imp_cat.fit_transform(train[cat_cols]), columns=cat_cols, index=train.index) if cat_cols else pd.DataFrame(index=train.index)
    X=pd.concat([X_cat, X_num], axis=1)
    y=train[target].astype(int).values

    print(f"[Info] Train={train.shape}, PosRate={np.mean(y):.4f}")
    woe=WOETransformer(max_bins=10, min_bin_frac=0.03, rare_frac=0.01, alpha=0.5).fit(X,y)
    woe.iv_table_.to_csv(os.path.join(OUT,"iv_table_extratrees.csv"), index=False, encoding="utf-8")

    Xw=woe.transform(X)

    if RandomUnderSampler is not None:
        sampler=RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=0.5)
    else:
        sampler="passthrough"
        warnings.warn("imblearn 未安装，跳过欠采样。")

    model=ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    if sampler == "passthrough":
        pipe=SkPipeline([("clf", model)])
    else:
        pipe=Pipeline([("sampler", sampler), ("clf", model)])

    skf=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof=np.zeros(len(Xw), float)
    for i,(tr,va) in enumerate(skf.split(Xw,y),1):
        p=clone(pipe)
        p.fit(Xw.iloc[tr], y[tr])
        oof[va]=p.predict_proba(Xw.iloc[va])[:,1]
        auc=roc_auc_score(y[va], oof[va]); ap=average_precision_score(y[va], oof[va]); ks=ks_score(y[va], oof[va])
        print(f"[Fold {i}] AUC={auc:.4f} AP={ap:.4f} KS={ks:.4f}")
    auc=roc_auc_score(y,oof); ap=average_precision_score(y,oof); ks=ks_score(y,oof)
    print(f"[OOF] AUC={auc:.4f} AP={ap:.4f} KS={ks:.4f}")
    pipe.fit(Xw,y)

    t_num=pd.DataFrame(imp_num.transform(test[num_cols]), columns=num_cols, index=test.index)
    t_cat=pd.DataFrame(imp_cat.transform(test[cat_cols]), columns=cat_cols, index=test.index) if cat_cols else pd.DataFrame(index=test.index)
    tX=pd.concat([t_cat, t_num], axis=1)
    tXw=woe.transform(tX)

    prob=pipe.predict_proba(tXw)[:,1]
    sub=pd.read_csv(SAMPLE_CSV)
    sub["target"]=prob
    out_path=os.path.join(OUT,"submission_extratrees.csv")
    sub.to_csv(out_path, index=False, float_format="%.8f", encoding="utf-8")
    print("✅ 概率预测文件：", out_path)

    # OOF
    pd.DataFrame({"id": train[id_col], "oof": oof, "target": y}).to_csv(os.path.join(OUT,"oof_extratrees.csv"), index=False, encoding="utf-8")

    # Feature importance
    imp = pipe.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"feature": Xw.columns, "importance": imp}).sort_values("importance", ascending=False).head(30)
    plt.figure(figsize=(9,7)); plt.barh(fi["feature"], fi["importance"]); plt.gca().invert_yaxis()
    plt.title("Top 30 Feature Importances (ExtraTrees)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "extratrees_feature_importance.png"), dpi=220); plt.close()
    print("📊 特征重要性图已保存。")

if __name__ == "__main__":
    main()
