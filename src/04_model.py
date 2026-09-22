"""Primary associations (logit CR1/t9), separate prediction model, composition."""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import OUTPUT, PROCESSED, load_sponsors, with_reference

CONTROLS = ("LogTenure + LeftCensored + C(Industry) + MeanRaceFinishPos"
            " + SeasonCentered + TeamSponsorCount")
TERMS = {"QualRaceCorrelation": "Predictability", "MeanRaceFinishPos": "Mean race finish",
         "C(Industry)[T.Automotive]": "Automotive vs Technology",
         "C(Industry)[T.Crypto & Web3]": "Crypto & Web3 vs Technology"}

def fit(panel, predictor="QualRaceCorrelation", extra="", categorical_season=False):
    controls = CONTROLS.replace("SeasonCentered", "C(Season)") if categorical_season else CONTROLS
    m = smf.logit(f"Retained ~ {controls} + {predictor}{extra}", panel).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": panel.Team}, use_t=True)
    if not m.mle_retvals["converged"] or not np.isfinite(m.params).all():
        raise RuntimeError("Logistic model did not converge")
    return m

def estimates(m, panel, terms):
    rows=[]
    for term in terms:
        for scale in ([1., float(panel[term].std())] if term in ["QualRaceCorrelation", "PredStability"] else [1.]):
            ci=m.conf_int().loc[term]*scale
            rows.append(dict(Term=term, Scale="per_unit" if scale==1 else "per_SD", Multiplier=scale,
                OR=float(np.exp(m.params[term]*scale)), CI_low=float(np.exp(ci.iloc[0])),
                CI_high=float(np.exp(ci.iloc[1])), P=float(m.pvalues[term])))
    return pd.DataFrame(rows)

def prediction_checks(panel):
    features=["LogTenure","LeftCensored","MeanRaceFinishPos","QualRaceCorrelation","SeasonCentered","TeamSponsorCount"]
    output=[]
    for label,omit in [("full",[]),("without_predictability",["QualRaceCorrelation"]),
                       ("without_performance",["QualRaceCorrelation","MeanRaceFinishPos"])]:
        cols=[c for c in features if c not in omit]
        # Fixed retrospective taxonomy; this is not a prospective forecasting evaluation.
        X=pd.get_dummies(panel[cols+["Industry"]],columns=["Industry"],drop_first=True).astype(float)
        y=panel.Retained.to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(X,y,groups=panel.Team),1):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
            model.fit(X.iloc[tr],y[tr]);pred=model.predict_proba(X.iloc[te])[:,1]
            output.append(dict(Model=label,Fold=fold,TestTeams=" | ".join(sorted(panel.iloc[te].Team.unique())),
                N=len(te),AUC=roc_auc_score(y[te],pred),Brier=brier_score_loss(y[te],pred),
                BaselineBrier=brier_score_loss(y[te],np.full(len(te),y[tr].mean()))))
    return pd.DataFrame(output)

def held_out_auc(panel):
    return float(prediction_checks(panel).query("Model == 'full'").AUC.mean())

def main():
    panel=with_reference(pd.read_csv(PROCESSED/'sponsor_panel.csv'))
    sp=load_sponsors();m=fit(panel);alt=fit(panel,'PredStability')
    coef=pd.concat([estimates(m,panel,TERMS),estimates(alt,panel,['PredStability'])],ignore_index=True)
    pred=prediction_checks(panel)
    composition=pd.crosstab(sp.Season,sp.Industry)
    OUTPUT.mkdir(exist_ok=True)
    coef.to_csv(OUTPUT/'model_coefficients.csv',index=False)
    pred.to_csv(OUTPUT/'prediction_folds.csv',index=False)
    composition.to_csv(OUTPUT/'composition_counts.csv')
    composition.div(composition.sum(axis=1),axis=0).mul(100).to_csv(OUTPUT/'composition_percent.csv')
    summary=dict(source_records=2082,corrected_records=len(sp),unique_sponsor_keys=sp.Key.nunique(),
        observations=len(panel),continued=int(panel.Retained.sum()),continuation_rate=float(panel.Retained.mean()),
        pseudo_r2=float(m.prsquared),auc=float(pred.query("Model == 'full'").AUC.mean()),
        team_clusters=panel.Team.nunique(),team_seasons=panel[['Team','Season']].drop_duplicates().shape[0])
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    text='Primary logit: CR1 cluster covariance and t(9) inference; conditional associations.\n'
    text+=json.dumps(summary,indent=2)+'\n\n'+coef.to_string(index=False,float_format=lambda v:f'{v:.6f}')
    text+='\n\nPrediction: separate L2-regularized logit; five team-grouped folds.\n'
    text+=pred.groupby('Model')[['AUC','Brier','BaselineBrier']].mean().to_string()
    text+='\n\nNo ordinary independent-observation industry chi-square test is reported.\n'
    (OUTPUT/'results.txt').write_text(text,encoding='utf-8');print(text)

if __name__=='__main__':main()
