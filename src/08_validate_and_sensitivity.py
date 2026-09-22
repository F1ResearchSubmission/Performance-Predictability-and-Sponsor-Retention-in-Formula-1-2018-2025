"""Verify source fixes, exclusions, model invariants, and specification sensitivity."""
import importlib
import hashlib
import json
import platform
from importlib.metadata import version
import pandas as pd
import numpy as np
from common import ROOT, RAW, DATA, PROCESSED, OUTPUT, METRICS, load_sponsors, with_reference

def main():
    p=with_reference(pd.read_csv(PROCESSED/'sponsor_panel.csv'))
    sp=load_sponsors();met=pd.read_csv(METRICS)
    checks=[]
    def check(condition,message):
        if not condition: raise AssertionError(message)
        checks.append('PASS '+message)
    for r in pd.read_csv(ROOT/'data'/'raw_sha256.csv').itertuples(index=False):
        check(hashlib.sha256((ROOT/r.Path).read_bytes()).hexdigest()==r.SHA256,'Original preserved: '+r.Path)
    check(len(sp)==2083,'2083 corrected sponsor-team-season records')
    check(len(p)==1769,'1769 observable transitions')
    check(not p.duplicated(['Team','Season','Key']).any(),'Unique sponsor-team-season panel keys')
    check(set(p.Season)==set(range(2018,2025)),'Only 2018-2024 origin performance enters panel')
    check(len(met)==70 and met.Season.max()==2024,'70 performance team-seasons; all 2025 performance quarantined')
    for year in [2018,2019]:
        r=p[(p.Team=='McLaren')&(p.Season==year)&(p.Key=='volvotrucks')]
        check(len(r)==1 and r.Retained.iloc[0]==1,f'McLaren Volvo Trucks continuation {year}')
    r=p[(p.Team=='McLaren')&(p.Season==2020)&(p.Key=='volvotrucks')].iloc[0]
    check(r.Tenure==3 and r.LeftCensored==1,'Volvo tenure/censoring follows corrected uninterrupted sequence')
    check(len(sp[(sp.Team=='Haas')&(sp.Season==2020)&(sp.Key=='bluedef')])==1,'2020 BlueDEF is one sponsor')
    check(not ((sp.Team=='Haas')&sp.Key.isin(['blue','def'])).any(),'No Blue/DEF fragments remain in matched data')
    for key in ['peak','bluedef']:
        r=p[(p.Team=='Haas')&(p.Season==2018)&(p.Key==key)]
        check(len(r)==1 and r.Retained.iloc[0]==1,'2018 Haas presence and continuation: '+key)
    check(p[['QualRaceCorrelation','MeanRaceFinishPos','LogTenure','PredStability']].notna().all().all(),'No missing main model covariates')
    check(set(p.Retained)=={0,1},'Binary continuation outcomes')
    model=importlib.import_module('04_model');base=model.fit(p)
    check(base.use_t and base.df_resid_inference==9,'Main inference uses t(9)')
    # Centering the year must not change the likelihood or predictions.
    import statsmodels.formula.api as smf
    uncentered=smf.logit('Retained ~ '+model.CONTROLS.replace('SeasonCentered','Season')+' + QualRaceCorrelation',p).fit(disp=0)
    check(np.allclose(base.predict(),uncentered.predict(),atol=1e-7),'Centering year preserves fitted probabilities')
    # Independently validate the bootstrap's linear algebra against statsmodels.
    boot=importlib.import_module('07_few_cluster_inference')
    import patsy
    by,bx=patsy.dmatrices('Retained ~ '+boot.RHS,p,return_type='dataframe')
    import statsmodels.api as sm
    ols=sm.OLS(by.to_numpy().ravel(),bx).fit(cov_type='cluster',cov_kwds={'groups':p.Team},use_t=True)
    x=bx.to_numpy();y=by.to_numpy().ravel();pinv=np.linalg.pinv(x);beta=pinv@y;bread=pinv@pinv.T
    for term in boot.TERMS:
        k=list(bx.columns).index(term)
        se=boot.cr1_t(x,y-x@beta,p.Team.to_numpy(),bread,k)
        check(np.isclose(beta[k],ols.params.iloc[k],rtol=1e-7,atol=1e-9) and
              np.isclose(se,ols.bse.iloc[k],rtol=1e-6,atol=1e-9),'Bootstrap coefficient/CR1 SE agrees with statsmodels: '+term)
    rows=[]
    for name,extra,cat in [('primary','',False),('year_indicators','',True),
                           ('team_indicators',' + C(Team)',False),('team_and_year_indicators',' + C(Team)',True)]:
        m=model.fit(p,extra=extra,categorical_season=cat)
        estimates=model.estimates(m,p,model.TERMS)
        estimates.insert(0,'Specification',name);rows.append(estimates)
    sensitivity=pd.concat(rows,ignore_index=True)
    sensitivity.to_csv(OUTPUT/'specification_sensitivity.csv',index=False)
    env={'python':platform.python_version(),**{n:version(n) for n in ['numpy','pandas','scipy','statsmodels','scikit-learn','matplotlib','patsy']}}
    (OUTPUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'validation.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    print('\n'.join(checks));print(sensitivity.to_string(index=False))

if __name__=='__main__':main()
