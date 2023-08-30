from array import array
import pandas as pa
import numpy as np
from sklearn.metrics import roc_auc_score,matthews_corrcoef,cohen_kappa_score
from  sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from self_paced_ensemble.self_paced_ensemble.self_paced_ensemble import SelfPacedEnsembleClassifier
from self_paced_ensemble.canonical_ensemble.canonical_ensemble import SMOTEBaggingClassifier, \
    RUSBoostClassifier, UnderBaggingClassifier,BalanceCascadeClassifier
from self_paced_ensemble.canonical_resampling.canonical_resampling import ResampleClassifier
import math
def Gmean(tru,pre):
    s_t =f_t =0
    # print(tru)
    tu = pa.DataFrame(tru,columns = ['1'])
    f = tu['1'].value_counts().values[0]
    s = tu['1'].value_counts().values[1]
    for i in range(len(pre)):
        if(pre[i]==1 and tru[i]==1):
            f_t = f_t+1
        elif(pre[i]==0 and tru[i]==0):
            s_t = s_t+1
    return math.sqrt(float(s_t*1.0*f_t/s/f))
def CART_HD(end_data, test,num,train):
    accuracy = f1 = roc = mcc = kappa = gmean = 0
    accuracy_under = f1_under = roc_under = mcc_under = kappa_under = gmean_under = 0
    accuracy_bcc = f1_bcc = roc_bcc = mcc_bcc = kappa_bcc = gmean_bcc = 0
    accuracy_bc = f1_bc = roc_bc = mcc_bc = kappa_bc = gmean_bc = 0
    accuracy_smb = f1_smb = roc_smb = mcc_smb = kappa_smb = gmean_smb = 0
    accuracy_rc = f1_rc = roc_rc = mcc_rc = kappa_rc = gmean_rc = 0
    accuracy_rusb = f1_rusb = roc_rusb = mcc_rusb = kappa_rusb = gmean_rusb = 0
    accuracy_smo = f1_smo = roc_smo = mcc_smo = kappa_smo = gmean_smo = 0
    origin = []
    under1 = []
    bcc1 = []
    bc1 = []
    smb1 = []
    rc1 = []
    rusb1 = []
    smo1 = []
    for iter in range(len(end_data)):
        # ===========================================开始集成===========================================
        classification = list()
        for i in range(num):
            classification.append(DecisionTreeClassifier())
            classification[i].fit(end_data[iter][i][:, :-1], end_data[iter][i][:, -1])
        result = list()
        for i in test[iter].values:

            toupiao = [0, 0]
            for l in range(num):
                if (classification[l].predict(i[:-1].reshape(1, -1)) > 0):
                    toupiao[1] += 1
                else:
                    toupiao[0] += 1
            result.append(np.argmax(toupiao))
        met = []
        accuracy += metrics.accuracy_score(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        f1 += metrics.f1_score(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        roc += roc_auc_score(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        mcc += matthews_corrcoef(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        kappa += cohen_kappa_score(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        gmean += Gmean(test[iter].iloc[:,-1].values,np.array(result).reshape(-1,1))
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        met.append(Gmean(test[iter].iloc[:, -1].values, np.array(result).reshape(-1, 1)))
        origin.append(met)
        # ===========================================UnderBaggingClassifier======================
        met = []
        under = UnderBaggingClassifier(base_estimator=DecisionTreeClassifier())
        under.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = under.predict(test[iter].iloc[:, :-1].values)
        accuracy_under += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_under += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_under += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_under += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_under += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_under += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        under1.append(met)
        # ===========================================BalanceCascadeClassifier======================
        met = []
        bcc = BalanceCascadeClassifier(base_estimator=DecisionTreeClassifier())
        bcc.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = bcc.predict(test[iter].iloc[:, :-1].values)
        accuracy_bcc += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_bcc += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_bcc += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_bcc += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_bcc += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_bcc += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        bcc1.append(met)
        # ===========================================BaggingClassifier======================
        met = []
        bc = BaggingClassifier(base_estimator=DecisionTreeClassifier())
        bc.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = bc.predict(test[iter].iloc[:, :-1].values)
        accuracy_bc += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_bc += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_bc += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_bc += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_bc += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_bc += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        bc1.append(met)
        # =============================SelfPacedEnsembleClassifier===========================
        met = []
        smb = SelfPacedEnsembleClassifier(base_estimator=DecisionTreeClassifier())
        smb.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = smb.predict(test[iter].iloc[:, :-1].values)
        accuracy_smb += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_smb += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_smb += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_smb += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_smb += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_smb += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        smb1.append(met)
        # ===========================================ResampleClassifier======================
        met = []
        rc = ResampleClassifier(base_estimator=DecisionTreeClassifier())
        rc.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values, by='RUS')
        pre = rc.predict(test[iter].iloc[:, :-1].values)
        accuracy_rc += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_rc += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_rc += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_rc += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_rc += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_rc += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        rc1.append(met)
        # ===========================================RUSBoostClassifier======================
        met = []
        rusb = RUSBoostClassifier(base_estimator=DecisionTreeClassifier())
        rusb.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = rusb.predict(test[iter].iloc[:, :-1].values)
        accuracy_rusb += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_rusb += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_rusb += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_rusb += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_rusb += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_rusb += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        rusb1.append(met)
        # ===========================================SMOTEBaggingClassifier======================
        met = []
        smo = SMOTEBaggingClassifier(base_estimator=DecisionTreeClassifier())
        smo.fit(train[iter].iloc[:, :-1].values, train[iter].iloc[:, -1].values)
        pre = smo.predict(test[iter].iloc[:, :-1].values)
        accuracy_smo += metrics.accuracy_score(test[iter].iloc[:, -1].values, pre)
        f1_smo += metrics.f1_score(test[iter].iloc[:, -1].values, pre)
        roc_smo += roc_auc_score(test[iter].iloc[:, -1].values, pre)
        mcc_smo += matthews_corrcoef(test[iter].iloc[:, -1].values, pre)
        kappa_smo += cohen_kappa_score(test[iter].iloc[:, -1].values, pre)
        gmean_smo += Gmean(test[iter].iloc[:, -1].values, pre)
        met.append(metrics.accuracy_score(test[iter].iloc[:, -1].values, pre))
        met.append(metrics.f1_score(test[iter].iloc[:, -1].values, pre))
        met.append(roc_auc_score(test[iter].iloc[:, -1].values, pre))
        met.append(matthews_corrcoef(test[iter].iloc[:, -1].values, pre))
        met.append(cohen_kappa_score(test[iter].iloc[:, -1].values, pre))
        met.append(Gmean(test[iter].iloc[:, -1].values, pre))
        smo1.append(met)
    origin = np.std(np.array(origin), axis=0)
    under = np.std(np.array(under1), axis=0)
    bcc = np.std(np.array(bcc1), axis=0)
    bc = np.std(np.array(bc1), axis=0)
    smb = np.std(np.array(smb1), axis=0)
    rc = np.std(np.array(rc1), axis=0)
    rusb = np.std(np.array(rusb1), axis=0)
    smo = np.std(np.array(smo1), axis=0)
    print("=======================滑动===========================")
    print("accuracy", accuracy / 10)
    print("f1", f1 / 10)
    print("roc", roc / 10)
    print("mcc", mcc / 10)
    print("kappa", kappa / 10)
    print("gmean", gmean / 10)
    print("accuracy_std", origin[0])
    print("f1_std", origin[1])
    print("roc_std", origin[2])
    print("mcc_std", origin[3])
    print("kappa_std", origin[4])
    print("gmean_std", origin[5])
    print("=======================UnderBaggingClassifier===========================")
    print("accuracy", accuracy_under / 10)
    print("f1", f1_under / 10)
    print("roc", roc_under / 10)
    print("mcc", mcc_under / 10)
    print("kappa", kappa_under / 10)
    print("gmean", gmean_under / 10)
    print("accuracy_std", under[0])
    print("f1_std", under[1])
    print("roc_std", under[2])
    print("mcc_std", under[3])
    print("kappa_std", under[4])
    print("gmean_std", under[5])
    print("=======================BalanceCascadeClassifier===========================")
    print("accuracy", accuracy_bcc / 10)
    print("f1", f1_bcc / 10)
    print("roc", roc_bcc / 10)
    print("mcc", mcc_bcc / 10)
    print("kappa", kappa_bcc / 10)
    print("gmean", gmean_bcc / 10)
    print("accuracy_std", bcc[0])
    print("f1_std", bcc[1])
    print("roc_std", bcc[2])
    print("mcc_std", bcc[3])
    print("kappa_std", bcc[4])
    print("gmean_std", bcc[5])
    print("=======================BaggingClassifier===========================")
    print("accuracy", accuracy_bc / 10)
    print("f1", f1_bc / 10)
    print("roc", roc_bc / 10)
    print("mcc", mcc_bc / 10)
    print("kappa", kappa_bc / 10)
    print("gmean", gmean_bc / 10)
    print("accuracy_std", bc[0])
    print("f1_std", bc[1])
    print("roc_std", bc[2])
    print("mcc_std", bc[3])
    print("kappa_std", bc[4])
    print("gmean_std", bc[5])
    print("=======================SelfPacedEnsembleClassifier===========================")
    print("accuracy", accuracy_smb / 10)
    print("f1", f1_smb / 10)
    print("roc", roc_smb / 10)
    print("mcc", mcc_smb / 10)
    print("kappa", kappa_smb / 10)
    print("gmean", gmean_smb / 10)
    print("accuracy_std", smb[0])
    print("f1_std", smb[1])
    print("roc_std", smb[2])
    print("mcc_std", smb[3])
    print("kappa_std", smb[4])
    print("gmean_std", smb[5])
    print("=======================ResampleClassifier===========================")
    print("accuracy", accuracy_rc / 10)
    print("f1", f1_rc / 10)
    print("roc", roc_rc / 10)
    print("mcc", mcc_rc / 10)
    print("kappa", kappa_rc / 10)
    print("gmean", gmean_rc / 10)
    print("accuracy_std", rc[0])
    print("f1_std", rc[1])
    print("roc_std", rc[2])
    print("mcc_std", rc[3])
    print("kappa_std", rc[4])
    print("gmean_std", rc[5])
    print("=======================RUSBoostClassifier===========================")
    print("accuracy", accuracy_rusb / 10)
    print("f1", f1_rusb / 10)
    print("roc", roc_rusb / 10)
    print("mcc", mcc_rusb / 10)
    print("kappa", kappa_rusb / 10)
    print("gmean", gmean_rusb / 10)
    print("accuracy_std", rusb[0])
    print("f1_std", rusb[1])
    print("roc_std", rusb[2])
    print("mcc_std", rusb[3])
    print("kappa_std", rusb[4])
    print("gmean_std", rusb[5])

    print("=======================SMOTEBaggingClassifier===========================")
    print("accuracy", accuracy_smo / 10)
    print("f1", f1_smo / 10)
    print("roc", roc_smo / 10)
    print("mcc", mcc_smo / 10)
    print("kappa", kappa_smo / 10)
    print("gmean", gmean_smo / 10)
    print("accuracy_std", smo[0])
    print("f1_std", smo[1])
    print("roc_std", smo[2])
    print("mcc_std", smo[3])
    print("kappa_std", smo[4])
    print("gmean_std", smo[5])