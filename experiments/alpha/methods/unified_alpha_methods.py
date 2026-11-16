"""All qualification-only single-alpha candidates, under one Stage 3 objective.

The evaluator is deliberately absent. Historical ensemble-target experiments
are not claimed to be reproduced: this registry isolates initialization only.
"""
import numpy as np
from initial_alpha_methods import METHODS as LEGACY, initial_alpha as legacy
from next_alpha_initialization import initial_alpha as next_initial
from conservative_shrinkage import initial_alpha as conservative
from adaptive_correlation_alpha import initial_alpha as correlation
from explore_cv_ab_switch import cv_choice

METHODS = list(LEGACY) + ['bootstrap_oob_ensemble_shrink',
    'bootstrap_conservative_quality', 'crossfit_conservative_unshrunk',
    'crossfit_conservative_shrink', 'adaptive_corr_precision',
    'top_two_expertise', 'bootstrap_expertise_50', 'B_stacked_half',
    'stacked_full', 'cv_AB_switch']

DEFINITIONS = {
 'A_uniform': ('A：均勻權重', '所有人等權，三人各 1/3。'),
 'B_stage0': ('B：品質反比', '共同 20 筆資格盲測的個人 MSE 加 0.00001，倒數正規化。'),
 'quality_shrink': ('固定一半品質收縮', '一半 A 加一半 B。'),
 'ensemble_shrink': ('最佳組合向均勻收縮', '20 筆擬合非負且總和 1 的組合，最小化混合 MSE，再與 A 各一半。'),
 'target_peer_shrink': ('目標同儕品質收縮', 'B 乘其他模型對個人候選點接近目標的指數分數，再向 A 收縮一半。'),
 'target_local_precision': ('目標附近品質反比', '資格輸出越接近目標越加重；距離尺度 0.5，加入五筆全域 MSE 的保護，再取倒數。'),
 'mse_power_1p5': ('品質反比強化 1.5', '分數為 (MSE＋0.00001) 的 −1.5 次方，正規化；比 B 更集中。'),
 'corr_precision_0p5': ('固定相關權重 0.5', '未中心化誤差二階矩與個人 MSE 對角矩陣各一半，求最小風險的非負組合。'),
 'best_agent': ('最佳單一模型', 'MSE 最低者權重 1，其他 0；所有候選點保留。'),
 'local_bias_risk': ('局部偏差風險（原 20 筆）', '資格殘差做 ridge 擬合，在候選點平均位置估偏差平方加剩餘誤差，取倒數。'),
 'marginal_information_50': ('邊際資訊混合一半', '拿掉一個模型預測後 B 混合 MSE 增加量取正值正規化，再與 B 各一半；非參與者倒推貢獻。'),
 'target_marginal_blend_50': ('目標品質與邊際資訊混合', '目標附近品質與邊際資訊權重各一半。'),
 'bootstrap_quality': ('重抽樣品質', '同一批資格點共同有放回抽樣 1,000 次，每次算 B，再平均權重。'),
 'bootstrap_oob_quality_shrink': ('品質留出收縮', '抽中資料算 B，未抽中資料選向 A 收縮比例，平均 1,000 次。'),
 'exponential_quality': ('指數品質權重', 'exp[−(MSE−最小 MSE)/τ] 正規化；τ＝max(中位 MSE,0.00001)。'),
 'bootstrap_oob_ensemble_shrink': ('最佳組合留出收縮', '抽中資料擬合混合 MSE 最佳組合，未抽中資料選向 A 收縮比例，平均 1,000 次。'),
 'bootstrap_conservative_quality': ('保守品質', '1,000 次抽樣的個人 MSE 第 90 百分位取倒數；不是信賴上界。'),
 'crossfit_conservative_unshrunk': ('切分保守品質（不收縮）', '20 次五折切分，各用 16 筆估保守權重，平均 100 組。'),
 'crossfit_conservative_shrink': ('保守品質＋留出收縮', '相同 100 個切分，以另外 4 筆選向 A 收縮比例，再平均權重。'),
 'adaptive_corr_precision': ('自適應相關權重', '一次五折，16 筆擬合、4 筆評混合 MSE，從 0／0.25／0.5／0.75／1 選矩陣強度，全量重估。'),
 'top_two_expertise': ('只保留最佳兩人', 'B 中 MSE 最低兩人保留並重新正規化，其他為 0；仍用共同 Stage 3 目標。'),
 'bootstrap_expertise_50': ('重抽樣勝出頻率混合', '共同抽樣 512 次，統計每人 MSE 最低的頻率，與 B 各一半；不同於平均倒數權重。'),
 'B_stacked_half': ('最佳組合向 B 收縮', '全 20 筆混合 MSE 最佳組合與 B 各一半。'),
 'stacked_full': ('純最佳組合', '全 20 筆求非負且總和 1 的混合 MSE 最佳組合，不收縮。'),
 'cv_AB_switch': ('留一驗證選 A 或 B', '逐筆留一，以另外 19 筆估 B，留出混合 MSE 與 A 比較，選一法；平手選 A。'),
}

class CachedAgent:
    def __init__(self, agent, x, prediction):
        self.agent_id = agent.agent_id
        self.agent, self.x, self.prediction = agent, x, prediction

    def api_predict(self, values):
        x = np.asarray(values)
        if x.shape == self.x.shape and np.array_equal(x, self.x):
            return self.prediction.copy()
        return self.agent.api_predict(values)

def initial_alpha(method, agents, mse, blind_x, blind_y, proposals, target, **kwargs):
    if method not in METHODS:
        raise ValueError('Unknown unified alpha method')
    x, y = np.asarray(blind_x), np.asarray(blind_y)
    p = kwargs.get('cached_blind_predictions')
    if p is not None:
        p = np.asarray(p, float)
        if p.shape != (len(y), len(agents)) or not np.isfinite(p).all():
            raise ValueError('Qualification cache shape/values mismatch')
        agents = [CachedAgent(a, x, p[:,i]) for i,a in enumerate(agents)]
    seed = kwargs.get('bootstrap_seed', 20261011)
    if method in ['top_two_expertise','bootstrap_expertise_50','B_stacked_half','stacked_full','cv_AB_switch']:
        base, audit = legacy('B_stage0', agents, mse, x, y, proposals, target, **kwargs)
        if method == 'top_two_expertise':
            selected = np.argsort(mse, kind='stable')[:min(2,len(agents))]
            w = np.zeros(len(agents)); w[selected] = base[selected]; w /= w.sum()
            audit['selected_indices'] = selected.tolist()
        else:
            if p is None: p = np.column_stack([a.api_predict(x) for a in agents])
            if method == 'bootstrap_expertise_50':
                rows = np.random.default_rng(seed).integers(0,len(y),(512,len(y)))
                winners = np.argmin(((p-y[:,None])**2)[rows].mean(axis=1),axis=1)
                frequency = np.bincount(winners,minlength=len(agents))/512
                w = .5*base + .5*frequency
                audit.update(winner_frequency=frequency.tolist(), expertise_replicates=512)
            elif method == 'cv_AB_switch':
                choice, cv = cv_choice(p,y,mse)
                w = base if choice=='B_stage0' else np.full(len(agents),1/len(agents))
                audit.update(cv, selected_method=choice)
            else:
                _, fitted = legacy('ensemble_shrink',agents,mse,x,y,proposals,target,**kwargs)
                if fitted['fallback_reason']:
                    w = base; audit['fallback_reason'] = 'ensemble_fit_failed_use_B'
                else:
                    combo = np.asarray(fitted['raw_scores'])
                    w = combo if method=='stacked_full' else .5*base+.5*combo
                    audit['ensemble_fit'] = fitted
    else:
        function = (correlation if method in ['corr_precision_0p5','adaptive_corr_precision'] else
                    conservative if method.startswith('crossfit_') else
                    next_initial if method in ['bootstrap_oob_ensemble_shrink','bootstrap_conservative_quality'] else legacy)
        w, audit = function(method,agents,mse,x,y,proposals,target,**kwargs)
    w = np.asarray(w,float)
    if w.shape != (len(agents),) or not np.isfinite(w).all() or np.min(w)<0 or abs(w.sum()-1)>1e-8:
        raise ValueError('Invalid alpha simplex')
    audit.update(unified_method=method, definition=DEFINITIONS[method][1],
                 bootstrap_seed=int(seed), qualification_cache_available=p is not None)
    return w,audit
