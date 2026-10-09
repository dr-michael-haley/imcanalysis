"""Synthetic regression tests; no project assets or managed runs are used."""
import json
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
pytest.importorskip("lifelines")

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from SpatialBiologyToolkit import cox_survival as cxs
from SpatialBiologyToolkit import cox_validation as cv


def cohort(n=60, p=4):
    rng = np.random.default_rng(327)
    frame = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    # Ties are deliberately substantial; identity is still one row per patient.
    frame["duration"] = rng.integers(1, 12, size=n).astype(float)
    frame["event"] = True
    frame.index = [f"case_{i}" for i in range(n)]
    return frame


def test_tied_coxnet_path_has_every_requested_alpha_and_matches_single_fits():
    data = cohort()
    features = ["x0", "x1", "x2", "x3"]
    path = cxs.coxnet_path_coefficients(data, features, n_alphas=30,
                                      alpha_min_ratio=.001, l1_ratio=1.)
    assert path.alpha.nunique() == 30
    assert len(path) == 30 * len(features)
    assert path.alpha.min() / path.alpha.max() == pytest.approx(.001)
    assert path.status.eq("ok").all()
    X = StandardScaler().fit_transform(data[features])
    y = cxs.make_survival_array(data)
    for alpha in [path.alpha.min(), path.alpha.max()]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=1.).fit(X, y)
        actual = path[path.alpha == alpha].set_index('feature').loc[features, 'coef']
        np.testing.assert_allclose(actual, model.coef_[:, 0])


def test_failed_path_alpha_is_recorded_without_losing_later_alphas():
    data = cohort()
    original = cxs._fit_sksurv_checked
    def fail_middle(model, X, y):
        if model.alphas is not None and model.alphas[0] == .1:
            raise RuntimeError("synthetic failure")
        return original(model, X, y)
    with patch.object(cxs, '_coxnet_alpha_grid', return_value=np.array([1., .1, .01])), \
         patch.object(cxs, '_fit_sksurv_checked', side_effect=fail_middle):
        path = cxs.coxnet_path_coefficients(data, ['x0', 'x1'])
    assert path.alpha.nunique() == 3
    assert path.loc[path.alpha == .1, 'coef'].isna().all()
    assert path.loc[path.alpha == .1, 'status'].str.contains('synthetic failure').all()
    assert path.loc[path.alpha == .01, 'status'].eq('ok').all()


def test_no_valid_inner_candidate_raises_instead_of_silent_fallback():
    with patch.object(cv, '_fit', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError, match='No Cox penalty completed all'):
            cv.tune_cox_penalty(cohort(), ['x0'], model_type='ridge',
                                ridge_alphas=[.1, 1.], n_splits=3)


def test_candidate_with_failed_fold_is_ineligible():
    original = cv._fit
    def sometimes_fail(table, features, model_type, alpha, **kwargs):
        if alpha == .1:
            raise RuntimeError('weak penalty fails')
        return original(table, features, model_type, alpha, **kwargs)
    with patch.object(cv, '_fit', side_effect=sometimes_fail):
        scores, alpha = cv.tune_cox_penalty(cohort(), ['x0', 'x1'],
                                           model_type='ridge', ridge_alphas=[.1, 1.], n_splits=3)
    assert alpha == 1.
    assert not scores.loc[scores.alpha == .1, 'eligible'].any()


def test_screening_and_tuning_receive_only_their_training_patients():
    data = cohort(48)
    visits = []
    original = cv.select_training_features
    def spy(table, *args, **kwargs):
        visits.append(set(table.index))
        return original(table, *args, **kwargs)
    with patch.object(cv, 'select_training_features', side_effect=spy):
        metrics, predictions, inner = cv.cross_validate_selected_cox_model(
            data, ['x0', 'x1', 'x2'], model_type='ridge', max_image_features=2,
            ridge_alphas=[1., 10.], n_splits=3, repeats=1)
    assert metrics.fit_status.eq('ok').all()
    assert len(predictions) == len(data)
    assert all(len(v) < len(data) for v in visits)
    # Each inner screen must be contained in a corresponding outer training set.
    outer_sets = [set(data.index) - set(g.case_id) for _, g in predictions.groupby('fold')]
    assert all(any(v <= outer for outer in outer_sets) for v in visits)
    assert all(len(json.loads(s)) <= 2 for s in metrics.selected_features)
    assert set(inner.outer_fold) == {1, 2, 3}
    assert len(inner) == 3 * 3 * 2


def test_coxnet_cv_uses_fold_specific_alpha_scales():
    scores, alpha = cxs.fit_coxnet_alpha_cv(cohort(), ['x0', 'x1'], n_splits=3, n_alphas=5)
    assert len(scores) == 15
    assert scores.alpha_fraction.nunique() == 5
    assert scores.groupby('alpha_fraction').alpha.nunique().gt(1).all()
    assert alpha > 0 and scores.selected_alpha.eq(alpha).all()


def test_nonconvergence_is_not_accepted_as_success():
    import warnings
    class Model:
        def fit(self, X, y):
            warnings.warn('failed to converge', ConvergenceWarning)
    with pytest.raises(RuntimeError, match='did not converge'):
        cxs._fit_sksurv_checked(Model(), np.zeros((3, 1)), None)


def test_coxph_fit_data_stays_raw_so_predictions_are_not_scaled_twice():
    data = cohort()
    data['x0'] = data.x0 * 9 + 100
    fit = cxs.fit_cox_model(data, ['x0', 'x1'], standardize=True, penalizer=1.)
    pd.testing.assert_frame_equal(fit.fit_data[['x0', 'x1']], data[['x0', 'x1']])
    actual = cxs.build_risk_table(fit, fit.fit_data).risk_score.reindex(data.index)
    direct = fit.model.predict_log_partial_hazard(fit.scaler.transform(data[['x0', 'x1']]))
    np.testing.assert_allclose(actual, direct)
    assert not cxs.test_proportional_hazards(fit).empty


def test_summary_exposes_failed_folds_including_all_failed():
    metrics = pd.DataFrame(dict(fit_status=['failed', 'failed'], heldout_c_index=[np.nan]*2,
                                train_c_index=[np.nan]*2, n_features=[2, 2]))
    row = cxs.summarise_cv_metrics(metrics).iloc[0]
    assert row.n_folds == 0 and row.n_failed_folds == 2 and row.n_requested_folds == 2
    assert row.validation_status == 'failed'


def test_feature_heatmap_keeps_zero_and_failed_rows(tmp_path):
    import matplotlib.pyplot as plt
    path = pd.DataFrame(dict(feature=['active', 'zero', 'failed']*2,
                              alpha=[.1]*3 + [1.]*3, coef=[1., 0., np.nan, 0., 0., np.nan]))
    fig = cxs.plot_coefficient_path_heatmap(path, tmp_path/'all.png')
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ['active', 'zero', 'failed']
    assert (tmp_path/'all.png').exists()
    plt.close(fig)


def test_validation_rejects_duplicate_patient_rows():
    data = cohort(); data.index = ['same']*len(data)
    with pytest.raises(ValueError, match='unique case'):
        cv.cross_validate_selected_cox_model(data, ['x0'])


def test_kkt_check_rejects_finite_but_incorrect_solution():
    data = cohort()
    X = data[['x0', 'x1']].to_numpy()
    class BadModel:
        alpha = 1.
        def fit(self, X, y):
            self.coef_ = np.array([5., -5.])
            return self
        def predict(self, X):
            return X @ self.coef_
    with pytest.raises(RuntimeError, match='KKT residual'):
        cxs._fit_sksurv_checked(BadModel(), X, cxs.make_survival_array(data))


def test_stage_fits_all_models_and_writes_nested_audits(tmp_path):
    from SpatialBiologyToolkit.config.models import CoxConfig
    from SpatialBiologyToolkit.scripts import cox_survival as stage
    data = cohort(48, 3)
    data.attrs['image_features'] = ['x0', 'x1', 'x2']
    data.attrs['clinical_features'] = []
    config = CoxConfig(feature_sets=['image'], feature_selection_top_n=2,
                       validation_folds=3, validation_repeats=1,
                       ridge_alphas=[1., 10.], coxnet_n_alphas=5,
                       coxnet_alpha_min_ratio=.1)
    stage._run_analysis(cxs, data, tmp_path, config, None, None)
    summary = pd.read_csv(tmp_path/'survival_model_validation_summary.csv')
    assert len(summary) == 3 and summary.status.eq('ok').all()
    assert summary.n_folds.eq(3).all() and summary.n_failed_folds.eq(0).all()
    for model in ['ridge_cox', 'coxnet']:
        folder = tmp_path/'model_comparisons'/'image'/model
        assert (folder/f'{model}_coefficient_path_all_features.png').exists()
        eligibility = pd.read_csv(folder/f'{model}_feature_eligibility.csv')
        assert len(eligibility) == 3 and eligibility.included_in_final_fit.sum() == 2
        nested = pd.read_csv(folder/f'{model}_nested_alpha_cv_scores.csv')
        assert set(nested.outer_fold) == {1, 2, 3}
    path = pd.read_csv(tmp_path/'model_comparisons/image/coxnet/coxnet_path_coefficients.csv')
    assert path.alpha.nunique() == 5
    ridge_summary = pd.read_csv(tmp_path/'model_comparisons/image/ridge_cox/ridge_cox_alpha_cv_scores.csv')
    assert {'mean_heldout_c_index', 'n_folds', 'mean_train_c_index'} <= set(ridge_summary)
    assert len(ridge_summary) == 2


@pytest.mark.parametrize('model_type', ['ridge', 'coxnet'])
def test_numerical_checks_support_censoring_and_tied_times(model_type):
    data = cohort()
    data.loc[data.index[::3], 'event'] = False
    fit = cv._fit(data, ['x0', 'x1'], model_type, .1)
    assert fit.model.sbt_kkt_residual_ < .005


def test_changing_outer_test_outcomes_does_not_change_its_model():
    from sklearn.model_selection import KFold
    original = cohort(42)
    changed = original.copy()
    _, test = next(KFold(3, shuffle=True, random_state=1).split(original))
    changed.iloc[test, changed.columns.get_loc('duration')] *= 100
    kwargs = dict(model_type='ridge', max_image_features=2,
                  ridge_alphas=[1., 10.], n_splits=3, repeats=1, seed=1)
    ma, pa, _ = cv.cross_validate_selected_cox_model(original, ['x0', 'x1', 'x2'], **kwargs)
    mb, pb, _ = cv.cross_validate_selected_cox_model(changed, ['x0', 'x1', 'x2'], **kwargs)
    assert ma.iloc[0].selected_features == mb.iloc[0].selected_features
    assert ma.iloc[0].alpha == mb.iloc[0].alpha
    np.testing.assert_allclose(pa[pa.fold == 1].heldout_risk_score,
                               pb[pb.fold == 1].heldout_risk_score)


def test_all_failed_nested_search_retains_audit_rows():
    with patch.object(cv, '_fit', side_effect=RuntimeError('synthetic failure')):
        metrics, predictions, inner = cv.cross_validate_selected_cox_model(
            cohort(), ['x0', 'x1'], model_type='ridge', ridge_alphas=[1., 10.],
            n_splits=3, repeats=1)
    assert metrics.fit_status.str.startswith('failed:').all()
    assert predictions.heldout_risk_score.isna().all()
    assert len(inner) == 18 and not inner.eligible.any()


def test_clinical_features_are_retained_outside_image_screen_cap():
    data = cohort()
    selected = cv.select_training_features(data, ['x0', 'x1'], ['x2', 'x3'], max_image_features=1)
    assert {'x2', 'x3'} <= set(selected)
    assert len(selected) == 3
