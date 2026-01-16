import os

import numpy as np
import optuna

from app.src.metrics.metrics import TS_Evaluator
from optimization.nc_combination_optimization.combination import run_combination

# Paths to novelty curve results
id = "combined_f1_beta_0.5_2"

chroma_ncs_path = f"C:\\Users\\Hugo\\Repositories\\Desktop\\EPFL\\DCML\\Ehrenreich\\ehrenreich-collection-semester-project\\optimization\\params_optimization\\results\\combined-234cd2b3\\chroma_ds-20_metric-f1\\test2\\nc.npy"
tempo_ncs_path = f"C:\\Users\\Hugo\\Repositories\\Desktop\\EPFL\\DCML\\Ehrenreich\\ehrenreich-collection-semester-project\\optimization\\params_optimization\\results\\combined-234cd2b3\\mfcc20_ds-20_metric-f1\\test2\\nc.npy"
mfcc_ncs_path = f"C:\\Users\\Hugo\\Repositories\\Desktop\\EPFL\\DCML\\Ehrenreich\\ehrenreich-collection-semester-project\\optimization\\params_optimization\\results\\combined-234cd2b3\\tempo_ds-20_metric-f1\\test1\\nc.npy"


def load_ncs(path):
    from app.src.audio_features.features import NoveltyCurve

    data = np.load(
        path, allow_pickle=True
    ).item()  # .item() extracts the dictionary from 0-d array

    raw_curves = data["novelty_curves"]
    srs = data["srs"]
    t_ann_lists = data["t_ann_lists"]

    # Reconstruct NoveltyCurve objects from raw numpy arrays
    novelty_curves = []
    for curve, sr in zip(raw_curves, srs):
        novelty_curves.append(NoveltyCurve(curve, sr))

    return novelty_curves, srs, t_ann_lists


chroma_ncs, srs, t_ann_lists = load_ncs(chroma_ncs_path)
tempo_ncs, _, _ = load_ncs(tempo_ncs_path)
mfcc_ncs, _, _ = load_ncs(mfcc_ncs_path)

metric_name = "f1"  # or "precision", "recall"
ts_evaluator = TS_Evaluator(tolerance_seconds=15.0)
if metric_name == "precision":
    metric_fn = ts_evaluator.precision
elif metric_name == "recall":
    metric_fn = ts_evaluator.recall
elif metric_name == "f1":
    metric_fn = ts_evaluator.f1_score
else:
    raise ValueError(f"Unknown metric: {metric_name}")

import time

date_str = time.strftime("%Y%m%d-%H%M%S")

out_path = (
    f"optimization/nc_combination_optimization/results/study_{metric_name}_{date_str}"
)
os.makedirs(out_path, exist_ok=True)


# Store best results in a mutable dict
best_results = {"metric": -np.inf, "var": None, "precision": None, "recall": None}

# Global variable for objective function to access current output path
objective_out_path = None


def objective(trial):
    w_chroma = trial.suggest_float("w_chroma", 0.0, 1.0)
    w_mfcc = trial.suggest_float("w_mfcc", 0.0, 1.0)
    w_tempo = trial.suggest_float("w_tempo", 0.0, 1.0)
    peak_threshold = trial.suggest_float("peak_threshold", 0.1, 1.0)
    method = trial.suggest_categorical("method", ["mean", "max", "weighted"])

    # Normalize weights
    total = w_chroma + w_mfcc + w_tempo
    if total == 0:
        w_chroma, w_mfcc, w_tempo = 1.0, 0.0, 0.0
    else:
        w_chroma /= total
        w_mfcc /= total
        w_tempo /= total

    metric_scores = []
    precisions = []
    recalls = []
    combined_ncs_sto, peaks_secs_sto, t_anns_sto, srs_sto = [], [], [], []

    for i, (chroma_nc, mfcc_nc, tempo_nc, sr, t_ann) in enumerate(
        zip(chroma_ncs, mfcc_ncs, tempo_ncs, srs, t_ann_lists)
    ):
        combined_nc, peaks_sec = run_combination(
            method,
            chroma_nc,
            mfcc_nc,
            tempo_nc,
            sr,
            w_chroma,
            w_mfcc,
            w_tempo,
            peak_threshold,
        )
        metric_score = metric_fn(t_ann, peaks_sec)
        metric_scores.append(metric_score)
        precisions.append(ts_evaluator.precision(t_ann, peaks_sec))
        recalls.append(ts_evaluator.recall(t_ann, peaks_sec))
        combined_ncs_sto.append(combined_nc)
        peaks_secs_sto.append(peaks_sec)
        t_anns_sto.append(t_ann)
        srs_sto.append(sr)

    mean_metric = float(np.mean(metric_scores))
    var_metric = float(np.var(metric_scores))
    mean_precision = float(np.mean(precisions))
    mean_recall = float(np.mean(recalls))

    # Save best trial results for visualization
    if mean_metric > best_results["metric"]:
        best_results["metric"] = mean_metric
        best_results["var"] = var_metric
        best_results["precision"] = mean_precision
        best_results["recall"] = mean_recall
        # Use the global objective_out_path that will be set by main function
        trial_dir = os.path.join(objective_out_path, f"trial{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        save_n_trials = 5
        for i, (nc, peaks_sec, t_ann, sr) in enumerate(
            zip(
                combined_ncs_sto[:save_n_trials],
                peaks_secs_sto[:save_n_trials],
                t_anns_sto[:save_n_trials],
                srs_sto[:save_n_trials],
            )
        ):
            if (
                nc is not None
                and peaks_sec is not None
                and t_ann is not None
                and sr is not None
            ):
                nc.plot(
                    x_axis_type="time",
                    novelty_name=f"Combined Novelty Curve",
                    time_annotations=t_ann,
                    peaks=peaks_sec * sr,
                    save_path=f"{trial_dir}/combined_novelty_curve_seg_{i}.png",
                )
            if t_ann is not None and peaks_sec is not None:
                ts_evaluator.plot_evaluation(
                    y_true=t_ann,
                    y_pred=peaks_sec,
                    save_path=f"{trial_dir}/evaluation_seg_{i}.png",
                )

    return -mean_metric


def main(n_trials=500):
    """Main optimization function that can be called with custom trial count."""
    import json
    import time

    # Create unique output path
    date_str = time.strftime("%Y%m%d-%H%M%S")
    test_folder = f"test_{date_str}"
    current_out_path = f"{out_path}/{test_folder}"
    os.makedirs(current_out_path, exist_ok=True)

    # Set global variable for objective function
    global objective_out_path, best_results
    objective_out_path = current_out_path

    # Reset best_results for this run
    best_results = {"metric": -np.inf, "var": None, "precision": None, "recall": None}

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print("Optuna study complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {-study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # Save study results
    results_path = f"{current_out_path}/optuna_study_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "metric": metric_name,
                "best_trial_number": study.best_trial.number,
                "best_value": -study.best_value,
                "variance": best_results["var"],
                "precision": best_results["precision"],
                "recall": best_results["recall"],
                "best_params": study.best_trial.params,
            },
            f,
            indent=4,
        )

    # Generate best combined novelty curves for all segments
    best_novelty_curves = []
    for i, (chroma_nc, mfcc_nc, tempo_nc, sr, t_ann) in enumerate(
        zip(chroma_ncs, mfcc_ncs, tempo_ncs, srs, t_ann_lists)
    ):
        combined_nc, _ = run_combination(
            study.best_trial.params["method"],
            chroma_nc,
            mfcc_nc,
            tempo_nc,
            sr,
            study.best_trial.params["w_chroma"],
            study.best_trial.params["w_mfcc"],
            study.best_trial.params["w_tempo"],
            study.best_trial.params["peak_threshold"],
        )
        best_novelty_curves.append(combined_nc)

    np.savez_compressed(
        f"{current_out_path}/best_combined_novelty_curves.npz",
        novelty_curves=np.array(best_novelty_curves, dtype=object),
        srs=np.array(srs),
        t_ann_lists=np.array(t_ann_lists, dtype=object),
    )

    # Return results for multiple runs analysis
    return {
        "best_trial_number": study.best_trial.number,
        "best_value": -study.best_value,
        "variance": best_results["var"],
        "precision": best_results["precision"],
        "recall": best_results["recall"],
        "best_params": study.best_trial.params,
        "test_folder": current_out_path,
        "metric": metric_name,
    }


if __name__ == "__main__":
    main()
