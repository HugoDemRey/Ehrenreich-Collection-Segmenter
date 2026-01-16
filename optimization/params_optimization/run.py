import hashlib
import os

import numpy as np
import optuna

from app.src.metrics.metrics import TS_Evaluator
from optimization.params_optimization.segmentation import run_optimized_segmentation


def main():
    data_folder = "optimization/params_optimization/data/"

    audio_ids = ["bar20-t2-c2", "bar103-t2-c2", "bar2-t1-c1"]
    feature_name = "tempo"
    downsample_factor = 20
    metric_name = "f1"

    ts_evaluator: TS_Evaluator = TS_Evaluator(tolerance_seconds=15.0)
    if metric_name == "precision":
        metric_fn = ts_evaluator.precision
    elif metric_name == "recall":
        metric_fn = ts_evaluator.recall
    elif metric_name == "f1":
        metric_fn = ts_evaluator.f1_score
    elif metric_name == "f1_weighted":
        metric_fn = ts_evaluator.weighted_f1_score
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    id = f"{feature_name}_ds-{downsample_factor}"
    dataset_paths = [
        os.path.join(data_folder, audio_id, id, "dataset.npz") for audio_id in audio_ids
    ]
    id = f"{id}_metric-{metric_name}"

    out_path = "optimization/params_optimization/results"

    if len(audio_ids) == 1:
        out_path = os.path.join(out_path, audio_ids[0], id)
        # Deleting all previous results
    else:
        hash_combined = "_".join(audio_ids)
        hash_combined = hashlib.md5(hash_combined.encode("utf-8")).hexdigest()[:8]
        out_path = os.path.join(out_path, f"combined-{hash_combined}", id)

    os.makedirs(out_path, exist_ok=True)

    # Deleting all previous results (files and folders) in out_path
    for f in os.listdir(out_path):
        f_path = os.path.join(out_path, f)
        if os.path.isfile(f_path):
            os.remove(f_path)
        elif os.path.isdir(f_path):
            import shutil

            shutil.rmtree(f_path)

    datas = [np.load(dataset_path, allow_pickle=True) for dataset_path in dataset_paths]
    ssms = [data["ssms"] for data in datas]
    # flatten the list of ssms
    ssms = [ssm for sublist in ssms for ssm in sublist]
    srs = [data["srs"] for data in datas]
    # flatten the list of srs
    srs = [sr for sublist in srs for sr in sublist]
    t_ann_lists = [data["t_ann_lists"] for data in datas]
    # flatten the list of t_ann_lists
    t_ann_lists = [t_ann for sublist in t_ann_lists for t_ann in sublist]

    first_ssm = ssms[0]
    first_sr = srs[0]
    first_t_ann = t_ann_lists[0]

    print(f"Loaded dataset from: {dataset_paths}")
    print(f"Number of SSMs: {len(ssms)}")
    print(
        f"First SSM shape: {first_ssm.shape}, SR: {first_sr}, Annotations: {first_t_ann}"
    )
    print("Starting Optuna study...")

    best_metric = 0.0
    best_var = np.inf
    best_precision = 0.0
    best_recall = 0.0

    def objective(trial) -> float:
        nonlocal best_metric
        nonlocal best_var
        nonlocal best_precision
        nonlocal best_recall

        if best_metric == 1.0:
            return -best_metric  # Early stopping if perfect score is reached
        # Suggest parameters to optimize
        threshold = trial.suggest_float("threshold", 0.2, 0.8)
        binarize = trial.suggest_categorical("binarize", [True, False])
        kernel_size = trial.suggest_int("kernel_size", 1, 50)
        variance = trial.suggest_float("variance", 0.1, 10.0)
        peak_threshold = trial.suggest_float("peak_threshold", 0.1, 0.9)
        sigma = trial.suggest_float("sigma", 0.1, 10.0)

        # Compute segmentation on your SSMs and evaluate metric
        metric_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        print(
            f"\n\n[Trial {trial.number}] Params: threshold={threshold}, binarize={binarize}, kernel_size={kernel_size}, variance={variance}, peak_threshold={peak_threshold}, sigma={sigma}"
        )

        ncs_sto, peaks_secs_sto, t_anns_sto, srs_sto = [], [], [], []

        for i, (ssm, sr, t_ann) in enumerate(zip(ssms, srs, t_ann_lists)):
            print(
                f"    * Segment {i + 1}/{len(ssms)}, Running segmentation...", end="\r"
            )
            # This function should implement your segmentation, returning predicted segment boundaries
            nc, peaks_sec = run_optimized_segmentation(
                ssm,
                sr,
                threshold,
                binarize,
                kernel_size,
                variance,
                peak_threshold,
                sigma,
            )

            # This function should compute (tolerant) metric comparing seg_preds vs t_ann
            metric_score: float = metric_fn(t_ann, peaks_sec)
            metric_scores.append(metric_score)
            # print(f"    {metric} score: {metric:.4f}")

            precisions.append(ts_evaluator.precision(t_ann, peaks_sec))
            recalls.append(ts_evaluator.recall(t_ann, peaks_sec))

            ncs_sto.append(nc)
            peaks_secs_sto.append(peaks_sec)
            t_anns_sto.append(t_ann)
            srs_sto.append(sr)

        mean_metric = float(np.mean(metric_scores))
        var_metric = float(np.var(metric_scores))
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))

        print(
            f"\n\033[92m    * Mean {metric_name} score: {mean_metric:.4f}, Variance: {var_metric:.4f}\033[0m"
        )
        print(
            f"\033[94m    * Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}\033[0m"
        )
        save_n_trials = 5
        ncs_sto, peaks_secs_sto, t_anns_sto, srs_sto = (
            ncs_sto[:save_n_trials],
            peaks_secs_sto[:save_n_trials],
            t_anns_sto[:save_n_trials],
            srs_sto[:save_n_trials],
        )  # Save only first 5 for visualization

        if mean_metric >= best_metric:
            best_metric = mean_metric
            best_var = var_metric
            best_precision = mean_precision
            best_recall = mean_recall
            print(f"Saving {save_n_trials} best trial results in ", out_path)
            for i, (nc, peaks_sec, t_ann, sr) in enumerate(
                zip(ncs_sto, peaks_secs_sto, t_anns_sto, srs_sto)
            ):
                os.makedirs(
                    os.path.join(out_path, f"trial{trial.number}"), exist_ok=True
                )
                if (
                    nc is not None
                    and peaks_sec is not None
                    and t_ann is not None
                    and sr is not None
                ):
                    nc.plot(
                        x_axis_type="time",
                        novelty_name=f"Novelty Curve - Last Segment",
                        time_annotations=t_ann,
                        peaks=peaks_sec
                        * sr,  # Because the plot function expects sample indices
                        save_path=f"{out_path}/trial{trial.number}/novelty_curve_seg_{i}.png",
                    )
                if t_ann is not None and peaks_sec is not None:
                    ts_evaluator.plot_evaluation(
                        y_true=t_ann,
                        y_pred=peaks_sec,
                        save_path=f"{out_path}/trial{trial.number}/evaluation_seg_{i}.png",
                    )

        # Minimization Problem
        return -mean_metric

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=250, n_jobs=1, show_progress_bar=False)
    print("Optuna study complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {-study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # Save study results
    import json

    results_path = f"{out_path}/optuna_study_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "datasets": audio_ids,
                "metric": metric_name,
                "feature_name": feature_name,
                "downsample_factor": downsample_factor,
                "best_trial_number": study.best_trial.number,
                "best_value": -study.best_value,
                "variance": best_var,
                "peak_threshold": study.best_trial.params["peak_threshold"],
                "precision": best_precision,
                "recall": best_recall,
                "best_params": study.best_trial.params,
            },
            f,
            indent=4,
        )

    best_novelty_curves = []
    datas = [np.load(dataset_path, allow_pickle=True) for dataset_path in dataset_paths]
    ssms = [data["ssms"] for data in datas]
    # flatten the list of ssms
    ssms = [ssm for sublist in ssms for ssm in sublist]
    srs = [data["srs"] for data in datas]
    # flatten the list of srs
    srs = [sr for sublist in srs for sr in sublist]
    t_ann_lists = [data["t_ann_lists"] for data in datas]
    # flatten the list of t_ann_lists
    t_ann_lists = [t_ann for sublist in t_ann_lists for t_ann in sublist]

    for i, (ssm, sr, t_ann) in enumerate(zip(ssms, srs, t_ann_lists)):
        print(f"Generating best segmentation for segment {i + 1}/{len(ssms)}...")
        nc, _ = run_optimized_segmentation(
            ssm,
            sr,
            study.best_trial.params["threshold"],
            study.best_trial.params["binarize"],
            study.best_trial.params["kernel_size"],
            study.best_trial.params["variance"],
            study.best_trial.params["peak_threshold"],
            study.best_trial.params["sigma"],
        )
        # Save the best novelty curves
        best_novelty_curves.append(nc)

    np.savez_compressed(
        f"{out_path}/best_novelty_curves.npz",
        novelty_curves=np.array(best_novelty_curves, dtype=object),
        srs=np.array(srs),
        t_ann_lists=np.array(t_ann_lists, dtype=object),
    )


if __name__ == "__main__":
    main()
