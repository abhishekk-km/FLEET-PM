import _bootstrap

import argparse
import json
import os
import shutil
import time
import traceback

import joblib
import mlflow
import numpy as np

from sklearn.model_selection import GroupKFold

from fleetpm.config import load_config, art
from fleetpm.pipeline import load_prepared, feat_cols
from fleetpm.models import rul_tabular as T
from fleetpm.models.rul_seq import SeqRegressor
from fleetpm.models.metrics import summarize, cut_eval, last_rows
from fleetpm.features.sequences import build_sequences
from fleetpm.mlops import tracking as mt


def log(message):
    """Print progress immediately, even when output is piped."""
    print(message, flush=True)


def aggregate_metrics(fold_metrics):
    """Aggregate metrics across cross-validation folds."""
    if not fold_metrics:
        raise ValueError("No cross-validation metrics were produced.")

    return {
        **{
            f"cv_{key}": float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
        },
        **{
            f"cv_{key}_std": float(
                np.std([fold[key] for fold in fold_metrics])
            )
            for key in fold_metrics[0]
        },
    }


def train_subset(fd, cfg, args):
    """Train and evaluate RUL models for one FD001/FD002/etc. subset."""

    start_time = time.time()
    rul_cfg = cfg["rul"]
    eval_cfg = cfg["eval"]
    seed = cfg["seed"]

    log(f"\n{'=' * 65}")
    log(f"Starting RUL training for {fd}")
    log(f"{'=' * 65}")

    log(f"[{fd}] Loading prepared data...")
    P = load_prepared(cfg, fd)

    cap = P["cap"]
    window = cfg["features"]["window"]

    F = P["Ftr"]
    Ft = P["Fte"]

    cols = feat_cols(F) + (["cycle"] if rul_cfg["use_cycle"] else [])
    if args.stats:
        cols = [c for c in cols if c.split("__")[-1] in args.stats or c == "cycle"]
        if not cols:
            raise ValueError(f"{fd}: --stats {args.stats} matched no feature columns.")
    X = F[cols].to_numpy(dtype=float)
    y = F["rul_cap"].to_numpy(dtype=float)
    engine_ids = F.engine_id.to_numpy()

    Xt = Ft[cols].to_numpy(dtype=float)
    test_engine_ids = Ft.engine_id.to_numpy()
    last_indices = last_rows(test_engine_ids)

    if not np.isfinite(X).all() or not np.isfinite(Xt).all():
        raise ValueError(f"{fd}: Non-finite feature values detected.")

    y_raw = P["rul"]
    y_cap = np.minimum(y_raw, cap)

    splits = list(
        GroupKFold(n_splits=eval_cfg["cv_folds"]).split(
            X, y, engine_ids
        )
    )

    if not splits:
        raise ValueError(f"{fd}: No cross-validation splits generated.")

    model_names = args.models or rul_cfg["models"]

    log(f"[{fd}] Training samples: {len(X)}")
    log(f"[{fd}] Test samples: {len(Xt)}")
    log(f"[{fd}] Features: {len(cols)}")
    log(f"[{fd}] CV folds: {len(splits)}")
    log(f"[{fd}] Models: {', '.join(model_names)}")
    log(f"[{fd}] Primary model: {rul_cfg['primary']}")

    sequences = None
    results = {}

    with mlflow.start_run(run_name=f"{fd}_rul_study"):
        mlflow.log_params({
            "fd": fd,
            "cap": cap,
            "window": window,
        })
        oof = np.full(len(y), np.nan)
        for model_name in model_names:
            model_start = time.time()

            log(f"\n[{fd}] Starting model: {model_name}")

            with mlflow.start_run(
                run_name=f"{fd}_{model_name}",
                nested=True,
            ) as run:

                if model_name in ("ridge", "extratrees", "xgb"):
                    fold_metrics = []

                    for fold_index, (train_idx, val_idx) in enumerate(
                        splits, start=1
                    ):
                        log(
                            f"[{fd}/{model_name}] "
                            f"CV fold {fold_index}/{len(splits)}: fitting..."
                        )

                        fold_model = T.fit(
                            T.make(model_name, rul_cfg, seed),
                            model_name,
                            X[train_idx],
                            y[train_idx],
                            T.weights(y[train_idx], rul_cfg),
                        )

                        predictions = T.predict(
                            fold_model,
                            X[val_idx],
                        )
                        if model_name == rul_cfg["primary"]:
                            oof[val_idx] = predictions

                        metrics = cut_eval(
                            y[val_idx],
                            predictions,
                            engine_ids[val_idx],
                            eval_cfg["min_cut_len"],
                            eval_cfg["n_cut_reps"],
                            seed,
                        )

                        fold_metrics.append(metrics)

                        log(
                            f"[{fd}/{model_name}] "
                            f"Fold {fold_index} completed."
                        )

                    log(
                        f"[{fd}/{model_name}] "
                        "Fitting final model on all training data..."
                    )

                    model = T.fit(
                        T.make(model_name, rul_cfg, seed),
                        model_name,
                        X,
                        y,
                        T.weights(y, rul_cfg),
                    )

                    log(
                        f"[{fd}/{model_name}] "
                        "Predicting test engine RUL..."
                    )

                    test_predictions = T.predict(
                        model,
                        Xt[last_indices],
                    )

                else:
                    if sequences is None:
                        log(
                            f"[{fd}] Building sequences "
                            f"(window={window})..."
                        )

                        sequences = (
                            build_sequences(
                                P["Ztr"],
                                P["pre"].sensors,
                                window,
                            ),
                            build_sequences(
                                P["Zte"],
                                P["pre"].sensors,
                                window,
                            ),
                        )

                    train_sequences, test_sequences = sequences

                    train_idx, val_idx = splits[0]

                    log(
                        f"[{fd}/{model_name}] "
                        "Training sequence model on CV fold 1..."
                    )

                    fold_model = SeqRegressor(
                        model_name, cfg, cap, seed
                    ).fit(
                        train_sequences[train_idx],
                        y[train_idx],
                        engine_ids[train_idx],
                    )

                    fold_predictions = fold_model.predict(
                        train_sequences[val_idx]
                    )

                    fold_metrics = [
                        cut_eval(
                            y[val_idx],
                            fold_predictions,
                            engine_ids[val_idx],
                            eval_cfg["min_cut_len"],
                            eval_cfg["n_cut_reps"],
                            seed,
                        )
                    ]

                    log(
                        f"[{fd}/{model_name}] "
                        "Fitting final sequence model..."
                    )

                    model = SeqRegressor(
                        model_name, cfg, cap, seed
                    ).fit(
                        train_sequences,
                        y,
                        engine_ids,
                    )

                    test_predictions = model.predict(
                        test_sequences[last_indices]
                    )

                metrics = {
                    **aggregate_metrics(fold_metrics),
                    "cv_fold0_rmse": float(fold_metrics[0]["rmse"]),
                    **{
                        f"test_cap_{key}": value
                        for key, value in summarize(
                            y_cap, test_predictions
                        ).items()
                    },
                    **{
                        f"test_raw_{key}": value
                        for key, value in summarize(
                            y_raw, test_predictions
                        ).items()
                    },
                }

                results[model_name] = metrics

                mlflow.log_params({"model": model_name})
                mlflow.log_metrics(metrics)

                display_keys = (
                    "cv_rmse",
                    "cv_fold0_rmse",
                    "test_cap_rmse",
                    "test_cap_nasa",
                    "test_raw_rmse",
                )

                log(
                    f"[{fd}] {model_name}: "
                    + json.dumps({
                        key: round(float(value), 3)
                        for key, value in metrics.items()
                        if key in display_keys
                    })
                )

                if model_name == rul_cfg["primary"] and not args.stats:
                    if model_name not in T.TREE:
                        raise ValueError(
                            "The primary RUL model must be a tree model."
                        )

                    log(
                        f"[{fd}] Logging primary model to MLflow..."
                    )

                    model_uri = mt.log_sk_model(model)

                    log(
                        f"[{fd}] Fitting engine bootstrap ensemble "
                        f"(n={rul_cfg['bootstrap_n']})..."
                    )

                    ensemble = T.EngineBootstrap(
                        model_name,
                        rul_cfg,
                        rul_cfg["bootstrap_n"],
                        seed,
                    ).fit(X, y, engine_ids)

                    log(
                        f"[{fd}] Predicting with bootstrap ensemble..."
                    )

                    ensemble_mean, ensemble_std = ensemble.predict(
                        Xt[last_indices]
                    )

                    m_ok = (F.cycle.to_numpy() >= eval_cfg["min_cut_len"]) & np.isfinite(oof)
                    if not m_ok.any():
                        raise ValueError("no out-of-fold predictions for conservative calibration")
                    oe, op = (oof - y)[m_ok], oof[m_ok]
                    bins = [30, 60]; bi = np.digitize(op, bins)
                    deltas = [max(0.0, float(np.quantile(oe[bi == b], rul_cfg["cons_q"])))
                              if (bi == b).any() else 0.0 for b in range(3)]
                    conservative_predictions = np.clip(
                        test_predictions - np.array(deltas)[np.digitize(test_predictions, bins)], 0, None)

                    def diag(p):
                        out = {}
                        for nm, msk in (("gt_cap", y_raw > cap), ("le_cap", y_raw <= cap)):
                            if msk.any():
                                e = p[msk] - y_raw[msk]
                                out.update({f"diag_{nm}_n": float(msk.sum()),
                                            f"diag_{nm}_rmse": float(np.sqrt((e ** 2).mean())),
                                            f"diag_{nm}_mae": float(np.abs(e).mean()),
                                            f"diag_{nm}_bias": float(e.mean()),
                                            f"diag_{nm}_over_pct": float((e > 0).mean())})
                        return out

                    ensemble_metrics = {
                        **{f"ens_mean_cap_{k}": v for k, v in summarize(y_cap, ensemble_mean).items()},
                        **{f"ens_mean_raw_{k}": v for k, v in summarize(y_raw, ensemble_mean).items()},
                        **{f"cons_cap_{k}": v for k, v in summarize(y_cap, conservative_predictions).items()},
                        **{f"cons_raw_{k}": v for k, v in summarize(y_raw, conservative_predictions).items()},
                        "ens_spread_cov95_cap": float(np.mean(np.abs(ensemble_mean - y_cap) <= 1.96 * ensemble_std)),
                        "cons_delta_lt30": deltas[0], "cons_delta_30_60": deltas[1], "cons_delta_ge60": deltas[2],
                        **diag(test_predictions),
                    }

                    mlflow.log_metrics(ensemble_metrics)
                    results[model_name].update(ensemble_metrics)

                    log(
                        f"[{fd}] Ensemble metrics: "
                        f"{json.dumps(ensemble_metrics)}"
                    )

                    log(
                        f"[{fd}] Registering model and checking "
                        "champion promotion..."
                    )

                    promoted, version = mt.register_and_maybe_promote(
                        model_uri,
                        run.info.run_id,
                        f"{cfg['mlops']['model_prefix']}_{fd}",
                        "cv_rmse",
                        cfg["mlops"]["promote_margin"],
                    )

                    log(
                        f"[{fd}] Saving RUL model bundle..."
                    )

                    bundle = {
                        "cols": cols,
                        "base": model,
                        "ens": ensemble,
                        "name": model_name,
                        "cap": cap,
                        "version": (
                            f"{fd}-{model_name}-"
                            f"{run.info.run_id[:8]}"
                        ),
                        "cons_delta": deltas,
                        "cons_bins": bins,
                    }

                    bundle_path = art(cfg, "models", fd, "rul_bundle.joblib")
                    if promoted or not bundle_path.exists():
                        if bundle_path.exists():
                            shutil.copy2(bundle_path, bundle_path.with_name("rul_bundle.prev.joblib"))
                        tmp = bundle_path.with_name("rul_bundle.tmp.joblib")
                        joblib.dump(bundle, tmp)
                        os.replace(tmp, bundle_path)
                    else:
                        bundle_path = bundle_path.with_name("rul_bundle.candidate.joblib")
                        joblib.dump(bundle, bundle_path)
                        log(f"[{fd}] Candidate NOT promoted; live bundle untouched.")

                    log(
                        f"[{fd}] Model bundle saved: {bundle_path}"
                    )

                    log(
                        f"[{fd}] Registry version {version}; "
                        f"promoted to champion: {promoted}"
                    )

            log(
                f"[{fd}/{model_name}] Completed in "
                f"{time.time() - model_start:.1f} seconds."
            )

        if not results:
            raise ValueError(
                f"{fd}: No models were trained. Check the model config."
            )

        best_model = min(
            results,
            key=lambda name: results[name]["cv_rmse"],
        )

        if best_model != rul_cfg["primary"]:
            log(
                f"WARNING [{fd}]: Best CV RMSE model is "
                f"'{best_model}', but primary is "
                f"'{rul_cfg['primary']}'."
            )

        study_path = art(
            cfg, "models", fd, f"{args.study_name}.json"
        )

        # Merge into the existing study file so earlier models (e.g. xgb) are kept.
        merged = {}
        if study_path.exists():
            try:
                merged = json.loads(study_path.read_text())
            except json.JSONDecodeError:
                backup = study_path.with_suffix(".json.corrupt")
                study_path.replace(backup)
                log(f"WARNING [{fd}]: existing study file unreadable; moved to {backup}")
        merged.update(results)

        study_path.write_text(
            json.dumps(merged, indent=2)
        )

        log(f"[{fd}] Study results saved: {study_path}")

    log(
        f"\n[{fd}] Training finished in "
        f"{time.time() - start_time:.1f} seconds."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Remaining Useful Life models."
    )

    parser.add_argument(
        "--fd",
        nargs="*",
        help="Dataset subsets, e.g. --fd FD001",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model override, e.g. --models extratrees",
    )

    parser.add_argument(
        "--study-name",
        default="rul_study",
        help="Output file name (without .json) for results, e.g. rul_study_cnnlstm",
    )

    parser.add_argument("--stats", nargs="*",
                        help="Ablation: keep only these feature families, e.g. --stats mean slope. "
                             "Skips bundle and registry so the live model cannot change.")
    args = parser.parse_args()
    if args.stats and args.study_name == "rul_study":
        parser.error("--stats requires a distinct --study-name, e.g. rul_study_abl_mean")

    cfg = load_config()
    mt.setup(cfg)

    subsets = args.fd or cfg["data"]["subsets"]

    for fd in subsets:
        try:
            train_subset(fd, cfg, args)
        except Exception:
            log(f"\nERROR: Training failed for {fd}.")
            traceback.print_exc()
            raise

    log("\nAll requested RUL training jobs completed successfully.")


if __name__ == "__main__":
    main()