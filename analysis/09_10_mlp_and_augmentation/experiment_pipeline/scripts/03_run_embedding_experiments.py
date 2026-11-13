"""Block C: embedding MLP epoch selection, then the real-only embedding fits.

    python scripts/03_run_embedding_experiments.py --device cuda --resume
    python scripts/03_run_embedding_experiments.py --resume --task prognostic --model mlp_embedding
    python scripts/03_run_embedding_experiments.py --resume --unit repeat_02_fold_04

Real patients only (synthetic rows have no embeddings). Every result carries
evaluation_scope=exploratory_full_cohort_embedding, strict_outer_transform_isolation=false.
"""
import sys

from _exp12_common import Log, parser, refuse_cpu_identity, select

from src import contracts as C
from src.execution import Runner
from src.manifests import verify_plan
from src.paths import load_config, out_dir

REPRESENTATION = "embedding"


def main(argv=None):
    a = parser(__doc__).parse_args(argv)
    cfg = load_config(a.config)
    refuse_cpu_identity(cfg, a)
    log = Log(f"run_{REPRESENTATION}", out_dir(cfg, "logs", a.output_root))
    inputs = C.Inputs(cfg)
    frozen = verify_plan(cfg, inputs, a.output_root)
    ev, sel = C.build_matrix(cfg, inputs)
    ev = select(ev[ev.representation == REPRESENTATION], a)
    sel = select(sel[sel.representation == REPRESENTATION], a, size_col="__none__")
    if a.model and not a.model.startswith("mlp_"):
        sel = sel.iloc[0:0]
    needed = set(ev["selection_key"]) - {""}
    sel = sel[sel["selection_key"].isin(needed)] if len(needed) else sel.iloc[0:0]
    if a.dry_run:
        log(f"dry run: {len(sel)} selection fits and {len(ev)} evaluation fits selected")
        return 0
    R = Runner(cfg, inputs, frozen, a.device, a.output_root, log)
    log(f"plan {frozen['plan_sha256'][:16]}; {len(sel)} selections, {len(ev)} evaluation fits")
    for i, r in enumerate(sel.itertuples(index=False), 1):
        rec, how = R.run_selection(r._asdict(), resume=a.resume)
        log(f"[sel {i}/{len(sel)}] {r.task}/{r.unit_id}/{r.representation}/seed{r.model_seed}: "
            f"{rec['status']} E={rec.get('selected_epoch')} ({how})")
    # heavy in-context models are grouped so each runs sequentially on the GPU
    order = {"tabpfn_classifier": 3, "tabicl_v2": 4, "tabdpt": 5,
             "tabpfn_landmark_survival": 3, "survivalpfn_native": 4}
    ev = ev.assign(_o=ev["model_id"].map(order).fillna(1)).sort_values(
        ["_o", "task", "model_id", "model_seed", "augmentation_size", "unit_id"])
    bad = 0
    for i, r in enumerate(ev.drop(columns="_o").itertuples(index=False), 1):
        rec, how = R.run_fit(r._asdict(), resume=a.resume)
        if rec["status"] != "completed":
            bad += 1
        if how != "skipped_verified" or i % 50 == 0:
            log(f"[fit {i}/{len(ev)}] {r.task}/{r.model_id}/+{r.augmentation_size}/"
                f"seed{r.model_seed}/{r.unit_id}: {rec['status']} {rec.get('wall_seconds')}s "
                f"({how}) {(rec.get('reason') or '')[:160]}")
    log(f"done: {len(ev) - bad} completed, {bad} not completed")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
