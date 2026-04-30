from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


def extract_json_from_stdout(output: str) -> Dict[str, Any]:
    output = (output or "").strip()
    if not output:
        return {}
    try:
        return json.loads(output)
    except Exception:
        pass

    lines = output.splitlines()
    buf: List[str] = []
    in_json = False
    brace_count = 0
    for line in lines:
        s = line.strip()
        if not in_json and s.startswith("{"):
            in_json = True
            buf = [line]
            brace_count = line.count("{") - line.count("}")
            if brace_count == 0:
                try:
                    return json.loads(line)
                except Exception:
                    in_json = False
                    buf = []
            continue
        if in_json:
            buf.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0:
                try:
                    return json.loads("\n".join(buf))
                except Exception:
                    in_json = False
                    buf = []
    return {}


def run_experiment(cmd: List[str], timeout: int = 7200) -> Dict[str, Any]:
    print("[RUN] " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"[ERROR] Failed to run command: {e}", file=sys.stderr)
        return {}

    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[WARN] Non-zero exit code: {result.returncode}", file=sys.stderr)

    parsed = extract_json_from_stdout(result.stdout)
    if not parsed:
        print("[WARN] No valid JSON found in stdout.", file=sys.stderr)
    return parsed


def maybe_append(cmd: List[str], flag: str, value: Any):
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([flag, str(value)])


def build_run_eval_base_cmd(args) -> List[str]:
    run_eval_path = args.run_eval_path
    if not os.path.isabs(run_eval_path):
        run_eval_path = os.path.join(os.getcwd(), run_eval_path)
    return [
        sys.executable,
        run_eval_path,
        "--fieldsA", args.fieldsA,
        "--fieldsB", args.fieldsB,
        "--pairs", args.pairs,
        "--device", args.device,
    ]


def resolve_result_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    if len(raw) == 1:
        only_val = next(iter(raw.values()))
        if isinstance(only_val, dict):
            return only_val
    return raw


def apply_serialization_flags(cmd: List[str], args, input_mode: str):
    cmd.extend(["--input_mode", input_mode])
    if args.mask_name:
        cmd.append("--mask_name")
    if args.drop_type:
        cmd.append("--drop_type")
    if args.drop_path:
        cmd.append("--drop_path")
    if args.drop_desc:
        cmd.append("--drop_desc")
    if args.drop_example:
        cmd.append("--drop_example")
    if args.no_placeholder:
        cmd.append("--no_placeholder")
    maybe_append(cmd, "--max_len", args.max_len)


def run_single_eval(args, encoder: str, ckpt: str, input_mode: str, main_label: Optional[str] = None) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "single"])
    apply_serialization_flags(cmd, args, input_mode)
    if main_label:
        cmd.extend(["--main_label", main_label])
    return resolve_result_payload(run_experiment(cmd, timeout=args.timeout))


def run_table_all(args, encoder: str, ckpt: str, input_mode: str, main_label: str) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "all", "--main_label", main_label])
    apply_serialization_flags(cmd, args, input_mode)
    maybe_append(cmd, "--baseline_module_path", args.baseline_module_path)
    maybe_append(cmd, "--rule_baselines_path", args.rule_baselines_path)
    maybe_append(cmd, "--logsy_ckpt", args.logsy_ckpt)
    maybe_append(cmd, "--logsy_base_model", args.logsy_base_model)
    maybe_append(cmd, "--roberta_diffcse_dir", args.roberta_diffcse_dir)
    return run_experiment(cmd, timeout=args.timeout)


def run_table3(args, encoder: str, ckpt: str, input_mode: str) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "table3"])
    apply_serialization_flags(cmd, args, input_mode)
    maybe_append(cmd, "--s_name_weight", args.s_name_weight)
    maybe_append(cmd, "--s_type_weight", args.s_type_weight)
    maybe_append(cmd, "--s_path_weight", args.s_path_weight)
    return run_experiment(cmd, timeout=args.timeout)


def run_table4(args, encoder: str, ckpt: str, input_mode: str, calibrate: bool) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "table4"])
    apply_serialization_flags(cmd, args, input_mode)
    if calibrate:
        cmd.append("--calibrate")
        maybe_append(cmd, "--ece_bins", args.ece_bins)
        maybe_append(cmd, "--calib_ratio", args.calib_ratio)
        maybe_append(cmd, "--calib_seed", args.calib_seed)
        maybe_append(cmd, "--pairs_calib", args.pairs_calib)
    maybe_append(cmd, "--s_name_weight", args.s_name_weight)
    maybe_append(cmd, "--s_type_weight", args.s_type_weight)
    maybe_append(cmd, "--s_path_weight", args.s_path_weight)
    return resolve_result_payload(run_experiment(cmd, timeout=args.timeout))


def run_stress(args, encoder: str, ckpt: str, input_mode: str) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "stress"])
    apply_serialization_flags(cmd, args, input_mode)
    return run_experiment(cmd, timeout=args.timeout)


def run_latency(args, encoder: str, ckpt: str, input_mode: str) -> Dict[str, Any]:
    cmd = build_run_eval_base_cmd(args)
    cmd.extend(["--encoder", encoder, "--ckpt", ckpt, "--mode", "latency"])
    apply_serialization_flags(cmd, args, input_mode)
    if args.latency_with_components:
        cmd.append("--latency_with_components")
        maybe_append(cmd, "--s_name_weight", args.s_name_weight)
        maybe_append(cmd, "--s_type_weight", args.s_type_weight)
        maybe_append(cmd, "--s_path_weight", args.s_path_weight)
    return run_experiment(cmd, timeout=args.timeout)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "  N/A   "
    try:
        return f"{100.0 * float(value):6.1f}%"
    except Exception:
        return "  N/A   "


def fmt_num(value: Any) -> str:
    if value is None:
        return "   N/A    "
    try:
        return f"{float(value):10.4f}"
    except Exception:
        return "   N/A    "


def print_table_header(title: str, columns: List[str]):
    line = "| " + " | ".join(f"{c:^20}" for c in columns) + " |"
    sep = "|" + "|".join(["-" * 22 for _ in columns]) + "|"
    width = max(len(title), len(line))
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(line)
    print(sep)


def print_table_row(values: List[str]):
    print("| " + " | ".join(f"{v:^20}" for v in values) + " |")


def print_table_footer(title: str, columns: List[str]):
    line = "| " + " | ".join(f"{c:^20}" for c in columns) + " |"
    width = max(len(title), len(line))
    print("=" * width + "\n")


def rank_key(res: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(res.get("Hit@1", -1.0) or -1.0),
        float(res.get("MRR", -1.0) or -1.0),
        float(res.get("NDCG@5", -1.0) or -1.0),
    )


def choose_best_model(table1_nmo: Dict[str, Dict[str, Any]], preferred: str) -> Tuple[str, Dict[str, Any]]:
    if preferred != "best":
        key = preferred.upper()
        if key in table1_nmo and table1_nmo[key]:
            return preferred.lower(), table1_nmo[key]
        raise ValueError(f"Requested encoder '{preferred}' not available among NMO results")

    candidates = []
    for name in ["M", "MD", "MDV"]:
        if name in table1_nmo and table1_nmo[name]:
            candidates.append((name.lower(), table1_nmo[name]))

    if not candidates:
        raise ValueError("No valid NMO single-eval results available to choose best encoder")

    best_name, best_res = max(candidates, key=lambda x: rank_key(x[1]))
    return best_name, best_res


def main(args):
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but no GPU is available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    if args.ntp_only:
        args.drop_desc = True
        args.drop_example = True

    results = {
        "table1": {},
        "table2": {},
        "table3": {},
        "table4": {},
        "table5": {},
        "table6": {},
        "meta": {},
    }

    context_ckpts = {
        "M": args.ckptM_context,
        "MD": args.ckptMD_context,
        "MDV": args.ckptMDV_context,
    }
    nmo_ckpts = {
        "M": args.ckptM_nmo,
        "MD": args.ckptMD_nmo,
        "MDV": args.ckptMDV_nmo,
    }

    # ========================================================
    # Table 1: Context vs NMO
    # ========================================================
    if args.table in {"table1", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 1: Context vs NMO")
        print("=" * 80)

        table1_context: Dict[str, Dict[str, Any]] = {}
        table1_nmo: Dict[str, Dict[str, Any]] = {}

        for name in ["M", "MD", "MDV"]:
            ckpt = context_ckpts.get(name)
            if ckpt and os.path.exists(ckpt):
                print(f"Running Context-{name}...", file=sys.stderr)
                res = run_single_eval(args, name.lower(), ckpt, args.context_input_mode, main_label=f"Context-{name}")
                table1_context[name] = res
            else:
                print(f"[WARN] Skipping Context-{name}: checkpoint not found: {ckpt}", file=sys.stderr)
                table1_context[name] = {}

        for name in ["M", "MD", "MDV"]:
            ckpt = nmo_ckpts.get(name)
            if ckpt and os.path.exists(ckpt):
                print(f"Running NMO-{name}...", file=sys.stderr)
                res = run_single_eval(args, name.lower(), ckpt, args.nmo_input_mode, main_label=f"NMO-{name}")
                table1_nmo[name] = res
            else:
                print(f"[WARN] Skipping NMO-{name}: checkpoint not found: {ckpt}", file=sys.stderr)
                table1_nmo[name] = {}

        results["table1"] = {"context": table1_context, "nmo": table1_nmo}

        title = "Table 1: Input Normalization (Context vs NMO)"
        columns = ["Setting", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5", "MRR"]
        print_table_header(title, columns)
        for label_prefix, block in [("Context", table1_context), ("NMO", table1_nmo)]:
            for name in ["M", "MD", "MDV"]:
                res = block.get(name, {})
                print_table_row([
                    f"{label_prefix}-{name}",
                    fmt_pct(res.get("Hit@1")),
                    fmt_pct(res.get("Hit@3")),
                    fmt_pct(res.get("Hit@5")),
                    fmt_pct(res.get("NDCG@3")),
                    fmt_pct(res.get("NDCG@5")),
                    fmt_pct(res.get("MRR")),
                ])
        print_table_footer(title, columns)
    else:
        table1_nmo = {}

    table1_nmo_results = results.get("table1", {}).get("nmo", {})

    if args.table in {"table1", "all"}:
        chosen_encoder, chosen_res = choose_best_model(table1_nmo_results, args.main_encoder)
    else:
        if args.main_encoder == "best":
            raise ValueError(
                "When running --table table2/table3/table4/table5/table6 only, "
                "set --main_encoder explicitly (m/md/mdv), or run --table all."
            )
        chosen_encoder = args.main_encoder.lower()
        chosen_res = {}
        if chosen_encoder.upper() not in nmo_ckpts or not nmo_ckpts[chosen_encoder.upper()]:
            raise ValueError(f"No checkpoint configured for requested main encoder '{chosen_encoder}'")

    chosen_ckpt = nmo_ckpts[chosen_encoder.upper()]
    results["meta"]["main_encoder_for_tables23456"] = chosen_encoder
    results["meta"]["main_encoder_result"] = chosen_res

    if args.table in {"table1", "all"}:
        s_encoder = args.s_encoder if args.s_encoder != "best" else chosen_encoder
    else:
        s_encoder = chosen_encoder if args.s_encoder == "best" else args.s_encoder.lower()

    if s_encoder.upper() not in nmo_ckpts or not nmo_ckpts[s_encoder.upper()]:
        raise ValueError(f"No checkpoint configured for requested S encoder '{s_encoder}'")

    s_ckpt = nmo_ckpts[s_encoder.upper()]
    results["meta"]["s_encoder_for_tables34"] = s_encoder

    # ========================================================
    # Table 2: Baseline comparison under NMO
    # ========================================================
    if args.table in {"table2", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 2: Baseline Comparison under NMO")
        print("=" * 80)
        print(f"Running Table 2 using main encoder '{chosen_encoder.upper()}' under NMO representation...", file=sys.stderr)

        t2 = run_table_all(args, chosen_encoder, chosen_ckpt, args.nmo_input_mode, f"NMO-{chosen_encoder.upper()}")
        results["table2"] = t2

        title = "Table 2: Baseline Comparison (same NMO representation)"
        columns = ["Setting", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5", "MRR"]
        print_table_header(title, columns)
        order = [
            f"NMO-{chosen_encoder.upper()}",
            "RULE_HEUR",
            "RULE_ENH",
            "TFIDF",
            "SBERT",
            "LOGSY",
            "ROBERTA_DIFFCSE",
            "E5",
            "BM25",
        ]
        label_map = {
            f"NMO-{chosen_encoder.upper()}": f"NMO-{chosen_encoder.upper()}",
            "RULE_HEUR": "Heuristic-rule",
            "RULE_ENH": "Rule+Regex/Dict",
            "TFIDF": "TF-IDF",
            "SBERT": "SBERT",
            "LOGSY": "Logsy",
            "ROBERTA_DIFFCSE": "RoBERTa+DiffCSE",
            "E5": "E5",
            "BM25": "BM25",
        }
        for key in order:
            res = t2.get(key, {})
            print_table_row([
                label_map[key],
                fmt_pct(res.get("Hit@1")),
                fmt_pct(res.get("Hit@3")),
                fmt_pct(res.get("Hit@5")),
                fmt_pct(res.get("NDCG@3")),
                fmt_pct(res.get("NDCG@5")),
                fmt_pct(res.get("MRR")),
            ])
        print_table_footer(title, columns)

    # ========================================================
    # Table 3: Integrated Score S
    # ========================================================
    if args.table in {"table3", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 3: Integrated Score S")
        print("=" * 80)
        print(f"Running Table 3 using encoder '{s_encoder.upper()}'...", file=sys.stderr)

        t3_raw = run_table3(args, s_encoder, s_ckpt, args.nmo_input_mode)
        if len(t3_raw) == 1:
            t3 = next(iter(t3_raw.values()))
        else:
            t3 = t3_raw
        results["table3"] = {"encoder": s_encoder.upper(), "results": t3}

        title = f"Table 3: Component-wise Integrated Score S ({s_encoder.upper()})"
        columns = ["Setting", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5", "MRR"]
        print_table_header(title, columns)
        for key in ["Name-only", "Type-only", "Path-only", "S (weighted)"]:
            res = t3.get(key, {})
            print_table_row([
                key,
                fmt_pct(res.get("Hit@1")),
                fmt_pct(res.get("Hit@3")),
                fmt_pct(res.get("Hit@5")),
                fmt_pct(res.get("NDCG@3")),
                fmt_pct(res.get("NDCG@5")),
                fmt_pct(res.get("MRR")),
            ])
        print_table_footer(title, columns)

    # ========================================================
    # Table 4: Calibration for integrated score S
    # ========================================================
    if args.table in {"table4", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 4: Calibration for integrated score S")
        print("=" * 80)
        print(f"Running Table 4 using encoder '{s_encoder.upper()}' (before calibration)...", file=sys.stderr)
        t4_pre = run_table4(args, s_encoder, s_ckpt, args.nmo_input_mode, calibrate=False)
        print(f"Running Table 4 using encoder '{s_encoder.upper()}' (after calibration)...", file=sys.stderr)
        t4_post = run_table4(args, s_encoder, s_ckpt, args.nmo_input_mode, calibrate=True)
        results["table4"] = {"before": t4_pre, "after": t4_post, "encoder": s_encoder.upper()}

        title = f"Table 4: ECE Calibration with Isotonic Regression ({s_encoder.upper()})"
        columns = ["Method", "Hit@1", "Hit@3", "Hit@5", "ECE (Before)", "ECE (After)", "Improvement"]
        print_table_header(title, columns)

        before_ece = t4_pre.get("ECE_pre", None)
        after_ece = t4_post.get("ECE_post", None)
        improvement = None
        if before_ece is not None and after_ece is not None and float(before_ece) != 0.0:
            improvement = (float(before_ece) - float(after_ece)) / float(before_ece)

        print_table_row([
            f"{s_encoder.upper()} + S (Before Cal.)",
            fmt_pct(t4_pre.get("Hit@1")),
            fmt_pct(t4_pre.get("Hit@3")),
            fmt_pct(t4_pre.get("Hit@5")),
            fmt_num(before_ece),
            fmt_num(t4_pre.get("ECE_post")),
            fmt_pct(0.0),
        ])
        print_table_row([
            f"{s_encoder.upper()} + S + Isotonic",
            fmt_pct(t4_post.get("Hit@1")),
            fmt_pct(t4_post.get("Hit@3")),
            fmt_pct(t4_post.get("Hit@5")),
            fmt_num(t4_post.get("ECE_pre")),
            fmt_num(after_ece),
            fmt_pct(improvement),
        ])
        print_table_footer(title, columns)

    # ========================================================
    # Table 5: Shortcut / leakage robustness stress test
    # ========================================================
    if args.table in {"table5", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 5: Shortcut Robustness Stress Test")
        print("=" * 80)
        t5 = run_stress(args, chosen_encoder, chosen_ckpt, args.nmo_input_mode)
        results["table5"] = {"encoder": chosen_encoder.upper(), "results": t5}

        title = f"Table 5: Shortcut Robustness ({chosen_encoder.upper()})"
        columns = ["Setting", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5", "MRR"]
        print_table_header(title, columns)
        for key in ["Original", "MaskName", "MaskType", "MaskPath", "PathDrift", "TypeDrift"]:
            res = t5.get(key, {})
            print_table_row([
                key,
                fmt_pct(res.get("Hit@1")),
                fmt_pct(res.get("Hit@3")),
                fmt_pct(res.get("Hit@5")),
                fmt_pct(res.get("NDCG@3")),
                fmt_pct(res.get("NDCG@5")),
                fmt_pct(res.get("MRR")),
            ])
        print_table_footer(title, columns)

    # ========================================================
    # Table 6: Latency benchmark
    # ========================================================
    if args.table in {"table6", "all"}:
        print("\n" + "=" * 80)
        print("Starting Table 6: Latency Benchmark")
        print("=" * 80)
        t6 = run_latency(args, chosen_encoder, chosen_ckpt, args.nmo_input_mode)
        results["table6"] = {"encoder": chosen_encoder.upper(), "results": t6}

        title = f"Table 6: Inference Latency ({chosen_encoder.upper()})"
        columns = ["Setting", "Encode A", "Encode B", "Score", "Integrated S", "Total", "ms/source"]
        print_table_header(title, columns)
        print_table_row([
            f"{chosen_encoder.upper()} latency",
            fmt_num(t6.get("encode_A_sec")),
            fmt_num(t6.get("encode_B_sec")),
            fmt_num(t6.get("score_sec")),
            fmt_num(t6.get("integrated_S_sec")),
            fmt_num(t6.get("integrated_total_sec") if t6.get("integrated_total_sec") is not None else t6.get("total_sec")),
            fmt_num(t6.get("integrated_ms_per_source") if t6.get("integrated_ms_per_source") is not None else t6.get("ms_per_source")),
        ])
        print_table_footer(title, columns)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Core inputs
    ap.add_argument("--fieldsA", required=True)
    ap.add_argument("--fieldsB", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pairs_calib", default="")
    ap.add_argument("--run_eval_path", default="run_eval.py")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None} else "cpu")
    ap.add_argument("--timeout", type=int, default=7200)

    # Table control
    ap.add_argument("--table", choices=["table1", "table2", "table3", "table4", "table5", "table6", "all"], default="all")
    ap.add_argument("--output_json", default="benchmark_results.json")

    # Serialization modes
    ap.add_argument("--context_input_mode", choices=["raw_msg", "flat_field", "nmo", "msg"], default="flat_field")
    ap.add_argument("--nmo_input_mode", choices=["nmo", "flat_field", "raw_msg", "msg"], default="nmo")
    ap.add_argument("--mask_name", action="store_true")
    ap.add_argument("--drop_type", action="store_true")
    ap.add_argument("--drop_path", action="store_true")
    ap.add_argument("--drop_desc", action="store_true")
    ap.add_argument("--drop_example", action="store_true")
    ap.add_argument("--no_placeholder", action="store_true")
    ap.add_argument("--ntp_only", action="store_true", help="Shortcut for --drop_desc --drop_example")

    # Checkpoints
    ap.add_argument("--ckptM_context", default="")
    ap.add_argument("--ckptMD_context", default="")
    ap.add_argument("--ckptMDV_context", default="")
    ap.add_argument("--ckptM_nmo", default="")
    ap.add_argument("--ckptMD_nmo", default="")
    ap.add_argument("--ckptMDV_nmo", default="")

    # Which encoder to use for table2 and table3/4
    ap.add_argument("--main_encoder", choices=["best", "m", "md", "mdv"], default="best")
    ap.add_argument("--s_encoder", choices=["best", "m", "md", "mdv"], default="best")

    # Model config overrides for run_eval
    ap.add_argument("--encoder_model", default=None)
    ap.add_argument("--mlm_model", default=None)
    ap.add_argument("--max_len", type=int, default=None)

    # Table 3/4 integrated score weights
    ap.add_argument("--s_name_weight", type=float, default=0.4)
    ap.add_argument("--s_type_weight", type=float, default=0.2)
    ap.add_argument("--s_path_weight", type=float, default=0.4)

    # Table 4 calibration config
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--calib_ratio", type=float, default=0.2)
    ap.add_argument("--calib_seed", type=int, default=42)

    # Optional baseline paths
    ap.add_argument("--baseline_module_path", default=None)
    ap.add_argument("--rule_baselines_path", default=None)
    ap.add_argument("--logsy_ckpt", default=None)
    ap.add_argument("--logsy_base_model", default="bert-base-uncased")
    ap.add_argument("--roberta_diffcse_dir", default=None)

    # Table 6 latency options
    ap.add_argument("--latency_with_components", action="store_true")

    args = ap.parse_args()
    main(args)
