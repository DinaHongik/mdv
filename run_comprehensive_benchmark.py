#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import os


def run_experiment(cmd):
    """Run experiment and parse JSON result"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        output = result.stdout

        try:
            parsed = json.loads(output.strip())
            return parsed
        except:
            pass


        lines = output.strip().split('\n')
        json_buffer = []
        in_json = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()


            if stripped.startswith('{'):
                in_json = True
                json_buffer = [line]
                brace_count = line.count('{') - line.count('}')

                if brace_count == 0:
                    try:
                        parsed = json.loads(line)
                        return parsed
                    except:
                        pass
            elif in_json:
                json_buffer.append(line)
                brace_count += line.count('{') - line.count('}')

                if brace_count == 0:
                    try:
                        json_str = '\n'.join(json_buffer)
                        parsed = json.loads(json_str)
                        return parsed
                    except:
                        in_json = False
                        json_buffer = []

        print(f"Warning: No valid JSON found")
        return {}

    except Exception as e:
        print(f"Error running command: {e}")
        return {}


def run_table1(args):
    results = []

    for encoder, ckpt in [("m", args.ckptM_nmo), ("md", args.ckptMD_nmo), ("mdv", args.ckptMDV_nmo)]:
        if not ckpt or not os.path.exists(ckpt):
            print(f"Warning: Skipping {encoder.upper()}-MSG - checkpoint not found: {ckpt}")
            results.append((f"{encoder.upper()}-MSG", {}))
            continue

        cmd = [
            sys.executable, "run_eval.py",
            "--fieldsA", args.fieldsA,
            "--fieldsB", args.fieldsB,
            "--pairs", args.pairs,
            "--device", args.device,
            "--input_mode", "msg",
            "--encoder", encoder,
            "--ckpt", ckpt,
            "--mode", "mdv"
        ]

        print(f"Running {encoder.upper()}-MSG (NMO ckpt + MSG eval)...")
        result = run_experiment(cmd)

        if result and "MDV" in result:
            results.append((f"{encoder.upper()}-MSG", result["MDV"]))
        else:
            print(f"Warning: No MDV key found")
            results.append((f"{encoder.upper()}-MSG", {}))

    for encoder, ckpt in [("m", args.ckptM_nmo), ("md", args.ckptMD_nmo), ("mdv", args.ckptMDV_nmo)]:
        if not ckpt or not os.path.exists(ckpt):
            print(f"Warning: Skipping {encoder.upper()}-NMO - checkpoint not found: {ckpt}")
            results.append((f"{encoder.upper()}-NMO", {}))
            continue

        cmd = [
            sys.executable, "run_eval.py",
            "--fieldsA", args.fieldsA,
            "--fieldsB", args.fieldsB,
            "--pairs", args.pairs,
            "--device", args.device,
            "--input_mode", "nmo",
            "--encoder", encoder,
            "--ckpt", ckpt,
            "--mode", "mdv"
        ]

        print(f"Running {encoder.upper()}-NMO (NMO ckpt + NMO eval)...")
        result = run_experiment(cmd)

        if result and "MDV" in result:
            results.append((f"{encoder.upper()}-NMO", result["MDV"]))
        else:
            print(f"Warning: No MDV key found")
            results.append((f"{encoder.upper()}-NMO", {}))

    return results


def run_table2(args):
    """Table 2 NMO-MDV + Baselines """
    results = []

    if not args.ckptMDV_nmo or not os.path.exists(args.ckptMDV_nmo):
        print("Warning: NMO-MDV checkpoint not found")
        return results

    cmd = [
        sys.executable, "run_eval.py",
        "--fieldsA", args.fieldsA,
        "--fieldsB", args.fieldsB,
        "--pairs", args.pairs,
        "--device", args.device,
        "--input_mode", "nmo",  
        "--encoder", "mdv",
        "--ckpt", args.ckptMDV_nmo,
        "--mode", "all"
    ]

    print("Running Table 2 (NMO-MDV + Baselines)...")
    result = run_experiment(cmd)

    baseline_order = [
        "MDV",
        "TFIDF",
        "SBERT",
        "LOGSY",
        "ROBERTA_DIFFCSE",
        "E5",
        "BM25"
    ]

    if result:
        for key in baseline_order:
            if key in result:
                if key == "MDV":
                    results.append(("MDV-NMO (ours)", result[key]))
                else:
                    results.append((key, result[key]))
    else:
        print("Warning: No results from Table 2")

    return results


def run_table3(args):
    """Table 3: NMO-MDV Ablation Study (NMO)"""
    results = []

    if not args.ckptMDV_nmo or not os.path.exists(args.ckptMDV_nmo):
        print("Warning: NMO-MDV checkpoint not found")
        return results

    cmd = [
        sys.executable, "run_eval.py",
        "--fieldsA", args.fieldsA,
        "--fieldsB", args.fieldsB,
        "--pairs", args.pairs,
        "--device", args.device,
        "--input_mode", "nmo",
        "--encoder", "mdv",
        "--ckpt", args.ckptMDV_nmo,
        "--mode", "table3"
    ]

    print("Running Table 3 (Ablation Study with NMO format)...")
    result = run_experiment(cmd)

    ablation_order = ["Name-only", "Type-only", "Path-only", "S (weighted)"]

    if result:
        for key in ablation_order:
            if key in result:
                results.append((key, result[key]))
    else:
        print("Warning: No results from Table 3")

    return results


def run_table4(args):
    """
    Table 4: ECE Calibration with Isotonic Regression (Paper-aligned)
    - Compare Integrated Score S before vs after isotonic calibration.
    - Both runs use: --mode table4
    """
    results = []

    if not args.ckptMDV_nmo or not os.path.exists(args.ckptMDV_nmo):
        print("Warning: NMO-MDV checkpoint not found")
        return results

    # (1) Integrated Score S (Before Cal.)  [--mode table4, no --calibrate]
    cmd_pre = [
        sys.executable, "run_eval.py",
        "--fieldsA", args.fieldsA,
        "--fieldsB", args.fieldsB,
        "--pairs", args.pairs,
        "--device", args.device,
        "--input_mode", "nmo",
        "--encoder", "mdv",
        "--ckpt", args.ckptMDV_nmo,
        "--mode", "table4"
    ]
    print("Running Table 4: NMO-MDV + S (Before Cal.) ...")
    result_pre = run_experiment(cmd_pre)
    if result_pre and "MDV" in result_pre:
        results.append(("NMO-MDV + S (Before Cal.)", result_pre["MDV"]))
    else:
        print("Warning: No MDV key found for Table 4 (Before Cal.)")

    # (2) Integrated Score S + Isotonic (After Cal.)  [--mode table4 --calibrate]
    cmd_post = [
        sys.executable, "run_eval.py",
        "--fieldsA", args.fieldsA,
        "--fieldsB", args.fieldsB,
        "--pairs", args.pairs,
        "--device", args.device,
        "--input_mode", "nmo",
        "--encoder", "mdv",
        "--ckpt", args.ckptMDV_nmo,
        "--mode", "table4",
        "--calibrate"
    ]
    print("Running Table 4: NMO-MDV + S + Isotonic (After Cal.) ...")
    result_post = run_experiment(cmd_post)
    if result_post and "MDV" in result_post:
        results.append(("NMO-MDV + S + Isotonic", result_post["MDV"]))
    else:
        print("Warning: No MDV key found for Table 4 (After Cal.)")

    return results


def format_calibration_table(title, results):
    """Format calibration comparison table with ECE before/after"""
    print(f"\n{'='*110}")
    print(f"{title}")
    print(f"{'='*110}")

    headers = ["Method", "Hit@1", "Hit@3", "Hit@5", "ECE (Before)", "ECE (After)", "Improvement"]
    widths = [26, 10, 10, 10, 14, 14, 14]

    header_parts = [h.center(w) for h, w in zip(headers, widths)]
    print("|" + "|".join(header_parts) + "|")

    sep_parts = ["-" * w for w in widths]
    print("|" + "|".join(sep_parts) + "|")

    for name, res in results:
        if res:
            ece_pre = res.get('ECE_pre', 0)
            ece_post = res.get('ECE_post', 0)
            improvement = ((ece_pre - ece_post) / ece_pre * 100) if ece_pre > 0 else 0

            row = [
                name.ljust(widths[0]),
                f"{res.get('Hit@1', 0)*100:.1f}%".center(widths[1]),
                f"{res.get('Hit@3', 0)*100:.1f}%".center(widths[2]),
                f"{res.get('Hit@5', 0)*100:.1f}%".center(widths[3]),
                f"{ece_pre:.4f}".center(widths[4]),
                f"{ece_post:.4f}".center(widths[5]),
                f"{improvement:.1f}%".center(widths[6])
            ]
        else:
            row = [name.ljust(widths[0])] + ["N/A".center(w) for w in widths[1:]]
        print("|" + "|".join(row) + "|")

    print(f"{'='*110}\n")

    print("Note:")
    print("  - ECE (Before): Expected Calibration Error before calibration")
    print("  - ECE (After): Expected Calibration Error after Isotonic Regression")
    print("  - Improvement: Percentage reduction in ECE")
    print("  - Hit@k metrics remain unchanged (ranking preserved)\n")


def format_table(title, results):
    """Format results as table - Hit@1,3,5, NDCG@3,5, MRR"""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")

    setting_width = 18
    metric_width = 11

    headers = ["Setting", "Hit@1", "Hit@3", "Hit@5", "NDCG@3", "NDCG@5", "MRR"]

    header_parts = [headers[0].center(setting_width)]
    for h in headers[1:]:
        header_parts.append(h.center(metric_width))
    header_row = "|".join(header_parts)
    print(f"|{header_row}|")

    sep_parts = ["-" * setting_width]
    for _ in range(len(headers) - 1):
        sep_parts.append("-" * metric_width)
    sep_row = "|".join(sep_parts)
    print(f"|{sep_row}|")

    for name, res in results:
        if res:
            row_parts = [name.ljust(setting_width)]
            row_parts.append(f"{res.get('Hit@1', 0)*100:.1f}%".center(metric_width))
            row_parts.append(f"{res.get('Hit@3', 0)*100:.1f}%".center(metric_width))
            row_parts.append(f"{res.get('Hit@5', 0)*100:.1f}%".center(metric_width))
            row_parts.append(f"{res.get('NDCG@3', 0)*100:.1f}%".center(metric_width))
            row_parts.append(f"{res.get('NDCG@5', 0)*100:.1f}%".center(metric_width))
            row_parts.append(f"{res.get('MRR', 0)*100:.1f}%".center(metric_width))
        else:
            row_parts = [name.ljust(setting_width)]
            for _ in range(len(headers) - 1):
                row_parts.append("N/A".center(metric_width))

        row_str = "|".join(row_parts)
        print(f"|{row_str}|")

    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fieldsA", required=True)
    parser.add_argument("--fieldsB", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckptM_nmo", default="", help="NMO_M checkpoint")
    parser.add_argument("--ckptMD_nmo", default="", help="NMO_MD checkpoint")
    parser.add_argument("--ckptMDV_nmo", default="", help="NMO_MDV checkpoint")
    parser.add_argument("--table", choices=["1", "2", "3", "4", "all"], default="all")
    args = parser.parse_args()

    all_results = {}

    if args.table in ["1", "all"]:
        print("\n" + "="*80)
        print("Starting Table 1: MSG vs NMO Format Evaluation")
        print("="*80)
        t1_results = run_table1(args)
        all_results["Table 1"] = t1_results
        format_table("Table 1: NMO Checkpoint Evaluation (MSG vs NMO Format)", t1_results)

    if args.table in ["2", "all"]:
        print("\n" + "="*80)
        print("Starting Table 2: Baseline Comparison")
        print("="*80)
        t2_results = run_table2(args)
        all_results["Table 2"] = t2_results
        format_table("Table 2: Baseline Comparison Results", t2_results)

    if args.table in ["3", "all"]:
        print("\n" + "="*80)
        print("Starting Table 3: Ablation Study")
        print("="*80)
        t3_results = run_table3(args)
        all_results["Table 3"] = t3_results
        format_table("Table 3: Ablation Study Results", t3_results)

    if args.table in ["4", "all"]:
        print("\n" + "="*80)
        print("Starting Table 4: Isotonic Calibration (Paper-aligned: S before vs after)")
        print("="*80)
        t4_results = run_table4(args)
        all_results["Table 4"] = t4_results
        format_calibration_table("Table 4: ECE Calibration with Isotonic Regression (NMO-MDV + S)", t4_results)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
