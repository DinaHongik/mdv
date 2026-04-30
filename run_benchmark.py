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
        output = result.stdout
        
        try:
            return json.loads(output.strip())
        except:
            lines = output.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except:
                        continue
            return {}
    except:
        return {}

def run_table1(args):
    """Table 1: MSG vs NMO comparison"""
    results = []
    configs = [
        ("MSG-M", "msg", "m", args.ckptM_msg),
        ("MSG-MD", "msg", "md", args.ckptMD_msg),
        ("MSG-MDV", "msg", "mdv", args.ckptMDV_msg),
        ("NMO-M", "nmo", "m", args.ckptM_nmo),
        ("NMO-MD", "nmo", "md", args.ckptMD_nmo),
        ("NMO-MDV", "nmo", "mdv", args.ckptMDV_nmo),
    ]
    
    for name, input_mode, encoder, ckpt in configs:
        if not ckpt or not os.path.exists(ckpt):
            results.append((name, {}))
            continue
        
        cmd = [
            sys.executable, "run_eval.py",
            "--fieldsA", args.fieldsA,
            "--fieldsB", args.fieldsB,
            "--pairs", args.pairs,
            "--device", args.device,
            "--input_mode", input_mode,
            "--encoder", encoder,
            "--ckpt", ckpt,
            "--mode", "mdv"
        ]
        
        print(f"Running {name}...")
        result = run_experiment(cmd)
        if result and "MDV" in result:
            results.append((name, result["MDV"]))
        else:
            results.append((name, {}))
    
    return results

def run_table2(args):
    """Table 2: Baseline comparison"""
    results = []
    
    # NMO-MDV (ours)
    if args.ckptMDV_nmo and os.path.exists(args.ckptMDV_nmo):
        cmd = [
            sys.executable, "run_eval.py",
            "--fieldsA", args.fieldsA,
            "--fieldsB", args.fieldsB,
            "--pairs", args.pairs,
            "--device", args.device,
            "--input_mode", "nmo",
            "--encoder", "mdv",
            "--ckpt", args.ckptMDV_nmo,
            "--mode", "all"  # MDV + baselines
        ]
        
        print("Running Table 2 (MDV + Baselines)...")
        result = run_experiment(cmd)
        
        if result:
            for name, res in result.items():
                if name == "MDV":
                    results.append(("NMO-MDV (ours)", res))
                else:
                    results.append((name, res))
    
    return results

def format_table(title, results):
    """Format results as table"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    headers = ["Setting", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]
    col_width = 15
    
    header_row = "|".join(h.center(col_width) for h in headers)
    print(f"|{header_row}|")
    print(f"|{'-'*(col_width*len(headers) + len(headers)-1)}|")
    
    for name, res in results:
        if res:
            row = [
                name,
                f"{res.get('Hit@1', 0)*100:.1f}%",
                f"{res.get('Hit@3', 0)*100:.1f}%",
                f"{res.get('Hit@5', 0)*100:.1f}%",
                f"{res.get('Hit@10', 0)*100:.1f}%",
                f"{res.get('MRR', 0)*100:.1f}%"
            ]
        else:
            row = [name] + ["N/A"] * 5
        
        row_str = "|".join(str(cell).center(col_width) for cell in row)
        print(f"|{row_str}|")
    
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fieldsA", required=True)
    parser.add_argument("--fieldsB", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckptM_msg", default="")
    parser.add_argument("--ckptM_nmo", default="")
    parser.add_argument("--ckptMD_msg", default="")
    parser.add_argument("--ckptMD_nmo", default="")
    parser.add_argument("--ckptMDV_msg", default="")
    parser.add_argument("--ckptMDV_nmo", default="")
    parser.add_argument("--table", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    all_results = {}

    if args.table in ["1", "all"]:
        print("\n" + "="*80)
        print("Starting Table 1: NMO Standardization Comparison")
        print("="*80)
        t1_results = run_table1(args)
        all_results["Table 1"] = t1_results
        format_table("Table 1: Evaluation Results", t1_results)

    if args.table in ["2", "all"]:
        print("\n" + "="*80)
        print("Starting Table 2: Baseline Comparison")
        print("="*80)
        t2_results = run_table2(args)
        all_results["Table 2"] = t2_results
        format_table("Table 2: Baseline Comparison Results", t2_results)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()
