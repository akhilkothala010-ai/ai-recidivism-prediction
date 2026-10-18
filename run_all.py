"""
Master Execution Script for CSET485 Assignment #1: COMPAS Analysis
Usage:
    python run_all.py
    python run_all.py --seed 100
"""

import os
import sys
import argparse
import subprocess
from pipeline import run_pipeline
from build_full_artifacts import generate_pdf_report, generate_jupyter_notebook

def main():
    parser = argparse.ArgumentParser(description="Run complete COMPAS fairness and stress-testing pipeline.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed (last 3 digits of your roll number, default: 100)")
    parser.add_argument("--skip-nb-exec", action="store_true", help="Skip pre-executing the notebook with nbconvert")
    args = parser.parse_args()

    seed = args.seed
    print("=" * 75)
    print(f" CSET485: AI and Society — Assignment #1 Execution")
    print(f" Dataset: COMPAS Recidivism (Assigned Digit: 0)")
    print(f" Random Seed (Train-Test Split): {seed}")
    print("=" * 75)

    # 1. Run Core Pipeline & Save Figures
    print("\n[Step 1/4] Running modeling pipeline, stress-tests, and plotting figures...")
    res = run_pipeline(seed=seed, figures_dir='figures')
    print("  -> Figures saved to ./figures/")
    print(f"  -> Model A vs B Mismatch Rate: {res['part1']['pct_mismatch']:.2f}% ({res['part1']['mismatches']} cases)")
    print(f"  -> Omitted Variable Bias (is_African_American shift): {res['part2_2']['table'][2]['pct_change']:+.2f}%")
    print(f"  -> Adversarial High-Confidence Failures: {res['part2_3']['adv_total_count']} ({res['part2_3']['adv_pct']:.2f}%)")

    # 2. Generate PDF Report
    print("\n[Step 2/4] Compiling publication-quality PDF report via ReportLab...")
    pdf_filename = "CSET485_Assignment1_Report.pdf"
    generate_pdf_report(res, filename=pdf_filename)
    print(f"  -> PDF Report generated: {pdf_filename}")

    # 3. Generate Jupyter Notebook
    print("\n[Step 3/4] Generating fully documented Jupyter Notebook...")
    nb_filename = "compas_recidivism_analysis.ipynb"
    generate_jupyter_notebook(res, filename=nb_filename)
    print(f"  -> Notebook file written: {nb_filename}")

    # 4. Pre-execute Notebook via nbconvert (so outputs and charts are populated)
    if not args.skip_nb_exec:
        print("\n[Step 4/4] Pre-executing notebook to embed all plots and table outputs...")
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_filename]
        try:
            subprocess.run(cmd, check=True)
            print("  -> Notebook successfully pre-executed and populated with outputs!")
        except Exception as e:
            print(f"  -> Warning: Could not pre-execute notebook via nbconvert ({e}). You can open and Run All in VS Code / Jupyter.")
    else:
        print("\n[Step 4/4] Skipped notebook execution as requested.")

    print("\n" + "=" * 75)
    print(" SUCCESS! All assignment deliverables are ready:")
    print(f" 1. PDF Report:        {os.path.abspath(pdf_filename)}")
    print(f" 2. Jupyter Notebook:  {os.path.abspath(nb_filename)}")
    print(f" 3. Generated Figures: {os.path.abspath('figures')}")
    print("=" * 75)

if __name__ == '__main__':
    main()
