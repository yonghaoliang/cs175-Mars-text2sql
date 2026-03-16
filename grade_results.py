import os
import warnings

warnings.filterwarnings("ignore")

from config import (
    DRIVE_DB_PATH, OUTPUT_DIR,
    BASELINE_RAW, FEWSHOT_RAW, COT_RAW,
    BASELINE_SCORED, FEWSHOT_SCORED, COT_SCORED,
    REFINEMENT_RAW, REFINEMENT_COMPARED,
)
from models import load_judge_model
from grading import grade_csv_with_execution, grade_multi_attempts, calculate_accuracy


def grade_main_methods(judge_tokenizer, judge_model):
    """Grade baseline, few-shot, and CoT predictions."""
    files_to_grade = [
        (BASELINE_RAW, BASELINE_SCORED),
        (FEWSHOT_RAW,  FEWSHOT_SCORED),
        (COT_RAW,      COT_SCORED),
    ]

    results_summary = {}
    print("🚀 Starting execution-aware grading. This tests against the actual databases...")

    for input_file, output_file in files_to_grade:
        try:
            total_pts, max_pts, acc = grade_csv_with_execution(
                input_file, output_file, DRIVE_DB_PATH, judge_tokenizer, judge_model
            )
            method_name = input_file.split("/")[-1].split("_")[0].upper()
            results_summary[method_name] = {"Points": f"{total_pts} / {max_pts}", "Accuracy": acc}
            print(f"✅ Finished grading {method_name}")
        except FileNotFoundError:
            print(f"⚠️  Could not find {input_file}. Skipping.")

    print("\n========================================")
    print("🏆 FINAL EXECUTION-AWARE SUMMARY (0-10)")
    print("========================================")
    for method, data in results_summary.items():
        print(f"- {method}: {data['Accuracy']:.2f}%  (Points: {data['Points']})")
    print("========================================")


def print_scoreboard():
    """Read already-graded CSVs and print a quick scoreboard."""
    files_to_check = {
        "Baseline":        BASELINE_SCORED,
        "Few-Shot":        FEWSHOT_SCORED,
        "Chain of Thought": COT_SCORED,
    }

    print("🏆 CURRENT SCOREBOARD 🏆")
    print("========================================")
    for method_name, file_path in files_to_check.items():
        if os.path.exists(file_path):
            result = calculate_accuracy(file_path)
            print(f"- {method_name}: {result}")
        else:
            print(f"- {method_name}: ⚠️  File not found. Make sure it finished running!")
    print("========================================")


def grade_refinement_attempts(judge_tokenizer, judge_model):
    """Grade the three refinement attempts and compare improvement per step."""
    try:
        avg1, avg2, avg3 = grade_multi_attempts(
            REFINEMENT_RAW, REFINEMENT_COMPARED, DRIVE_DB_PATH, judge_tokenizer, judge_model
        )
        print("\n========================================")
        print("📈 ATTEMPT COMPARISON SUMMARY (Avg Score 0-10)")
        print("========================================")
        print(f"🔹 Attempt 1 (Initial):  {avg1:.2f}")
        print(f"🔹 Attempt 2 (Refined):  {avg2:.2f}")
        print(f"🔹 Attempt 3 (Final):    {avg3:.2f}")
        print(f"🚀 Total Improvement:    {avg3 - avg1:+.2f} points")
        print("========================================")
    except FileNotFoundError:
        print("⚠️  Please run the Refinement experiment first to generate the attempts CSV!")


if __name__ == "__main__":
    judge_tokenizer, judge_model = load_judge_model()

    grade_main_methods(judge_tokenizer, judge_model)
    print_scoreboard()
    grade_refinement_attempts(judge_tokenizer, judge_model)