import argparse
import os
import subprocess
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="best_model_for.pth",
    help="Model filename"
)
args = parser.parse_args()

MODEL = args.model

DATASETS = {
    "Human": "Main data set/training voice model/original",
    "AI-generated": "Main data set/training voice model/ai",
}

results = []

for true_label, folder in DATASETS.items():
    for filename in sorted(os.listdir(folder)):
        if filename.startswith("."):
            continue

        filepath = os.path.join(folder, filename)

        cmd = [
            "python3",
            "inference.py",
            filepath,
            "--model",
            MODEL,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        output = proc.stdout + proc.stderr

        pred = "Unknown"
        conf = ""

        m = re.search(r"Prediction:\s*(.+)", output)
        if m:
            pred = m.group(1).strip()

        c = re.search(r"Confidence:\s*([0-9.]+)%", output)
        if c:
            conf = c.group(1)

        results.append((filename, true_label, pred, conf))

print("\n==============================")
print("PERSONAL DATASET EVALUATION")
print("==============================\n")

correct = 0

for name, truth, pred, conf in results:
    ok = truth == pred
    if ok:
        correct += 1

    print(
        f"{'✓' if ok else '✗'} "
        f"{name:40}"
        f" True={truth:12}"
        f" Pred={pred:12}"
        f" Conf={conf}%"
    )

accuracy = correct / len(results)

print("\n-----------------------------")
print(f"Total samples : {len(results)}")
print(f"Correct       : {correct}")
print(f"Wrong         : {len(results)-correct}")
print(f"Accuracy      : {accuracy*100:.2f}%")
print("-----------------------------")
