import argparse, json, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--generated_file', default=None, type=str)
parser.add_argument('--reference_file', default=None, type=str)
parser.add_argument('--classifier_file', default=None, type=str)
parser.add_argument('--paraphrase_file', default=None, type=str)
parser.add_argument('--acceptability_file', default=None, type=str)
parser.add_argument('--expected_classifier_value', default="correct", type=str)
parser.add_argument('--output_path', type=str, default=None)
args = parser.parse_args()

with open(args.generated_file, "r") as f:
    generated_dataset = f.read().strip().split("\n")
generated_data = [json.loads(x)["paraphrase"] for x in generated_dataset] 
orginal_tweet_data = [json.loads(x)["org_tweet"] for x in generated_dataset]

with open(args.classifier_file, "r") as f:
    classifier_data = f.read().strip().split("\n")
classifier_data = classifier_data[1:]

with open(args.paraphrase_file, "r") as f:
    paraphrase_data = f.read().strip().split("\n")

with open(args.acceptability_file, "r") as f:
    acceptability_data = f.read().strip().split("\n")

assert len(classifier_data) == len(paraphrase_data)
assert len(paraphrase_data) == len(acceptability_data)

scores = {
    "acc_sim": [],
    "cola_sim": [],
    "acc_cola": [],
    "acc_cola_sim": []
}

valid_count = {
    "acc_sim": 0,
    "cola_sim": 0,
    "acc_cola": 0,
    "acc_cola_sim": 0
}
normalized_generated_data = {
    "acc_sim": ["original_tweet\tgeneration\ttarget_cls\tpredict_label"],
    "cola_sim": ["original_tweet\tgeneration\ttarget_cls\tpredict_label"],
    "acc_cola_sim": ["original_tweet\tgeneration\ttarget_cls\tpredict_label"],
    "all_samples": ["original_tweet\tgeneration\tagreement\ttarget_cls\tpredict_label\tacceptability\tsimilarity"]
}

for cd, pd, gd, ad, otd in zip(classifier_data, paraphrase_data, generated_data, acceptability_data, orginal_tweet_data):
    curr_scores = max([float(x) for x in pd.split(",")])
    agreement = "correct" if cd.split("\t")[0] == cd.split("\t")[1] else "incorrect" 
    
    normalized_generated_data["all_samples"].append(otd +"\t"+ gd +"\t"+ agreement +"\t"+ cd.split("\t")[0] +"\t"+ cd.split("\t")[1] +"\t"+ ad + "\t"+ str(curr_scores))
    
    # check acc_sim
    if cd.split("\t")[0] == cd.split("\t")[1]:
        valid_count["acc_sim"] += 1
        scores["acc_sim"].append(curr_scores)
        normalized_generated_data["acc_sim"].append(otd +"\t"+ gd +"\t"+ cd.split("\t")[0] +"\t"+ cd.split("\t")[1])
    else:
        scores["acc_sim"].append(0)
        normalized_generated_data["acc_sim"].append("xxxxxxx\txxxxxxxxx\txxxxxxxx\txxxxxxxx")

    # check cola_sim
    if ad == "acceptable":
        valid_count["cola_sim"] += 1
        scores["cola_sim"].append(curr_scores)
        normalized_generated_data["cola_sim"].append(otd +"\t"+ gd +"\t"+ cd.split("\t")[0] +"\t"+ cd.split("\t")[1])
    else:
        scores["cola_sim"].append(0)
        normalized_generated_data["cola_sim"].append("xxxxxxx\txxxxxxxxx\txxxxxxxx\txxxxxxxx")

    # check acc_cola
    if ad == "acceptable" and cd.split("\t")[0] == cd.split("\t")[1]:
        valid_count["acc_cola"] += 1
        scores["acc_cola"].append(1)
        valid_count["acc_cola_sim"] += 1
        scores["acc_cola_sim"].append(curr_scores)
        normalized_generated_data["acc_cola_sim"].append(otd +"\t"+ gd +"\t"+ cd.split("\t")[0] +"\t"+ cd.split("\t")[1])
    else:
        scores["acc_cola"].append(0)
        scores["acc_cola_sim"].append(0)
        normalized_generated_data["acc_cola_sim"].append("xxxxxxx\txxxxxxxxx\txxxxxxxx\txxxxxxxx")

summary = ""

for metric in ["acc_sim", "cola_sim", "acc_cola", "acc_cola_sim"]:
    tmp = "Normalized pp score ({}) = {:.4f} ({:d} / {:d} valid)".format(
            metric, np.mean(scores[metric]), valid_count[metric], len(scores[metric]))
    print(tmp)
    summary = summary + tmp+"\n"

    if metric in normalized_generated_data:
        with open(args.output_path + "results.{}_normalized".format(metric), "w") as f:
            f.write("\n".join(normalized_generated_data[metric]) + "\n")

with open(args.output_path + "all_results_summary".format(metric), "w") as f:
    f.write(summary)
    
with open(args.output_path + "results.all_samples_normalized", "w") as f:
    f.write("\n".join(normalized_generated_data["all_samples"]) + "\n")