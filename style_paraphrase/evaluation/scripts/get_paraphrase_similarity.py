import argparse, os
import numpy as np
import tqdm
import torch
import json

from style_paraphrase.evaluation.similarity.test_sim import find_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--generated_path', type=str, default=None)
parser.add_argument('--reference_strs', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--output_file_name', type=str, default=None)
parser.add_argument("--decode", dest="decode", action="store_true")
parser.add_argument("--store_scores", dest="store_scores", action="store_true")
args = parser.parse_args()

args.output_path = args.output_path + "/" + args.generated_path.split("/")[-1].replace(".json", "/")
print(args.output_path)

with open(args.generated_path, "r") as f:
    data = f.read().strip().split("\n")
data = [json.loads(x)["paraphrase"] for x in data] 
    
output = [[] for _ in data]
all_sim_scores = []

for rs in args.reference_strs.split(","):
    with open(args.generated_path, "r") as f:
        ref_data = f.read().strip().split("\n")
        
    ref_data = [json.loads(x)[rs] for x in ref_data] 
    assert len(data) == len(ref_data)

    sim_scores = []
    for i in tqdm.tqdm(range(0, len(data), args.batch_size)):
        sim_scores.extend(
            find_similarity(data[i:i + args.batch_size], ref_data[i:i + args.batch_size])
        )
    all_sim_scores.append(sim_scores)

    print("Avg similarity score vs {} = {:4f}".format(rs, np.mean(sim_scores)))
    for i in range(len(output)):
        output[i].append("{:.4f} vs {}".format(sim_scores[i], rs))

max_scores = [max([ss[i] for ss in all_sim_scores]) for i in range(len(all_sim_scores[0]))]
print("Avg max score = {:4f}".format(np.mean(max_scores)))

output = [", ".join(x) for x in output]

if not args.output_path:
    args.output_path = "/".join(args.generated_path.split("/")[:-1]) + "/pp_similarity_unique.txt"

with open(os.path.join(args.output_path, args.output_file_name), "w") as f:
    f.write("\n".join(output) + "\n")
    
with open(os.path.join(args.output_path, args.output_file_name+"_sim.summary"), "w") as f:
    f.write("Avg score = {:6f}\n".format(np.mean(max_scores) * 100.00))

if args.store_scores:
    with open(os.path.join(args.output_path, "sim_scores_"+args.output_file_name), "w") as f:
        f.write(
            "\n".join([",".join([str(sim[i]) for sim in all_sim_scores]) for i in range(len(all_sim_scores[0]))]) + "\n"
        )
