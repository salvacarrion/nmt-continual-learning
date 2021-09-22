import os

def view_tokens():

    datasets = [
        "europarl_fairseq_50k_de-en",
        # "europarl_fairseq_50k_es-en",
        # "europarl_fairseq_50k_cs-en",

        # "europarl_fairseq_cs-en",
        # "europarl_fairseq_100k_cs-en",
        # "europarl_fairseq_es-en",
        # "europarl_fairseq_100k_es-en",
        # "europarl_fairseq_de-en",
        # "europarl_fairseq_100k_de-en",
        # "health_fairseq_vhealth_unconstrained_es-en",
        # "health_fairseq_vhealth_es-en",
        # "commoncrawl_es-en",
        # "commoncrawl_100k_es-en",
        # "newscommentaryv14_es-en",
        # "newscommentaryv14_35k_es-en",
        # "multi30k_de-en",
        # "iwlst2016_de-en",
    ]
    bpes = ["bpe.32000", "bpe.64"]
    base_path = "/home/scarrion/datasets/scielo/constrained/datasets"
    for dataset in datasets:
        print(f"{dataset}: ##############")
        for bpe_i in bpes:

            # Exceptions
            if dataset == "multi30k_de-en" and bpe_i == "bpe.32000":
                bpe_i = "bpe.16000"

            # Get sentences
            with open(os.path.join(base_path, bpe_i, dataset, "tok", bpe_i, "train.en"), "r") as f:
                file = f.readlines()
                tokens = [len(line.strip().split()) for line in file]
                print(f"\t- BPE: {bpe_i} -----------")
                print(f"\t\t- sentences: {len(file)}")
                print(f"\t\t- tokens: {sum(tokens)}")
                print("\t\t- tokens/sentences: {:.3f}".format(sum(tokens)/len(file)))

            # Get scores
            try:
                with open(os.path.join(base_path, bpe_i, dataset, "eval", "checkpoint_best.pt", dataset[:-6], "beam_metrics.json"), "r") as f:
                    print(f"\t\t- Scores: {f.readlines()}")
            except Exception as e:
                print(f"\t\t- Scores error: {e}")

            # Break line
            print("")


if __name__ == "__main__":
    view_tokens()
