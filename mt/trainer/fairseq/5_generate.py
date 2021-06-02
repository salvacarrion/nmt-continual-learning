import os
import subprocess

if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq"
    FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
    FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

VOCAB_SIZE = 32000
DOMAINS = ["health", "biological", "merged"]


def generate():
    EVAL_NAME = "evaluate_test_bleu5__hbm"

    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        for domain in DOMAINS:
            dataset1 = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"
            print(f"Evaluating model from: {dataset1}...")

            # Paths
            MODEL_BASEPATH = os.path.abspath(os.path.join(BASE_PATH, dataset1))

            for domain2 in DOMAINS:
                print(f"\t- Preprocessing test set from: {domain2}...")
                dataset2 = f"scielo_{domain2}_{SRC_LANG}_{TRG_LANG}"
                TEST_DATAPATH = os.path.abspath(os.path.join(BASE_PATH, dataset2))
                OUTPUT_PATH = os.path.abspath(os.path.join(BASE_PATH, EVAL_NAME, dataset1, domain2))

                subprocess.call(['sh', './scripts/3_preprocess-test.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, MODEL_BASEPATH, TEST_DATAPATH, OUTPUT_PATH, FAST_PATH])

                # Generate them
                print(f"\t- Generating translations for: {domain2}...")
                subprocess.call(['sh', './scripts/5_generate.sh', SRC_LANG, TRG_LANG, OUTPUT_PATH, MODEL_BASEPATH])

            print("")
            print("########################################################################")
            print("########################################################################")
            print("")
        print("")
        print("------------------------------------------------------------------------")
        print("------------------------------------------------------------------------")
        print("")


def generate_eval():
    EVAL_NAME = "evaluate_test_bleu5__seq"

    for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
        dataset1 = f"scielo_health_biological_{SRC_LANG}_{TRG_LANG}"
        print(f"Evaluating model from: {dataset1}...")

        # Paths
        MODEL_BASEPATH = os.path.abspath(os.path.join(BASE_PATH, dataset1))

        for domain2 in DOMAINS:
            print(f"\t- Preprocessing test set from: {domain2}...")
            dataset2 = f"scielo_{domain2}_{SRC_LANG}_{TRG_LANG}"
            TEST_DATAPATH = os.path.abspath(os.path.join(BASE_PATH, dataset2))
            OUTPUT_PATH = os.path.abspath(os.path.join(BASE_PATH, EVAL_NAME, dataset1, domain2))

            subprocess.call(['sh', './scripts/3_preprocess-generate.sh', str(VOCAB_SIZE), SRC_LANG, TRG_LANG, MODEL_BASEPATH, TEST_DATAPATH, OUTPUT_PATH, FAST_PATH])

            # Generate them
            print(f"\t- Generating translations for: {domain2}...")
            subprocess.call(['sh', './scripts/5_generate.sh', SRC_LANG, TRG_LANG, OUTPUT_PATH, MODEL_BASEPATH])

            print("")
            print("########################################################################")
            print("########################################################################")
            print("")
        print("")
        print("------------------------------------------------------------------------")
        print("------------------------------------------------------------------------")
        print("")


if __name__ == "__main__":
    generate_eval()
