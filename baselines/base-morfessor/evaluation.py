import sys
DELIMITER = "+"

# -----------------------
# Helpers
# -----------------------

def split_morphs(s):
    if not isinstance(s, str) or s.strip() == "":
        return []
    return s.split(DELIMITER)

def get_boundaries(morphs):
    boundaries = []
    pos = 0
    for m in morphs[:-1]:
        pos += len(m)
        boundaries.append(pos)
    return set(boundaries)

def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f

# -----------------------
# Evaluation
# -----------------------

def evaluate(gold_series, pred_series):
    missing = 0
    word_correct = 0
    total_words = len(gold_series)

    morph_acc_sum = 0

    morph_tp = morph_fp = morph_fn = 0
    bound_tp = bound_fp = bound_fn = 0

    for gold, pred in zip(gold_series, pred_series):
        gold = gold.lower().strip().replace(" + ", "+")
        # print(gold, pred)
        # --- Word accuracy ---
        # Skip monomorphemic
        #if gold.count("+") == 0:
        #    continue

        #pred = gold.replace("+", "").replace(" ", "")
        if gold == pred.replace(" + ", "+").lower().strip():
            word_correct += 1
        if len(pred.strip()) == 0:
            missing += 1
        #else:
        #    print(gold, pred)

        gold_morphs = split_morphs(gold)
        pred_morphs = split_morphs(pred)

        # --- Morph accuracy ---
        if len(gold_morphs) == 0:
            morph_acc = 1 if len(pred_morphs) == 0 else 0
        else:
            correct = sum(
                1 for g, p in zip(gold_morphs, pred_morphs) if g == p
            )
            morph_acc = correct / max(len(gold_morphs), len(pred_morphs), 1)

        morph_acc_sum += morph_acc

        # --- Morph P/R/F1 ---
        gold_set = set(gold_morphs)
        pred_set = set(pred_morphs)

        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        morph_tp += tp
        morph_fp += fp
        morph_fn += fn

        # --- Boundary P/R/F1 ---
        gold_b = get_boundaries(gold_morphs)
        pred_b = get_boundaries(pred_morphs)

        tp = len(gold_b & pred_b)
        fp = len(pred_b - gold_b)
        fn = len(gold_b - pred_b)

        bound_tp += tp
        bound_fp += fp
        bound_fn += fn

    # -----------------------
    # Aggregate
    # -----------------------

    word_acc = word_correct / total_words if total_words else 0
    morph_acc = morph_acc_sum / total_words if total_words else 0

    morph_p, morph_r, morph_f = precision_recall_f1(
        morph_tp, morph_fp, morph_fn
    )

    bound_p, bound_r, bound_f = precision_recall_f1(
        bound_tp, bound_fp, bound_fn
    )

    return {
        "word_accuracy": word_acc,
        "morph_accuracy": morph_acc,
        "morph_precision": morph_p,
        "morph_recall": morph_r,
        "morph_f1": morph_f,
        "boundary_precision": bound_p,
        "boundary_recall": bound_r,
        "boundary_f1": bound_f,
        "present": total_words - missing
    }

# -----------------------
# MAIN
# -----------------------

def read_txt(path):
    with open(path, "r") as r:
        return [line.strip().replace(" +", "+").replace("+ ", "+").replace(" ", "+").split("\t")[-1] for line in r]

def main():
    if len(sys.argv) != 3:
        print("Usage: python eval.py gold.tsv pred.tsv")
        sys.exit()

    gold_data = read_txt(sys.argv[1])
    pred_data = read_txt(sys.argv[2])
    results = evaluate(gold_data, pred_data)

    #print("\nEvaluation Results:\n")
    print("\t".join([str(x) for x in [sys.argv[2], results["word_accuracy"], results["boundary_precision"], results["boundary_recall"], results["boundary_f1"], str(results["present"])]]))
    #for k, v in :
    #    print(f"{k:25s}: {v:.4f}")

if __name__ == "__main__":
    main()
