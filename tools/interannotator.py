import pandas as pd
import sys

def extract_annotation_vector(annotation):
    annotation = annotation.replace("  ", " ").replace(" +", "+").replace("+ ", "+").split("+")
    vector = []
    for item in annotation:
        vector += [1] + [0 for _ in item[1:]]
    return vector[1:]

def compute_all_vector(annotations_a, annotations_b):
    vector_a = []
    vector_b = []
    for a, b in zip(annotations_a, annotations_b):
        va = extract_annotation_vector(a)
        vb = extract_annotation_vector(b)
        lim = min(len(va), len(vb))
        va, vb = va[:lim], vb[:lim]
        vector_a += va
        vector_b += vb
    return vector_a, vector_b

def cohens_kappa(annotations_a, annotations_b):
    vect_a, vect_b = compute_all_vector(annotations_a, annotations_b)
    if len(vect_a) == 0:
        return 0
    #print(len(vect_a), len(vect_b))
    zeros_a, ones_a = vect_a.count(0), vect_a.count(1)
    zeros_b, ones_b = vect_b.count(0), vect_b.count(1)
    agreements = len([(a,b) for a, b in zip(vect_a, vect_b) if a == b])/len(vect_a)
    expected_random_agreements = (zeros_a * zeros_b + ones_a  * ones_b)/len(vect_a)**2
    #print(agreements, expected_random_agreements)
    kappa = (agreements - expected_random_agreements) / (1 - expected_random_agreements)
    return kappa

def get_kappa(path_1, path_2):
    df_1 = pd.read_csv(path_1, sep="\t", dtype=str).fillna("")
    df_2 = pd.read_csv(path_2, sep="\t", dtype=str).fillna("")
    df_1_filtered = df_1[df_1['#ID'].isin(df_2['#ID'])]
    df_2_filtered = df_2[df_2['#ID'].isin(df_1['#ID'])]
    annotation_1 = list(df_1_filtered["ANNOTATION"])
    annotation_2 = list(df_2_filtered["ANNOTATION"])
    return cohens_kappa(annotation_1, annotation_2)

kappa = get_kappa(sys.argv[1], sys.argv[2])
print(sys.argv[2].replace("results/", "").replace("tsv", "").replace(".", "\t") + str(kappa))
