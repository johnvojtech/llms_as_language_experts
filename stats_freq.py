#!python3i
import sys
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, levene
import seaborn as sns
import matplotlib.pyplot as plt

# nicer defaults for papers
sns.set(style="whitegrid", context="talk")


def test_assumptions(model, df, main_effects, dependent):
    resid = model.resid
    print(shapiro(resid))
    for effect in main_effects:
        print(effect, levene(*[group[dependent].values for name, group in df.groupby(effect)]))


def run_type3_anova(df, dependent, main_effects, interactions):
    # Build formula with sum-to-zero contrasts
    main_terms = [f"C({col}, Sum)" for col in main_effects]
    interaction_terms = [f"C({a}, Sum):C({b}, Sum)" for (a, b) in interactions]
    formula = dependent + " ~ " + " + ".join(main_terms + interaction_terms)
    # Fit model
    model = smf.ols(formula, data=df).fit()
    # Type III ANOVA
    anova_table = sm.stats.anova_lm(model, typ=3)
    print("\nANOVA table (Type III):")
    print(anova_table)

    return model, anova_table

def run_frequentist_anova(df, dependent, main_effects, interactions, triple_interactions = []):
    # Build formula
    main_terms = [f"C({col})" for col in main_effects]
    interaction_terms = [f"C({a}):C({b})" for (a, b) in interactions]
    triple_interaction_terms = [f"C({a}):C({b}):C({c})" for (a, b, c) in triple_interactions]

    formula = dependent + " ~ " + " + ".join(main_terms + interaction_terms + triple_interaction_terms)

    # Fit model
    model = smf.ols(formula, data=df).fit(cov_type='HC3')
    test_assumptions(model, df, main_effects, dependent)

    # ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    #print("\nANOVA table (Type II):")
    #print(anova_table)

    anova_table = sm.stats.anova_lm(model, typ=3)
    #print("\nANOVA table (Type III):")
    #print(anova_table)

    ss_resid = anova_table.loc["Residual", "sum_sq"]
    anova_table["eta_sq_partial"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_resid)
    anova_table[["sum_sq", "eta_sq_partial"]]
    print(anova_table)
    return model, anova_table

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t")

    df = df[df["PRESENT"] != 0].copy()
    #df = df[df["LLM"] == sys.argv[2]]

    main_effects = ["LANGUAGE", "LLM", "EXAMPLES"] #"UD", "GUIDELINES", "EXAMPLES"]

    interactions = [(a, b) for a in main_effects for b in main_effects if a > b]
    #interactions.append(("GUIDELINES", "UD"))
    #("LANGUAGE", "UD"),
    #("LANGUAGE", "GUIDELINES"),
    #("LANGUAGE", "EXAMPLES")
    #]

    run_frequentist_anova(
        df,
        dependent="BOUNDARY_F1",
        main_effects=main_effects,
        interactions=interactions,
        triple_interactions =[("LANGUAGE", "LLM", "EXAMPLES")] #, ("LANGUAGE", "LLM", "UD"), ("LANGUAGE", "LLM", "GUIDELINES")] #, ("LANGUAGE", "GUIDELINES", "UD"), ("LLM", "GUIDELINES", "UD"), ("EXAMPLES", "GUIDELINES", "UD")]
        )

    plt.figure(figsize=(12, 6))

    sns.pointplot(data=df, x="LANGUAGE", y="BOUNDARY_F1", hue="LLM", dodge=True, errorbar=("ci", 95), markers="o", linestyles="-")
    plt.title("Interaction: LLM × Language")
    plt.ylabel("Performance")
    plt.xlabel("Language")
    plt.xticks(rotation=45)
    plt.legend(title="LLM", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/llm_language.jpg")
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="GUIDELINES", y="BOUNDARY_F1", hue="LLM", dodge=True, errorbar=("ci", 95))
    plt.title("Interaction: LLM × Guidelines")
    plt.ylabel("Performance")
    plt.xlabel("Guidelines")
    plt.tight_layout()
    plt.savefig("plots/llm_guidelines.jpg")
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="EXAMPLES", y="BOUNDARY_F1", hue="LLM")
    plt.title("Interaction: LLM × Examples")
    plt.ylabel("Performance")
    plt.xlabel("Examples")
    plt.tight_layout()
    plt.savefig("plots/llm_examples.jpg")
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="EXAMPLES", y="BOUNDARY_F1", hue="LANGUAGE")
    plt.title("Interaction: Language × Examples")
    plt.ylabel("Performance")
    plt.xlabel("Examples")
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/lang_examples.jpg")
    plt.clf()



    #run_type3_anova(df,"BOUNDARY_F1", main_effects, interactions)
