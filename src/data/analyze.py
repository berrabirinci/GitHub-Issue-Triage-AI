
import pandas as pd

# Load collected dataset
df = pd.read_csv("data/raw/github_issues_all.csv")

print("TOTAL ISSUES:", len(df))

# Count issues by repository
print("\nISSUES PER REPOSITORY:")
print(df["repository"].value_counts())

# Count labeled and unlabeled issues
print("\nLABELED ISSUES:")
print((df["labels"].fillna("").str.len() > 0).sum())

print("\nUNLABELED ISSUES:")
print((df["labels"].fillna("").str.len() == 0).sum())

# Analyze individual GitHub labels
all_labels = (
    df["labels"]
    .fillna("")
    .str.split(";")
    .explode()
)

all_labels = all_labels[all_labels != ""]

print("\nMOST COMMON GITHUB LABELS:")
print(all_labels.value_counts().head(30))
# Target category mapping

# Map GitHub labels to target categories
LABEL_MAPPING = {
    "bug": "Bug",
    "documentation": "Documentation",
    "docs": "Documentation",
    "new feature": "Feature Request",
    "enhancement": "Feature Request",
    "question": "Question"
}

# Find all matching categories for each issue
def map_labels(label_string):

    if pd.isna(label_string):
        return []

    labels = label_string.split(";")
    categories = set()

    for label in labels:

        normalized_label = label.strip().lower()

        if normalized_label in LABEL_MAPPING:
            categories.add(LABEL_MAPPING[normalized_label])

    return sorted(categories)


# Apply mapping to every issue
df["mapped_categories"] = df["labels"].apply(map_labels)

# Count issues with exactly one mapped category
single_category = df[
    df["mapped_categories"].apply(len) == 1
].copy()

single_category["category"] = (
    single_category["mapped_categories"].apply(lambda x: x[0])
)

# Find ambiguous and unmapped issues
ambiguous = df[
    df["mapped_categories"].apply(len) > 1
]

unmapped = df[
    df["mapped_categories"].apply(len) == 0
]

print("\nUNIQUE ISSUE CATEGORY COUNTS:")
print(single_category["category"].value_counts())

print("\nISSUES WITH MULTIPLE MATCHING CATEGORIES:")
print(len(ambiguous))

print("\nISSUES WITHOUT A MATCHING CATEGORY:")
print(len(unmapped))

print("\nTOTAL ISSUES:", len(df))
# Inspect 20 randomly selected unmapped issues

# Inspect 20 randomly selected unmapped issues

print("\n========== UNMAPPED ISSUE SAMPLES ==========")

sample_issues = unmapped.sample(
    n=min(20, len(unmapped)),
    random_state=42
).copy()

# Show only the first 200 characters of each issue body
sample_issues["body_preview"] = (
    sample_issues["body"]
    .fillna("")
    .str.replace("\n", " ", regex=False)
    .str.slice(0, 200)
)

print(
    sample_issues[
        ["repository", "title", "labels", "body_preview", "issue_url"]
    ].to_string(index=False)
)