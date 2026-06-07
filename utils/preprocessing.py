import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MAX_MISSING = 0.50
MIN_TARGET_CORRELATION = 0.03
HIGH_UTILISATION = 10


def preprocessing(df):
    """Cleans raw MEPS data from 2023 to be used by models

    1. Select ~70 columns
    2. Replace MEPS's negative missing codes with NaN
    3. Recode binary features from MEPS's 1 or 2 -> 1 or 0
    4. Drop zero-weight rows
    5. Build UTILISATION target (1 if >=10 visits across all types, else 0)
    6. Create race binary protected attribute (1 = non-Hispanic White, 0 = Non-White, following AIF360's definition)
    7. Drop features with >50% missing or ~0 target correlation
    8. Fill missing binary with 0

    Returns:
        h251: cleaned dataframe
        FINAL_FEATURES: list of column names to use as model features
    """
    # 70 cols filtered
    FILTER_COLUMNS = [
        # unique identifiers
        "DUPERSID",
        "PANEL",
        "PERWT23F",
        "VARSTR",
        "VARPSU",
        # protected attributes,
        "RACETHX",
        "RACEV2X",
        "HISPANX",
        "RACEV1X",
        # demographic
        "AGE53X",
        "AGE23X",
        "AGELAST",
        "SEX",
        "MARRY53X",
        "MARRY23X",
        "REGION53",
        "REGION23",
        "EDUCYR",
        "HIDEG",
        "BORNUSA",
        "YRSINUS",
        # socioeconomic
        "POVCAT23",
        "POVLEV23",
        "INSCOV23",
        "UNINS23",
        "EMPST53",
        "FTSTU53X",
        "ACTDTY53",
        "EVERSERVED",
        # chronic deseases
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON31",
        "CHOLDX",
        "CANCERDX",
        "DIABDX_M18",
        "JTPAIN31_M18",
        "ARTHDX",
        "ARTHTYPE",
        "ASTHDX",
        "ADHDADDX",
        # self reported health status
        "RTHLTH53",
        "MNHLTH53",
        # health limitations
        "WLKLIM31",
        "ACTLIM31",
        "SOCLIM31",
        "COGLIM31",
        "DFHEAR42",
        "DFSEE42",
        # mental/behavioral health
        "K6SUM42",
        "PHQ242",
        "VPCS42",
        "VMCS42",
        "ADSMOK42",
        "OFTSMK53",
        # target variables => utilisation
        "OBTOTV23",
        "OPTOTV23",
        "ERTOT23",
        "IPNGTD23",
        "IPDIS23",
        "HHTOTD23",
        # access to care
        "HAVEUS42",
        "TYPEPE42",
        "AFRDCA42",
        "AFRDPM42",
        # costs - don't include, cost is a consequence of healthcare utilisation
        # "TOTEXP23",
        # "TOTSLF23",
    ]

    h251_filtered = df[FILTER_COLUMNS]

    # Replace negative values with NaN
    NA_VALUES = [-1, -7, -8, -9, -15]
    h251 = h251_filtered.replace(NA_VALUES, np.nan).copy()

    # replace if values are negative
    h251["POVLEV23"] = h251["POVLEV23"].where(h251["POVLEV23"] >= 0, np.nan)

    # Recode binary features from MEPS's 1 or 2 -> 1 or 0
    RECODE_BINARY = [
        # demographic
        "BORNUSA",
        "UNINS23",
        # chronic
        "HIBPDX",
        "CHDDX",
        "ANGIDX",
        "MIDX",
        "OHRTDX",
        "STRKDX",
        "EMPHDX",
        "CHBRON31",
        "CHOLDX",
        "CANCERDX",
        "DIABDX_M18",
        "JTPAIN31_M18",
        "ARTHDX",
        "ASTHDX",
        "ADHDADDX",
        # health limitations
        "WLKLIM31",
        "ACTLIM31",
        "SOCLIM31",
        "COGLIM31",
        "DFHEAR42",
        "DFSEE42",
        # mental/behavioral health
        "ADSMOK42",
    ]

    for col in RECODE_BINARY:
        if col in h251.columns:
            h251[col] = h251[col].map({1: 1, 2: 0})  # 1=yes, 0:no

    # Target variable: utilisation
    TARGET_FEATURES = ["OBTOTV23", "OPTOTV23", "ERTOT23", "IPNGTD23", "HHTOTD23"]

    # Drop zero weight rows
    h251 = h251[h251["PERWT23F"] > 0].copy()

    h251["UTILISATION_RAW"] = h251[TARGET_FEATURES].fillna(0).clip(lower=0).sum(axis=1)

    h251["UTILISATION"] = (h251["UTILISATION_RAW"] >= HIGH_UTILISATION).astype(int)

    # Protected attributes 'White' and 'Non-White'
    def race(row):
        if (row["HISPANX"] == 2) and (row["RACEV2X"] == 1):  # non-Hispanic White
            return "White"
        return "Non-White"

    h251["RACE"] = h251.apply(race, axis=1)
    h251["RACE_BINARY"] = h251["RACE"].map(
        {"White": 1, "Non-White": 0}
    )  # 1 = White (privileged), 0 = Non-White

    # Drop selected features
    PROTECTED_ATTRIBUTES = ["RACETHX", "RACEV2X", "HISPANX", "RACEV1X"]

    EXCLUDE = (
        PROTECTED_ATTRIBUTES
        + TARGET_FEATURES
        + [
            "DUPERSID",
            "PANEL",
            "UTILISATION",
            "UTILISATION_RAW",
            "RACE",
            "RACE_BINARY",
        ]
        + ["PERWT23F", "VARSTR", "VARPSU"]
        + [
            "AGE53X",
            "AGELAST",
            "REGION53",
            "MARRY53X",
        ]
        + ["IPDIS23"]
    )

    chosen_features = [c for c in h251.columns if c not in EXCLUDE]

    summary = (
        pd.DataFrame(
            {
                "porcentage_na": h251[chosen_features].isnull().mean(),
                "target": h251[chosen_features].corrwith(h251["UTILISATION"]).abs(),
                "race": h251[chosen_features].corrwith(h251["RACETHX"]).abs(),
            }
        )
        .sort_values("race", ascending=False)
        .round(4)
    )

    # Decide when to drop rows
    def drop_decision(row):

        if row["porcentage_na"] > MAX_MISSING:
            return "drop"

        if row["target"] < MIN_TARGET_CORRELATION:
            return "drop"
        return "keep"

    summary["drop_decision"] = summary.apply(drop_decision, axis=1)

    FINAL_FEATURES = summary[summary["drop_decision"] == "keep"].index.tolist()

    # Change NA for 0s
    NO_BINARY = [c for c in FINAL_FEATURES if c in RECODE_BINARY]
    for col in NO_BINARY:
        h251[col] = h251[col].fillna(0)

    # no minors smoking xd
    if "OFTSMK53" in FINAL_FEATURES:
        h251["OFTSMK53"] = h251["OFTSMK53"].fillna(0)

    return h251, FINAL_FEATURES


def load_data(csv_path="data/h251.csv", test_size=0.2, random_state=2):
    """Clean MEPS data and return train/test split.

    Calls preprocessing() to clean the data, splits into train/test, fills missing values,
    scales continuous features, one-hot encodes categorical features.

    Returns:
        X_train, X_test   : feature DataFrames (model inputs)
        y_train, y_test    : UTILISATION target (1 = high utiliser, 0 = low utiliser)
        race_train, race_test : RACE_BINARY protected attribute (1 = White, 0 = Non-White)
        w_train, w_test    : survey weights (PERWT23F)
    """
    df = pd.read_csv(csv_path)
    clean, final_features = preprocessing(df)

    X = clean[final_features]
    y = clean["UTILISATION"]
    race = clean["RACE_BINARY"]
    weights = clean["PERWT23F"]

    X_train, X_test, y_train, y_test, race_train, race_test, w_train, w_test = (
        train_test_split(
            X,
            y,
            race,
            weights,
            test_size=test_size,
            random_state=random_state,
        )
    )

    CONTINUOUS_FEATURES = [
        "AGE23X",
        "EDUCYR",
        "POVLEV23",
        "RTHLTH53",
        "MNHLTH53",
        "K6SUM42",
        "PHQ242",
        "VPCS42",
        "VMCS42",
    ]

    # Fill continuous missing values with median
    for col in CONTINUOUS_FEATURES:
        if col in X_train.columns:
            fill = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill)
            X_test[col] = X_test[col].fillna(fill)

    # Categorical values, filled with mode if missing and one-hot encoded
    CATEGORICAL_FEATURES = [
        "MARRY23X",
        "REGION23",
        "HIDEG",
        "EMPST53",
        "ACTDTY53",
        "EVERSERVED",
        "INSCOV23",
        # Skip BORNUSA because it's already binary
        # "BORNUSA",
        "SEX",
        "POVCAT23",
        "HAVEUS42",
        "TYPEPE42",
        "AFRDCA42",
        "AFRDPM42",
    ]

    # Fill categorical missing values with mode (most common category)
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            fill = X_train[col].mode().iloc[0]
            X_train[col] = X_train[col].fillna(fill)
            X_test[col] = X_test[col].fillna(fill)

    # Scale and one-hot encode on training data only
    scale_cols = [c for c in CONTINUOUS_FEATURES if c in X_train.columns]
    onehot_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    encoder = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_cols,
            ),
        ],
        remainder="passthrough",  # binary 0/1 features pass through unchanged
        verbose_feature_names_out=False,
    )
    encoder.set_output(transform="pandas")  # keep DataFrames

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, y_train, y_test, race_train, race_test, w_train, w_test
