import pandas as pd

def split_data(path="training_projects.csv"):
    df = pd.read_csv(path)

    print("Total completed projects:", len(df))

    # Sort so it's stable and repeatable
    df_sorted = df.sort_values("project_id").reset_index(drop=True)

    # Leave last 2 projects as test set
    test_df = df_sorted.tail(2)
    train_df = df_sorted.head(len(df_sorted)-2)

    print("\nTraining set:", train_df["project_id"].tolist())
    print("Test set:", test_df["project_id"].tolist())

    # Save out
    train_df.to_csv("training_projects_train.csv", index=False)
    test_df.to_csv("training_projects_test.csv", index=False)

    print("\nSaved:")
    print(" - training_projects_train.csv")
    print(" - training_projects_test.csv")


if __name__ == "__main__":
    split_data()
