from sklearn.datasets import load_breast_cancer, fetch_california_housing, fetch_20newsgroups


def main():
    data = fetch_20newsgroups(as_frame=True)
    df = data.frame
    df.to_csv("data/20newsgroup.csv", index=False)


if __name__ == "__main__":
    main()
