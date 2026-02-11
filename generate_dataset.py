from sklearn.datasets import load_breast_cancer, fetch_california_housing, fetch_20newsgroups


def main():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv("data/california.csv", index=False)


if __name__ == "__main__":
    main()
