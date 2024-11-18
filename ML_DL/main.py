import testing.pull
import testing.encode
import testing.embed
import testing.cluster
import testing.tune_and_cv
import testing.train_model
import testing.results

def main():
    testing.pull.main()
    testing.encode.main()
    testing.embed.main()
    testing.cluster.main()
    testing.tune_and_cv.main()
    testing.train_model.main()
    testing.results.main()

if __name__ == "__main__":
    main()