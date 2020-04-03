from CartPole import ModelDataPreparation
from CartPole import SplitData
from CartPole import DeepLearningModel
from CartPole import BotTest


def main():
    training_data = ModelDataPreparation.model_data_preparation()
    X, y = SplitData.split_data(training_data)
    print(X.shape)
    print(y.shape)
    model = DeepLearningModel.deep_learning_model(X, y)
    BotTest.test_bot(model)


if __name__ == '__main__':
    main()
