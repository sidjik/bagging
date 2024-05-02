import cn45 as cn
import random
import mlMetrics as metrics
import argparse
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bagging method with cn4.5")
    parser.add_argument('-f', '--table_name')
    parser.add_argument('-y')
    parser.add_argument('-l', '--label')
    parser.add_argument('--treeCount')
    parser.add_argument('-p', '--portion_count')
    parser.add_argument('-b', '--boot_strap')
    args = parser.parse_args()
    print(args.table_name, args.label, args.y)

    color = ['blue', 'red', 'green']
    for percent, color in [[100, 'blue'], [7, 'red'], [12, 'green']]:
    #for treeCount, portion_count, boot_strap, color, label in [[1, 1, 1, 'blue', 'single_tree'], [12, 12, 5, 'green', "bagging"]]:  
        accuracy = []
        itteration = []
        for i in range(100):
            train, test = cn.ReadCsv(args.table_name, args.y, args.label).returnDataSplitter().split_data(70)
            #bagging = cn.BaggingCN45(train[0], train[1], int(args.treeCount), int(args.portion_count), int(args.boot_strap))
            bagging = cn.BaggingCN45(train[0], train[1], percent, 12, 7)
            accuracy.append(bagging.make_prediction(test[0], test[1], True))
            itteration.append(i+1)
        plt.scatter(itteration, accuracy, marker='o', color=color, label=percent)
    plt.title('apple_quality.csv 100 iter')
    plt.xlabel("Itteration count")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    exit()

    train, test = cn.ReadCsv(args.table_name, args.y, args.label).returnDataSplitter().split_data(70)
    bagging = cn.BaggingCN45(train[0], train[1], int(args.treeCount), int(args.portion_count), int(args.boot_strap))
    tp, fp, tn, fn = bagging.return_prediction(test[0], test[1])
    precision = metrics.precision(tp, fp)
    recall = metrics.recall(tp, fn)
    print("Accuracy: {}".format(metrics.accuracy(tp+tn, len(test[1]))))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 scores: {}".format(metrics.f1Scores(precision, recall)))
    metrics.confusionMatrix(tp, tn, fp, fn)
