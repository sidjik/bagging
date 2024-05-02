import cn45 as cn 
import random 
import mlMetrics as metrics 
import argparse 



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Bagging method with cn4.5 for review on 13 lab")
    parser.add_argument('-f', '--table_name')
    parser.add_argument('-y')
    parser.add_argument('-l', '--label')
    parser.add_argument('--treeCount')
    parser.add_argument('-p', '--portion_count')
    parser.add_argument('-b', '--boot_strap')
    args = parser.parse_args()
    train, test = cn.ReadCsv(args.table_name, args.y, args.label).returnDataSplitter().split_data(70)
    bagging = cn.BaggingCN45(train[0], train[1], int(args.treeCount), int(args.portion_count), int(args.boot_strap))
    tp, fp, tn, fn = bagging.return_prediction(test[0], test[1])
    accuracy = metrics.accuracy(tp+tn, len(test[1]))
    precision = metrics.precision(tp, fp)
    recall = metrics.recall(tp, fn)
    f1Scores = metrics.f1Scores(precision, recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Scores: {}".format(metrics.f1Scores(precision, recall)))
    metrics.confusionMatrix(tp, tn, fp, fn)

