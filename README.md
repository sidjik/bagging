### The main.py file is a file to run the algorithm that displays its metrics.
---
**Attention**: The program uses the following parameters when calling:
-f, --table_name: table file name
-y: column name that indicates the target attribute
-l, --label: how the positive class is labeled in the target attribute
--treeCount: number of trees used for Bagging
-p, --portion_count: the number of pieces into which the training data will be divided
-b, --boot_strap: number of pieces to use to build one tree

### Usage examples with data that is in a dir:
python main.py -f Employee.csv -y LeaveOrNot -l 1 -p 12 -b 5 --treeCount 10
python main.py -f apple_quality.csv -y Quality -l good -p 21 -b 7 --treeCount 12
python main.py -f startupdata.csv -y status -l closed -p 12 -b 5 --treeCount 7
