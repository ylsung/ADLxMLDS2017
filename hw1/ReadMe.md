# **Sequence Labeling**
Predict phone sequence by given feature of each phone

## **Data**
fbank, mfcc

## **Requirement**
python >= 3.5 <br/>
sklearn == 0.19.0 <br/>
pytorch == 0.2.0_4 (0.2.0) <br/>
numpy == 1.13.3 <br/>

OS == Ubuntu (Linux system)

## **Execution**

if you only have the authority of read, execute program by using commands below:

`bash hw1_rnn.sh [directory of data] [path for outfile (csv file)]`

`bash hw1_cnn.sh [directory of data] [path for outfile (csv file)]`

`bash hw1_best.sh [directory of data] [path for outfile (csv file)]`

ex: `bash hw1_best.sh ~/hw1/data ~/output/prediction.csv`

if you want to execute like: <br/>
`./hw1_rnn.sh [directory of data] [path for outfile (csv file)]` <br/>
you might have to get the authority of execution

## **Author**

* Yilin Sung, r06942076@ntu.edu.tw