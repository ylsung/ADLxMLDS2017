# **Video captioning**
Given video feature, predict probable caption

## **Data**
MSVD

## **Requirement**
python >= 3.5 <br/>
pytorch == 0.2.0_4 (0.2.0) <br/>
numpy == 1.13.3 <br/>

OS == Ubuntu (Linux system)

## **Execution**

### **testing**

`hw2_seq2seq.sh [data directory] [test output] [peer output]` <br/>
ex: 
`hw2_seq2seq.sh 'data' 'test.txt' 'peer.txt'`

### **training**

train seq2seq model<br/>

`python main.py --todo 'train' --model 'seq2seq' --data [data directory] --save [where to save model]`

## **Author**

* Yilin Sung, r06942076@ntu.edu.tw