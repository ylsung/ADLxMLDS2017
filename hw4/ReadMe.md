# **Conditional GAN**
Use GAN to generate images by given specific tags.

## **Dataset**
Anime dataset.

### **tags**

\[color hair\]: <br/>
'orange hair', 'white hair', 'aqua hair', 'gray hair',
'green hair', 'red hair', 'purple hair', 'pink hair',
'blue hair', 'black hair', 'brown hair', 'blonde hair'.

\[color eyes\]: <br/>
'gray eyes', 'black eyes', 'orange eyes',
'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
'green eyes', 'brown eyes', 'red eyes', 'blue eyes'.


## **Requirement**
python ==3.6.3 <br/>
pytorch == 0.2.0_4 (0.2.0) <br/>
torchvision <br/>
scipy == 1.0.0 <br/>
scikit-image == 0.13.1 <br/>

OS == Linux syste (Ubuntu, Arch Linux)

## **Execution**

### **testing**

Test by default-setting: <br/>
`bash run.sh [test_file.txt]` <br/>
Test by self-train model <br/>
`python3.6 main.py --todo 'test' --load [model_path] --model_id [model_id] --te_data [test_file.txt]` <br/>
You have to train the model first before executing the command above.

### **training**

Train the model by default setting <br/>

`bash train.sh` <br/>

Train your model<br/>

`python3.6 main.py --save [model_path]`

## **Author**

* Yilin Sung, r06942076@ntu.edu.tw
