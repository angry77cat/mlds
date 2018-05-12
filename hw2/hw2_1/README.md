- Packages  
torch==0.3.1.post2  
numpy==1.14.1  
matplotlib==2.0.0  
gensim==3.4.0  

- Evaluation  
```eval```開頭的.py都是用於evaluation  
執行```bash hw2_seq2seq.sh $1 $2```  
```$1```: data dir  
```$2```: output.txt  

- 產生詞向量  
執行```python wordvec.py --make```(重複執行會使corpus.txt疊更多喔，所以--make一次就好)  
讀入```testing_label.json```及```training_label.json```後  
產生儲存一個所有句子的```model/corpus.txt```  
一個字典```model/vocab.txt```  
一個詞向量```model/word2vec.128d```  

- 訓練  
```config.py```  
儲存不需更動的參數   
```dataset.py```  
在dataset底下load所需的資料  
```models.py```  
儲存model架構，包含encoder、decoder及attention  
```self_bleu_eval.py```  
稍微修改原本的bleu_eval.py，以便做early-stop。將會產生```score_history.npy```儲存歷來的BLEU  
```model_seq2seq.py```  
主程式，包含訓練及驗證過程  

- 執行  
執行```python model_seq2seq.py```其他命令詳見--help    
必須要有```testing_data/```及```training_data/```底下所有東西  
以及```testing_label.json```與```training_label.json```。  
輸出時將會有一些紀錄，不太重要。  
當BLEU score表現佳時，在model/底下會儲存model的參數。  
