wget -P stored_model https://www.dropbox.com/s/bbgyvwxozu3wruj/decoder
wget -P stored_model https://www.dropbox.com/s/ra866ly7acl2fcl/encoder
wget -P stored_model https://www.dropbox.com/s/w5vw8hyk3ujr6r8/vocab.txt
wget -P stored_model https://www.dropbox.com/s/al8nw81bsk1cf2p/word2vec.128d
python3 eval_main.py $1 $2