#python preprocess_kenlm.py | ./kenlm/bin/lmplz -o 3 > /data/mourad/kenlm/trigram.arpa
./kenlm/build/bin/lmplz -o 3 < /data/mourad/kenlm/preprocessed/tokenized_all_data.txt > /data/mourad/kenlm/trigram_wiki-web-books.arpa
#build_binary /data/mourad/kenlm/trigram_wiki-web-books.arpa /data/mourad/kenlm/trigram_wiki-web-books.binary

