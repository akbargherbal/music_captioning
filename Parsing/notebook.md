```python
#
```


```python
import pandas as pd
import regex as re
import os
from pathlib import Path
import json
```


```python
with open('../INPUT_FADL_SHAKER.json', encoding='utf-8') as f:
    original = json.loads(f.read())

def get_file_name(path):
    with open(path, encoding='utf-8') as f:
        data =  json.loads(f.read())
        return data['file']

list_json = []
for root, dirs, files in os.walk('./fadl_shaker_512/'):
    for file in files:
        if file.endswith('.json'):
            list_json.append(os.path.join(root, file).replace(os.sep, '/'))

list_done_files = [get_file_name(path) for path in list_json]
list_done_files
```




    ['01.Aash Men Shafak.mp3',
     '01.Esheqtak.mp3',
     '01.Fouk El Shouk.mp3',
     '01.Hobak Khayal.mp3',
     '01.Meta Habibi.mp3',
     '02.Aadiha.mp3',
     "02.Bayya'a El Qeloub.mp3",
     '02.El Qessa We Ma Fiha.mp3',
     '02.Yali Eshtarok Bel Dahab.mp3',
     '02.Yana Yana.mp3',
     '03.Adee Wala Tesalemsh.mp3',
     '03.Eyouno El Soud.mp3',
     '03.Hazzak Ya Albi.mp3',
     '03.Jar7etni 3ounou Essouda.mp3',
     '03.Nesity Ezay.mp3',
     '04.Khadarna 3la Balek.mp3',
     '04.Kul El Helween.mp3',
     '04.Ma Fi Magal.mp3',
     "04.We A'hd Allah.mp3",
     '04.Zaid El Malam.mp3',
     '05.El Meraya.mp3',
     "05.Ewa'a Tesada'ny.mp3",
     '05.Habaytoh.mp3',
     '05.Hayart 9albi Ma3ak.mp3',
     '05.Walla Zaman.mp3',
     '06.Awel Hob.mp3',
     "06.El Hob Shey' Tany.mp3",
     '06.Mserak Habibi.mp3',
     '06.Sawad El Ein.mp3',
     '06.Touba.mp3',
     '07.Ahbaby.mp3',
     '07.I7dhanou El Ayam.mp3',
     "07.Mamno'o Anni El Hawa.mp3",
     '07.Men Gher Sabab.mp3',
     '07.Yatkon Habibi.mp3',
     '08.Aamel Eih.mp3',
     '08.El Hob El Qadeem.mp3',
     '08.Hawa Ya Hawa.mp3',
     "08.Malleit Ana A'zar.mp3",
     '08.Nazra Wahda.mp3',
     '09.Inta 3omri.mp3']




```python

```


```python

```


```python

```
