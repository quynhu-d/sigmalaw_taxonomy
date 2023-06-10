# Построение таксономии судебных дел Верховного суда США (корпус SigmaLaw) 
## 0.	Данные
В качестве данных взят корпус [SigmaLaw](https://osf.io/qvg8s/wiki/home/),  собранный авторами статьи [Synergistic Union of Word2Vec and Lexicon for Domain Specific Semantic Similarity](). 
Корпус состоит из 39155 текстов судебных дел Верховного суда США с 76 категориями. 
Для данной работы отобраны 10 из 76 категорий, наиболее схожих с категориями дел Арбитражного суда РФ (приведены ниже), а также 3 категории из иных областей для сравнения.

1. "Administrative Law",
4. "Antitrust &amp; Trade Regulation",
6. "Banking Law",
7. "Bankruptcy Law",
11. "Commercial Law",
13. "Constitutional Law",
16. "Contracts",
17. "Corp. Governance",
24. "Dispute Resolution &amp; Arbitration",
39. "Government Contracts",
*********************************
9. 'Civil Rights',
19. 'Criminal Law &amp; Procedure',
44. 'Injury &amp; Tort Law'

## 1.	Обработка текстов. Выделение фраз
Обработка включает следующие шаги: 
-	удаление знаков препинания
-	удаление имён числительных, союзов и других вспомогательных частей речи
-	удаление стоп-слов: наиболее часто встречаемые слова английского языка 
(‘that’, ‘because’ и т.п., в качестве базового списка стоп-слов взят список из библиотеки nltk), а также часто встречаемые стоп-слова в корпусе (‘regarding’, ‘however’, ‘jurisdiction’, ‘whether’)
-	удаление слов длины меньше 3
```
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won['’‘`]t", "will not", phrase)
    phrase = re.sub(r"can['’‘`]t", "can not", phrase)
    phrase = re.sub(r"ain['’‘`]t", "am not", phrase)

    # general
    phrase = re.sub(r"n['’‘`]t", " not", phrase)
    phrase = re.sub(r"['’‘`]re", " are", phrase)
    phrase = re.sub(r"['’‘`]s", " is", phrase)
    phrase = re.sub(r"['’‘`]d", " would", phrase)
    phrase = re.sub(r"['’‘`]ll", " will", phrase)
    phrase = re.sub(r"['’‘`]t", " not", phrase)
    phrase = re.sub(r"['’‘`]ve", " have", phrase)
    phrase = re.sub(r"['’‘`]m", " am", phrase)

    #phrase = re.sub('([.;!?])', r' \1 ', phrase)
    phrase = re.sub(r'[^\w.?!;]', ' ', phrase)
    phrase = re.sub(' +', ' ', phrase)
    sentences = re.split('([.;!?] *)', phrase)

    return ' '.join([i.capitalize() for i in sentences])
```
```
nlp = en_core_web_lg.load(disable=['parser'])
nlp.max_length = 5000000

import re


def prepare_text(text, drop_words=[]):
    pat = re.compile(r'[^A-Za-z]+')
    cleared_text = re.sub(pat, ' ', decontracted(text))
    nlp_doc = nlp(cleared_text.strip())
    pos_drop_dict = ['ADP', 'AUX', 'CONJ', 'CCONJ', 'DET', 'PUNCT', 'SYM', 'X', 'SPACE', 'NUM']
    prepared_text = ''
    for token in nlp_doc:
        if token.pos_ in pos_drop_dict:
            continue
        elif len(token) <= 3:
            continue
        elif token.lemma_ in drop_words:
            continue

        else:
            prepared_text += str(token).lower() + ' '
    return prepared_text[:-1]
```
Предобработанные тексты расположены в папке `/data`. 

Для каждой категории из предобработанных текстов выделяются множества фраз-биграмм. Из каждого множества удаляются фразы, наиболее часто встречаемые по всему корпусу (100 первых фраз), а также фразы с частотой встречаемости в категории менее, чем 1е-5.
Одинаковые фразы со словами, начальные формы которых одинаковы, объединяются в одно множество (см. далее в п. 2). Например, фразы declaring bankruptcy, bankruptcy declared объединяются в множество $T_{\text{bankruptcy declare}}$ = {‘declaring bankruptcy’, ‘bankruptcy declared’, ‘bankruptcy declaring’…}.
Словари, содержающие множества собранных фраз, находятся в папке ``.
```
'provision unenforceable': {'provision unenforceable',
  'provisions unenforceable',
  'unenforceable provision',
  'unenforceable provisions'},
 'state unenforceable': {'unenforceable state'},
 'make state': {'made state',
  'made states',
  'make state',
  'making state',
  'making stated',
  'state made',
  'state make',
  'state makes',
  'stated made',
  'stated make',
  'states made',
  'states make',
  'states making',
  'stating making'},
  ...
```
|Category|# documents|# phrases|# sets (similar phrases)|
|:----|:----|:----|:----|
|Administrative Law|1053|13655|11391|
|Commercial Law|771|14697|12361|
|Constitutional Law|534|13166|11137|
|Contracts|1059|13889|11181|
|Corp.Governance|146|45665|38404|
|Criminal Law &amp; Procedure|405|18875|15479|
|Dispute Resolution &amp; Arbitration|864|16557|12855|
|Government Contracts|353|25207|20704|
|Antitrust &amp; Trade Regulation|278|18048|15024|
|Injury &amp; Tort Law|1000|13480|10969|
|Banking Law|505|16371|13345|
|Bankruptcy Law|988|15800|12375|
|Civil Rights|866|12704|10502|

## 2.	Построение матрицы релевантности документов-фраз
С помощью аннотированных суффиксных деревьев ([AST]()) считаем релевантность документов-фраз. 
Полные таблицы для каждой категории приведены в `/relevance_matrices/`.
```
matrices_dict = {}
for cat_id in final_cat_id:
    matrices_dict[cat_id] = np.load(f'../RES/relevance_matrices/{cat_id}_relevance_matrix_new.npy')

matrices_dict['16'].shape

>>> (1059, 13889)

len(final_substrings['16'])

>>> 13889
```

```
pd.DataFrame(matrices_dict['16'], columns=final_substrings['16'])
```
||insurance policy|contract claim|national union|state farm|insurance company|bargaining agreement|declaratory judgment|duty defend|purchase agreement|fiduciary duty|...|somewhat different|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|0|0.155288|0.301025|0.259554|0.169677|0.136015|0.439219|0.288281|0.078027|0.301091|0.046699|...|0.217674|
|1|0.105563|0.309418|0.211125|0.132222|0.131017|0.333497|0.418511|0.219038|0.319678|0.139401|...|0.224079|
|2|0.336342|0.203331|0.278760|0.140525|0.339379|0.252770|0.395075|0.248716|0.348564|0.090245|...|0.240544|
|3|0.131695|0.213948|0.280032|0.158949|0.152353|0.279968|0.293830|0.139243|0.261532|0.064790|...|0.320405|
|4|0.120229|0.238712|0.278670|0.134154|0.146509|0.309971|0.285762|0.169373|0.296134|0.073605|...|0.123978|
|...|...|...|...|...|...|...|...|...|...|...|...|...|
|1054|0.158320|0.267465|0.323955|0.110780|0.172185|0.470149|0.254765|0.156851|0.292698|0.059915|...|0.208339|
|1055|0.103198|0.278135|0.266870|0.142642|0.155817|0.298745|0.269439|0.248648|0.364146|0.330214|...|0.217807|
|1056|0.134878|0.266327|0.256261|0.143996|0.101463|0.282342|0.270063|0.159434|0.267849|0.073110|...|0.239327|
|1057|0.249496|0.272195|0.272539|0.133508|0.275326|0.344304|0.335216|0.089828|0.255859|0.061597|...|0.219461|
|1058|0.184335|0.310821|0.242651|0.179676|0.270103|0.309056|0.276753|0.193026|0.280953|0.120458|...|0.197454|

Для каждого документа из фраз одного множества (т.е. схожих фраз, см. п.1) отбираем максимальное значение релевантности. 
```
bow_matrices = {}
bow_phrases = {}
for cat_id in tqdm(final_cat_id):
    phrases = list(sorted(set(substring_dictionaries_reverse[cat_id][w] for w in final_substrings[cat_id])))
    matrix = matrices_dict[cat_id]

    bow_matrix = []
    for phrase in tqdm(phrases, leave=False):
        idx = []
        for p in substring_dictionaries[cat_id][phrase]:
            if p in final_substrings[cat_id]:
                idx.append(final_substrings[cat_id].index(p))
        bow_matrix.append(matrix[:, idx].max(axis=1))
    bow_matrices[cat_id] = bow_matrix
    bow_phrases[cat_id] = phrases
```
Итоговая матрица релевантности  ($R_{ij}$) имеет размеры $D \times N$, где $D$ --- число документов, а $N$ --- число различных фраз.

## 3.	Выделение релевантных фраз
Для каждой фразы находим среднее значение (“meanval”) релевантности по 50 наиболее релевантным документам. 
Отбираем фразы с полученными значениями meanval больше 0.3 и в качестве наиболее релевантных фраз объявляем первые 5% фраз.

```
def get_most_relevant_substrings(matrix, substrings, thrsh=0.3, max_num=100, rel_fraction=0.05):
    assert matrix.shape[1] == len(substrings)
    phrase_vals = matrix[:, 0]

    phrase_meanval = []
    for phrase_vals in matrix.T:
        phrase_meanval.append(np.mean(np.sort(phrase_vals)[-max_num:]))
    phrase_meanval = np.array(phrase_meanval)

    tmp_idx = np.where(phrase_meanval > thrsh)[0]

    most_relevant_idx = tmp_idx[np.argsort(phrase_meanval[tmp_idx])[-int(tmp_idx.shape[0] * rel_fraction):]]
    return pd.DataFrame(
        {'phrase': substrings[most_relevant_idx][::-1], 'mean_score': phrase_meanval[most_relevant_idx][::-1]}
    )
rel_phrases_dict = {}
for cat_id in tqdm(final_cat_id):
    matrix = bow_matrices[cat_id]
    substrings = np.array(bow_phrases[cat_id])
    rel_phrases_dict[cat_id] = get_most_relevant_substrings(matrix, substrings, max_num=50)
print(rel_phrases_dict['16'])    
>>>
                           phrase  mean_score
0         arbitration arbitration    0.552296
1      arbitration interpretation    0.541906
2    fraudulent misrepresentation    0.532886
3        agreement interpretation    0.529315
4         fraud misrepresentation    0.528854
..                            ...         ...
333             arbitration scope    0.453503
334       arbitration controversy    0.453503
335          clause unenforceable    0.453357
336            agreement employee    0.453308
337              agreement follow    0.453300

[338 rows x 2 columns]
```

Статистика выделенных релевантных фраз для каждой категории:
|Category|# relevant phrases|Average relevance value range|
|:----|:----|:----|
|Administrative Law|349|0.46-0.52|
|Commercial Law|332|0.44-0.53|
|Constitutional Law|283|0.44-0.53|
|Contracts|338|0.45-0.55|
|Corp.Governance|619|0.41-0.52|
|Criminal Law &amp; Procedure|375|0.43-0.52|
|Dispute Resolution &amp; Arbitration|412|0.48-0.58|
|Government Contracts|496|0.44-0.53|
|Antitrust &amp; Trade Regulation|326|0.42-0.54|
|Injury &amp; Tort Law|314|0.44-0.53|
|Banking Law|348|0.43-0.52|
|Bankruptcy Law|389|0.46-0.59|
|Civil Rights|304|0.45-0.60|

Списки отобранных релевантных фраз для каждой категории расположены в `relevant_bow/{cat_id}_bow.xlsx`.

## 4.	Выделение список слов-наполнителей категорий
Из релевантных фраз выделяем слова, которые встречаются более чем 6 раз. 
```
word_frequency_dict = {}
for cat_id in final_cat_id:
    w_f = Counter(' '.join(rel_phrases_dict[cat_id].phrase).split())
    word_frequency_dict[cat_id] = pd.DataFrame(
        {'word': w_f.keys(), 'frequency': w_f.values()}).sort_values('frequency', ascending=False)
word_frequency_dict['16'].query('frequency > 6')

>>>
                word  frequency
0        arbitration         63
1       jurisdiction         58
2          agreement         39
3     interpretation         26
4               that         10
5      determination          9
6          provision          9
7  misrepresentation          8
8        termination          8
9           contract          7
```

Отбираем дела, наиболее релевантные данным спискам слов: для каждого документа находим максимальное значение релевантности (словам из списка) и по полученным значениям отбираем первые 10% документов.
```
doc_to_word_dict = {}
for cat_id in tqdm(final_cat_id):
    doc_to_word_dict[cat_id] = get_corelevance_matrix(
        prepared_texts[cat_id], word_frequency_dict[cat_id].query('frequency > 6').word, k=1
    )
    
relevant_doc_id_dict = {}
for cat_id in final_cat_id:
    doc_to_word = doc_to_word_dict[cat_id]
    word_frequency = word_frequency_dict[cat_id].query('frequency > 6')
    relevant_doc_id_dict[cat_id] = np.argsort(
        doc_to_word.max(axis=1)
    )[-int(doc_to_word.shape[0] * 0.1):][::-1]
```
|Category|# relevant documents|
|:----|:----|
|Administrative Law|105|
|Commercial Law|77|
|Constitutional Law|53|
|Contracts|105|
|Corp.Governance|14|
|Criminal Law &amp; Procedure|40|
|Dispute Resolution &amp; Arbitration|86|
|Government Contracts|35|
|Antitrust &amp; Trade Regulation|27|
|Injury &amp; Tort Law|100|
|Banking Law|50|
|Bankruptcy Law|98|
|Civil Rights|86|

Номера релевантных документов можно найти в `relevant_doc_id_dict.npy`. В `doc_to_word/{cat_id}_matrix.npy` находятся матрицы релевантности документам спискам слов.

## 5.	Выделение наиболее релевантных фраз для отобранных документов
Повторяем процедуру, описанную в п. 3, но для отобранных документов. Списки отобранных фраз расположены в папке `category_phrases`.
```
print(category_phrases['16'])
>>>
                        phrase  mean_score
0            judgment judgment    0.415170
1        agreement termination    0.412065
2           agreement judgment    0.402458
3       determination district    0.401826
4       contract determination    0.399752
5          agreement agreement    0.398007
6         agreement obligation    0.397804
7     agreement interpretation    0.397457
8          agreement provision    0.397129
9        provision termination    0.392126
10    interpretation provision    0.389592
11             judgment motion    0.389397
12        contract termination    0.388398
13       agreement contractual    0.387740
14          agreement question    0.387402
15  application interpretation    0.387228
16  contractual interpretation    0.386429
17      contractual obligation    0.385691
18         court determination    0.384388
19       agreement application    0.384235
20         provision provision    0.384006
21           contract judgment    0.383274
22          provision question    0.383173
23   interpretation reasonable    0.382953
24     contract interpretation    0.382451
25        determination regard    0.382304
26      agreement construction    0.382123
27          determination make    0.381870
28           decision judgment    0.381465
29          agreement contract    0.380993
30        district termination    0.379727
31          agreement district    0.379490
32        agreement reasonable    0.378426
33          addition agreement    0.377187
34      consideration district    0.376678
35       contractual provision    0.376262
36           question question    0.374988
37        provision reasonable    0.374732
38         agreement violation    0.374507
39       determination factual    0.374282
40           agreement include    0.373976
41           agreement provide    0.373344
42             action judgment    0.372167
43          determination upon    0.371592
44          citation quotation    0.371462
45        district requirement    0.371067
46          district provision    0.370348
47          contract provision    0.369338
48        limitation provision    0.368249
49            date termination    0.367801
50          obligation provide    0.367627
51    district reconsideration    0.367293
52         district litigation    0.366763
53  enforcement interpretation    0.365494
54      motion reconsideration    0.365338
**********
```

|Category|# relevant phrases|Average relevance value range|
|:----|:----|:----|
|Administrative Law|45|0.38-0.44|
|Commercial Law|31|0.36-0.41|
|Constitutional Law|55|0.38-0.44|
|Contracts|55|0.37-0.42|
|Corp.Governance|222|0.38-0.45|
|Criminal Law &amp; Procedure|74|0.38-0.45|
|Dispute Resolution &amp; Arbitration|124|0.43-0.51|
|Government Contracts|114|0.39-0.47|
|Antitrust &amp; Trade Regulation|73|0.37-0.41|
|Injury &amp; Tort Law|39|0.36-0.40|
|Banking Law|39|0.36-0.41|
|Bankruptcy Law|59|0.42-0.52|
|Civil Rights|58|0.38-0.44|

## 6.	Построение матрицы корелевантности фраз
Пусть $R$ – матрица релевантности (оставляем только фразы, отобранные в предыдущем пункте).
```
new_relevance_matrix_dict = {}
for cat_id in final_cat_id:
    rel_phrases = rel_phrases_dict_[cat_id]
    matrix = bow_matrices[cat_id]
    substrings = bow_phrases[cat_id]

    new_relevance_matrix_dict[cat_id] = matrix[:, [substrings.index(el) for el in rel_phrases.phrase.to_list()]]
```
Тогда матрицу корелевантности $C$ получим следующим образом: $C = R^T x R / n$, где $n$ – число текстов, для которых релевантность фразы выше 0.28.
```
corelevance_matrix_dict = {}
for cat_id in final_cat_id:
    rel_matrix = new_relevance_matrix_dict[cat_id]
    n_relevant_for_texts = (rel_matrix > 0.28).sum(axis=1)
    corelevance_matrix_dict[cat_id] = (rel_matrix.T / n_relevant_for_texts) @ rel_matrix
```
Итоговые матрица корелевантности лежат в `corelevance_matrices`.

## 7.	Кластеризация FADDIS
Применяем метод кластеризации [FADDIS]() на матрице корелевантности $C$ с предварительным преобразованием Лапина. 
В каждом кластере удаляем фразы со степень принадлежности ниже порога 0.21.
Отбираем значимые кластера:
Кластер считается значимым, пока вклад кластера не меняется в 1.5 раз по сравнению со вкладом предыдущего кластера (см. функцию `transition_idx`), либо вклад кластера становится ниже порога 0.01 (см. функцию `threshold_idx`). 
В случае, если по значениям вклада не выявлены значимые кластера, для категории берём два кластера с наибольшим вкладом.
```
def transition_idx(cbs, thrsh=1.5):
    trs = np.where((cbs[:-1] / cbs[1:]) >= thrsh)[0]
    if len(trs) == 0:
        return len(cbs)
    return trs[0] + 1

def threshold_idx(cbs, thrsh=0.01):
    trs = np.where(cbs < thrsh)[0]
    if len(trs) == 0:
        return len(cbs)
    return trs[0]

for cat_id in final_cat_id:
    original_stdout = sys.stdout
    rel_phrases = rel_phrases_dict_[cat_id]
    annotations = rel_phrases.phrase
    B, member, contrib, intensity, lat, tt = faddis_dict[cat_id]
    idx = np.argsort(contrib)[::-1]
    i_ = min(
        threshold_idx(contrib[idx], 0.01),  # threshold 
        transition_idx(contrib[idx], 1.5)  # transition
    )
    with open(f'../RES/fuzzy_clusters/{cat_id}_clusters.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        print(f'Category {cat_id};', f'{i_} clusters')
        print('__' * 50)
        for cl_idx, (cluster, cb) in enumerate(zip(member[:, idx].T, contrib[idx])):
            print(f'Cluster {cl_idx}; contribution {cb:.4f}:')
            print([(phrase, f"{val:.4f}") for phrase, val in list(sorted(zip(annotations, cluster.flat),
                                                                         key=itemgetter(1), reverse=True)) if
                   val > 0.21])
            print('\n')
    sys.stdout = original_stdout
```
|Category|# significant clusters|cluster significance value range|
|:----|:----|:----|
|Administrative Law|18|0.0107-0.0208|
|Commercial Law|17|0.0178-0.0296|
|Constitutional Law|10|0.0103-0.0158|
|Contracts|14|0.0102-0.0163|
|Corp.Governance|2|0.0751-0.1067|
|Criminal Law &amp; Procedure|11|0.0101-0.0171|
|Dispute Resolution &amp; Arbitration|2|0.0075-0.0076|
|Government Contracts|2|0.0077-0.0082|
|Antitrust &amp; Trade Regulation|4|0.0104-0.0131|
|Injury &amp; Tort Law|18|0.0102-0.0225|
|Banking Law|15|0.0120-0.0213|
|Bankruptcy Law|13|0.0103-0.0161|
|Civil Rights|13|0.0102-0.0159|
Результат кластеризации сохранён в `faddis_dict.npy`.
```
faddis_dict = np.load('faddis_dict.npy', allow_pickle=True).item()
B, member, contrib, intensity, lat, tt = faddis_dict['16']
```

## 8.	Построение таксономии
В качестве названий кластера берём фразу с наибольшей степенью принадлежности кластера, а также вторую фразу, если её степень принадлежности выше 0.35. Строим дерево таксономии.
![alt text](https://github.com/quynhu-d/sigmalaw_taxonomy/blob/main/sigmalaw_taxonomy.jpeg?raw=true)
Дочерние кластера категории 16:
![alt text](https://github.com/quynhu-d/sigmalaw_taxonomy/blob/main/16_taxonomy.jpeg?raw=true)

Все графики находятся в `taxonomy_visualization`.

## 9. Дефаззификация
Проверим получившиеся кластера-детей. Для этого проведём процесс дефаззификации и сравним схожесть получившихся чётких кластеров по мере Кульчинского.

#TODO: пример для 16.

Все графики приведены в папке `defuzzification`.

