# CHOD-Dataset
An End-to-End Entity Linking Dataset of Cultural Heritage Objects


# First Steps

## Required files and folders

- To download the model files (2.61 GB), execute the following bash line:

```bash
$ bash src/download_model_files.sh
```

## Set up

1. `docker`

Pull the following image:

```bash
$ docker pull c1587s/aat_elq_entity_linking:latest
```

Use the image in a container with shared volume:


```bash
$ docker run --gpus all --rm -it --name elq_cont -v $(pwd):/home/shared_volume c1587s/aat_elq_entity_linking
```

2. conda environment

Create and activate the environment:

```bash
$ conda create -n elqel -y python=3.7 && conda activate elqel
```

Install requirements:

```bash
$ pip install -r ./requirements.txt
```

## Prediction Examples

# Basic Example

To produce predictions, ELQ requires text inputs as lists of dictionaries in python-format, as in the example provided below:

```python
text_to_link = [{
    "id": "BM-A_1936-1012-44",
    "text": "Figure (woman) wearing Rainbow Dance costume. \
    Made of red, black, blue, gold painted earthenware.".lower(),
},

{
    "id": "BM-A_1940-0716-13",
    "text": "Figure (Gaṇesa). Folk deity,seated feet crossed holding 2 lotuses. \
    Made of bronze.".lower(),
}
]
```

Once input data is properly formatted, predictions can be obtained as follows:

```python
import os
from elqel.entity_linking import ELQEntityLinker

biencoder_path = "./biencoder/pytorch_model.bin"
models_path = "./models/"

# instantiate model
elq_model = ELQEntityLinker(models_path=models_path,
                            biencoder_path=biencoder_path,
                            prediction_type="unique")

# predict
predictions = elq_model.entity_linking(data_to_link_example)
```

**Output format**

You may have noticed the `prediction_type` parameter when instantiating the model from the above example. This parameter allows you to indicate whether the candidates from the “entity disambiguation step” must be returned

1. `prediction_type=='unique'`

Using ‘unique’ as prediction_type in ELQEntityLinker will produce prediction raw outputs, only including the information for the best candidate found by the model using the following fields:

-id (str): unique text identifier

-pred_triples (list of tuples): Each tuple contains the knowledge base ID, and indicates the tagged item using start and end tokens

-pred_tuples_string (list of lists): Each sublist contains linked aat title and tagged item (str), respectively

-scores (list of floats): List of scores for the best candidate

-text (str): text being annotated

-tokens tokens (list of ints): tokenized text

**Output Example:**

```python
{'id': 'BM-A_1936-1012-44',
'pred_triples': [('31871', 8, 9),
                ('30731', 19, 20),
                ('30193', 20, 23),
                ('64866', 1, 2)],
'pred_tuples_string': [['costume', 'costume'],
                        ['painted', 'painted'],
                        ['earthen ware', 'earthenware'],
                        ['human figures', 'figure']],
'scores': [8.12,
        5.84,
        5.38,
        4.93],
'text': 'figure (woman) wearing rainbow dance costume. made of red, black, '
        'blue, gold painted earthenware.',
'tokens': [101,
        ...
        102]}
```

2. `prediction_type=='multiple'`

Using `multiple` as `prediction_type` in `ELQEntityLinker` will produce prediction raw outputs, including the information for the bests candidates found by the model using the fields defined above.


3. To convert the prediction results to a DataFrame, the `preds2dataframe()` method can be used as follows:

```python
predictions_df = elq_model.preds2dataframe(save_path = "annotations/predictions_df.csv")
predictions_df
```

**Output Example:**

| id                | text                                                      | chunk_text                                             | chunk_start | chunk_end                 | aat         |           |    |           |
|-------------------|-----------------------------------------------------------|--------------------------------------------------------|-------------|---------------------------|-------------|-----------|----|-----------|
| BM-A_1936-1012-44 | figure (woman) wearing rainbow dance costume. made of red | black                                                  | blue        | gold painted earthenware. | figure      | 0         | 6  | 300404114 |
| BM-A_1936-1012-44 | figure (woman) wearing rainbow dance costume. made of red | black                                                  | blue        | gold painted earthenware. | costume     | 37        | 44 | 300178802 |
| BM-A_1936-1012-44 | figure (woman) wearing rainbow dance costume. made of red | black                                                  | blue        | gold painted earthenware. | painted     | 77        | 84 | 300161986 |
| BM-A_1936-1012-44 | figure (woman) wearing rainbow dance costume. made of red | black                                                  | blue        | gold painted earthenware. | earthenware | 85        | 96 | 300140803 |
| BM-A_1940-0716-13 | figure (ganesa). folk deity                               | seated feet crossed holding 2 lotuses. made of bronze. | figure      | 0                         | 6           | 300189808 |    |           |
| BM-A_1940-0716-13 | figure (ganesa). folk deity                               | seated feet crossed holding 2 lotuses. made of bronze. | feet        | 35                        | 39          | 300310200 |    |           |
| BM-A_1940-0716-13 | figure (ganesa). folk deity                               | seated feet crossed holding 2 lotuses. made of bronze. | bronze      | 75                        | 81          | 300010957 |    |           |
