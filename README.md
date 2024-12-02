# The HierText Dataset

![samples](docs/images/dataset.png)

[![Paper](https://img.shields.io/badge/Paper-arxiv.2203.15143-green)](https://arxiv.org/abs/2203.15143)
[![Code](https://img.shields.io/badge/Code-TFModel%2FUnifiedDetector-green)](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector)
[![Competition](https://img.shields.io/badge/Competition-RobustReadingCompetition-red)](https://rrc.cvc.uab.es/?ch=18)


## News

- 2024.12.02: The test set annotation has been released so any one can use the eval script to evaluate their method directly. See the __Dataset downloading & processing__ section for details.
- 2023.10.29: Our new paper for [joint text spotting and layout analysis](https://arxiv.org/abs/2310.17674) is accepted to WACV 2024!
- 2023.08.23: Oral presentation at [ICDAR 2023](https://icdar2023.org/program/sessions/#:~:text=O11.2,and%20Michalis%20Raptis) for our competition.
- 2023.05.17: Competition report available [here](https://arxiv.org/abs/2305.09750).
- 2023.04.14: We have released the results [(click here)](https://github.com/google-research-datasets/hiertext/blob/main/docs/images/ICDAR_2023_Competition_on_Hierarchical_Text_Detection_and_Recognition_result_and_ranking_.pdf) of  __ICDAR 2023 Competition on Hierarchical TextDetection and Recognition__. Congratulations and thanks for all the efforts!
- 2022.12.12: We will be hosting __ICDAR 2023 Competition on Hierarchical TextDetection and Recognition__ with HierText! The competition will be held on the [Robust Reading Comprehension](https://rrc.cvc.uab.es/?ch=18&com=introduction) website, including two tasks, (1) __Hierarchical Text Detection__ and (2) __Word-Level End-to-End Text Detection and Recognition__. See the website for more info.
- 2022.08.17: The evaluation server is launched on [Robust Reading Comprehension](https://rrc.cvc.uab.es/?ch=18&com=introduction). Now you can submit the result of your method on this website.
- 2022.05.11: The Out-of-Vocabulary competition is launched as part of the [Text-in-Everything](https://sites.google.com/corp/view/tie-eccv2022/challenge) workshop at ECCV 2022. HierText is incorporated to construct the benchmark dataset.
- 2022.06.02: Code and weights for the unified detector model are released in [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector).
- 2022.03.03: Paper accepted to CVPR 2022.


## Overview

HierText is the first dataset featuring hierarchical annotations of text in
natural scenes and documents. The dataset contains 11639 images selected from
the
[Open Images dataset](https://storage.googleapis.com/openimages/web/index.html),
providing high quality word (~1.2M), line, and paragraph level annotations. Text
lines are defined as connected sequences of words that are aligned in spatial
proximity and are logically connected. Text lines that belong to the same
semantic topic and are geometrically coherent form paragraphs. Images in
HierText are rich in text, with average of more than 100 words per image.

We hope this dataset can help researchers developing more robust OCR models and
enables research into unified OCR and layout analysis. Check out our
[paper](https://arxiv.org/pdf/2203.15143.pdf) for more details.

### Opensourcing Unified Detector
In the paper, we also propose a novel method called **Unified Detector**, that
unifies text detection and layout analysis. The code and pretrained checkpoint
is now available at this [repository](https://github.com/tensorflow/models/tree/master/official/projects/unified_detector)

## Getting Started

First clone the project:

```
git clone https://github.com/google-research-datasets/hiertext.git
```

(Optional but recommended) Create and enter a virtual environment:

```
sudo pip install virtualenv
virtualenv -p python3 hiertext_env
source ./hiertext_env/bin/activate
```

Then install the required dependencies using:

```
cd hiertext
pip install -r requirements.txt
```

### Dataset downloading & processing

The ground-truth annotations of `train` / `validation` / `test` sets are stored in
`gt/train.jsonl.gz`, `gt/validation.jsonl.gz`, `gt/test.jsonl.gz` respectively. Use the following
command to decompress the two files:

```
gzip -d gt/*.jsonl.gz
```

The images are hosted by [CVDF](http://www.cvdfoundation.org/). To download them
one needs to install
[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
and run the following:

```
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/test.tgz .
tar -xzvf train.tgz
tar -xzvf validation.tgz
tar -xzvf test.tgz
```

### Dataset inspection and visualization

Run the visualization notebook locally to inspect the data using:

```
jupyter notebook HierText_Visualization.ipynb
```

## Dataset Description

We split the dataset into `train` (8281 images), `validation` (1724 images) and
`test` (1634 images) sets. Users should train their models on `train` set,
select the best model candidate based on evaluation results on `validation` set
and finally report the performance on `test` set.

There are five tasks:

-   Word-level
    -   Word detection (polygon)
    -   End-to-end
-   Line-level
    -   Line detection (union of words)
    -   End-to-end
-   Paragraph detection (union of words)

### Images

Images in HierText are of higher resolution with their long side constrained to
1600 pixels compared to previous datasets based on Open Images that are
constrained to 1024 pixels. This results in more legible small text. The
filename of each image is its corresponding image ID in the Open Images dataset.
All images are stored in JPG format.

### Annotations

The ground-truth has the following format:

```jsonc
{
  "info": {
    "date": "release date",
    "version": "current version"
  },
  "annotations": [  // List of dictionaries, one for each image.
    {
      "image_id": "the filename of corresponding image.",
      "image_width": image_width,  // (int) The image width.
      "image_height": image_height, // (int) The image height.
      "paragraphs": [  // List of paragraphs.
        {
          "vertices": [[x1, y1], [x2, y2],...,[xn, yn]],  // A loose bounding polygon with absolute values.
          "legible": true,  // If false, the region defined by `vertices` are considered as do-not-care in paragraph level evaluation.
          "lines": [  // List of dictionaries, one for each text line contained in this paragraph. Lines in paragraph may not follow the reading order.
            {
              "vertices": [[x1, y1], [x2, y2],...,[x4, y4]],  // A loose rotated rectangle with absolute values.
              "text": "the text content of the entire line",
              "legible": true,  // A line is legible if and only if all of its words are legible.
              "handwritten": false,  // True for handwritten text, false for printed text.
              "vertical": false,  // If true, characters have a vertical layout.
              "words": [  // List of dictionaries, one for each word contained in this line. Words inside a line follows the reading order.
                {
                  "vertices": [[x1, y1], [x2, y2],...,[xm, ym]],  // Tight bounding polygons. Curved text can have more than 4 vertices.
                  "text": "the text content of this word",
                  "legible": true,  // If false, the word can't be recognized and the `text` field will be an empty string.
                  "handwritten": false,  // True for handwritten text, false for printed text.
                  "vertical": false,  // If true, characters have a vertical layout.
                }, ...
              ]
            }, ...
          ]
        }, ...
      ]
    }, ...
  ]
}
```

-   Lines in a paragraph may not follow the reading order while words inside a
    line are ordered respect to the proper reading order.

-   Vertices in the ground-truth word polygon follow a specific order. See the
    below figure for details.

![samples](docs/images/vertices.png)

## Evaluation

Uses the following command for word-level detection evaluation:

```
python3 eval.py --gt=gt/validation.jsonl --result=/path/to/your/results.jsonl --output=/tmp/scores.txt --mask_stride=1
```

Add `--e2e` for end-to-end evaluation. Add `--eval_lines` and
`--eval_paragraphs` to enable line-level and paragraph-level evaluation.
Word-level evaluation is always performed.

Be careful when you set the `mask_stride` parameter. Please read the flag's
definition. For results intended to be included in any publications, users are
required to set `--mask_stride=1`.

To expedite the evaluation, users can also set the `num_workers` flag to run the
job in parallel. Note that using too many workers may result in OOM.

Your predictions should be in a `.jsonl` file with the following format, even
for word-level only evaluation, in which case a paragraph can contain a single
line which contains a single word. For detection only evaluation, `text` can be
set to an empty string.

```jsonc
{
  "annotations": [  // List of dictionaries, one for each image.
    {
      "image_id": "the filename of corresponding image.",
      "paragraphs": [  // List of paragraphs.
        {
          "lines": [  // List of lines.
            {
              "text": "the text content of the entire line",  // Set to empty string for detection-only evaluation.
              "words": [  // List of words.
                {
                  "vertices": [[x1, y1], [x2, y2],...,[xm, ym]],
                  "text": "the text content of this word",  // Set to empty string for detection-only evaluation.
                }, ...
              ]
            }, ...
          ]
        }, ...
      ]
    }, ...
  ]
}
```

**NOTE** In evaluation, lines and paragraphs are defined as the union of
pixel-level masks of the underlying word level polygons.

### Sample output on the validation set

We attached a sample output file in compressed form, `sample_output.jsonl.gz`,
to this repo. Use `gzip -d sample_output.jsonl.gz` to uncompress it and pass to
`--result`. You should be able to see the scores as those in
`sample_eval_scores.txt`. These are the outputs and results on the validation
set of the Unified Detector (line based) model proposed in our paper. Note the
results are different from the ones reported in the paper which are computed on
the test set.

### Evaluation on the test set

To evaluate on the test set, please go to the [Robust Reading Competition](https://rrc.cvc.uab.es/?ch=18) website. 
You will need to compress your `json` file with `gzip` before uploading it. The evaluation will take around `0.5`~`1` hour.


## License

The HierText dataset are released under
[**CC BY-SA 4.0**](https://creativecommons.org/licenses/by-sa/4.0/) license.

## BibTeX

Please cite our [paper](https://arxiv.org/pdf/2203.15143.pdf) if you use the
dataset in your work:

```
@inproceedings{long2022towards,
  title={Towards End-to-End Unified Scene Text Detection and Layout Analysis},
  author={Long, Shangbang and Qin, Siyang and Panteleev, Dmitry and Bissacco, Alessandro and Fujii, Yasuhisa and Raptis, Michalis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{long2023icdar,
  title={ICDAR 2023 Competition on Hierarchical Text Detection and Recognition},
  author={Long, Shangbang and Qin, Siyang and Panteleev, Dmitry and Bissacco, Alessandro and Fujii, Yasuhisa and Raptis, Michalis},
  journal={arXiv preprint arXiv:2305.09750},
  year={2023}
}
```

**This is not an officially supported Google product.** If you have any
question, please email us at hiertext@google.com.
