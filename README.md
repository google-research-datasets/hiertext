# The HierText Dataset

![samples](docs/images/dataset.png)

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
enables research into unified OCR and layout analysis.

## Getting Started

First clone the project:

```
git clone https://github.com/google-research-datasets/hiertext.git
```

(Optional) Create and enter a virtual environment:

```
virtualenv -p python3 hiertext_env
source ./hiertext_env/bin/activate
```

Then install the required dependencies using:

```
cd hiertext
pip install -r requirements.txt
```

The ground-truth annotations of `train`, `validation` and `test` sets are
contained in three compressed .jsonl files: `train.jsonl.gz`, `val.jsonl.gz` and
`test.jsonl.gz` respectively, under `gt` subdirectory. Use the following command
to decompress the three files:

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

NOTE In our evaluation, lines and paragraphs are defined as the union of
pixel-level masks of the underlying word level polygons.

### Images

Images in HierText are of higher resolution with their long side constrained to
1600 pixels compared to previous datasets based on Open Images that are
constrained to 1024 pixels. This results in more legible small text. The
filename of each image is its corresponding image ID in the Open Images dataset.

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
          "lines": [  // List of dictionaries, one for each text line contained in this paragraph.
            {
              "vertices": [[x1, y1], [x2, y2],...,[x4, y4]],  // A loose rotated rectangle with absolute values.
              "text": "the text content of the entire line",
              "legible": true,  // A line is legible if and only if all of its words are legible.
              "handwritten": false,  // True for handwritten text, false for printed text.
              "vertical": false,  // If true, characters have a vertical layout.
              "words": [  // List of dictionaries, one for each word contained in this line.
                {
                  "vertices": [[x1, y1], [x2, y2],...,[xm, ym]],  // Tight bounding polygons. Curved text can have more than 4 vertices.
                  "text": "the text content of this word",
                  "legible": true,  // If false, the word can't be recognized and the `text` field will be an empty string.
                  "handwritten": false,  // True for handwritten text, false for printed text.
                  "vertical": false,  // If true, characters have a vertical layout.
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## Evaluation

## License

The HierText dataset are released under
[**CC BY-SA 4.0**](https://creativecommons.org/licenses/by-sa/4.0/) license.
Please cite the following paper if you use the dataset in your work:

```
@inproceedings{long2022towards,
  title={Towards End-to-End Unified Scene Text Detection and Layout Analysis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

