# Reproduce the simple image-level dataset

This directory contains the construction code from `image_level_classification_simple`. The taxonomy and labeling logic are unchanged; the image dataset is exported as `labeled_images.jsonl`. The verification script accepts explicit paths so it can run from this directory.

Requires Python 3.9 or newer and the Python standard library. No package installation is needed.

## Files

- `build_image_level_simple.py`: constructs one record per image from shared pair descriptions.
- `taxonomy.json`: category definitions, normalization, aliases, regular expressions, and matching priorities.
- `verify_pipeline.py`: checks 96 phrase examples, compound-figure aggregation, and reproducibility across two complete rebuilds.
- `provenance.json`: source paths, file hashes, and the expected input and dataset hashes.

The input paired JSON and generated dataset files remain outside this code bundle.

## Build

Run from this directory. In the current workspace:

```sh
IMAGE_LEVEL_PAIR_INPUT=/cs/labs/tomhope/dhtandguy21/largeListsGuy/prepared_pairmed_inputs/broad_no_path_endo/pairs_no_pets_categories.json
python build_image_level_simple.py --input "$IMAGE_LEVEL_PAIR_INPUT" --output output
```

On another machine, set `IMAGE_LEVEL_PAIR_INPUT` to the location of the same input file. The input must be a JSON list whose records contain `pair_id` (two image IDs), `modality`, `anatomy`, and `diagnosis`. The supplied taxonomy expects 87,105 unique images.

The builder writes these files under `output/`:

- `labeled_images.jsonl`: image IDs, three label groups, and contributing zero-based pair indices.
- `phrase_mappings.json`: mappings and matching evidence for distinct input phrases.
- `unmatched_phrases.csv`: unmatched phrases and their pair frequencies.
- `summary.json`: category distributions, coverage, structural checks, and hashes.

The builder also accepts `--taxonomy PATH`. Always supply `--input` here: its original default refers to a paired JSON in the parent directory.

## Verify

Run the small software checks without building the dataset:

```sh
python verify_pipeline.py --checks-only
```

After building, verify the generated files against two independent complete rebuilds:

```sh
python verify_pipeline.py --input "$IMAGE_LEVEL_PAIR_INPUT" --dataset-dir output
```

Use the same input and taxonomy paths used for the initial build. Verification writes `output/checks.json`; temporary rebuild directories are removed automatically. It also accepts `--taxonomy PATH`.

## Construction and expected result

Only pair IDs and the modality, anatomy, and diagnosis descriptions are used. Each pair's mapped labels are assigned to both images. For each image, labels from all its pairs are combined, deduplicated, and sorted. A compound figure is one image record and may receive labels describing different panels.

The vocabulary has 12 modality, 20 anatomy, and 40 diagnosis categories, including `other`. Every image has at least one label per field. `other` is removed when a known label is available in that field.

The original input contains 86,316 pairs and produces 87,105 image records. Images assigned `other` number 67 for modality, 277 for anatomy, and 2,259 for diagnosis. The expected `labeled_images.jsonl` SHA-256 is:

```text
6799e564744ca220f9480acb2b3436ed4b98d3346b3067b750736029cfad272f
```

Moving the input file can change the path recorded in `summary.json`; it does not change the generated image labels. The verification script checks reproducibility using the paths supplied for that run.
