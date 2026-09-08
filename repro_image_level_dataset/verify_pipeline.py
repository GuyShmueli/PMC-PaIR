#!/usr/bin/env python3
"""Check conversion requirements, then compare two independent complete runs.

These are deterministic software checks using synthetic text examples, not
image annotation or a semantic accuracy evaluation of the source dataset.

Run only the phrase and compound-figure checks with --checks-only. To verify
built artifacts, use --input PAIRS.json --taxonomy taxonomy.json
--dataset-dir output; full verification writes output/checks.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from build_image_level_simple import Mapper, construct, digest, json_bytes

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE.parent / 'pairs_no_pets_categories.json',
                        help='Source pair JSON used for independent rebuilds')
    parser.add_argument('--taxonomy', type=Path, default=HERE / 'taxonomy.json',
                        help='Taxonomy JSON used for matching checks and rebuilds')
    parser.add_argument('--dataset-dir', type=Path, default=HERE / 'output',
                        help='Directory containing generated artifacts and receiving checks.json')
    parser.add_argument('--checks-only', action='store_true',
                        help='Run phrase and compound-figure checks without reading or rebuilding artifacts')
    args = parser.parse_args()
    taxonomy = json.loads(args.taxonomy.read_text())
    mapper = Mapper(taxonomy)
    cases = {
        'diagnosis': [
            ('cancer', ['mass_or_lesion']),
            ('tumor', ['mass_or_lesion']),
            ('tumour', ['mass_or_lesion']),
            ('neoplasm', ['mass_or_lesion']),
            ('lump', ['mass_or_lesion']),
            ('mass', ['mass_or_lesion']),
            ('nodule', ['mass_or_lesion']),
            ('lesion', ['mass_or_lesion']),
            ('TUMOURS', ['mass_or_lesion']),
            ('lesions', ['mass_or_lesion']),
            ('masses', ['mass_or_lesion']),
            ('hepatocellular carcinoma', ['mass_or_lesion']),
            ('HCC', ['mass_or_lesion']),
            ('brain metastasis', ['mass_or_lesion']),
            ('brain metastases', ['mass_or_lesion']),
            ('metastatic brain tumor', ['mass_or_lesion']),
            ('meningiomas', ['mass_or_lesion']),
            ('hepatic abscess', ['infection_or_abscess']),
            ('liver abscess', ['infection_or_abscess']),
            ('abscess lesion', ['infection_or_abscess']),
            ('cystic neoplasm', ['mass_or_lesion']),
            ('cystic lesion', ['cystic_disorder']),
            ('cystic lymphangioma', ['mass_or_lesion']),
            ('hematoma', ['hemorrhage_or_hematoma']),
            ('haematoma', ['hemorrhage_or_hematoma']),
            ('hematomas', ['hemorrhage_or_hematoma']),
            ('thrombus', ['thrombosis_or_embolism']),
            ('thrombosis', ['thrombosis_or_embolism']),
            ('embolism', ['thrombosis_or_embolism']),
            ('PRES', ['neurologic_disorder']),
            ('posterior reversible encephalopathy syndrome', ['neurologic_disorder']),
            ('tumor with hemorrhage', ['hemorrhage_or_hematoma', 'mass_or_lesion']),
            ('no tumor', ['normal_or_negative_report']),
            ('no tumor or cyst', ['normal_or_negative_report']),
            ('no tumor but pleural effusion', ['edema_or_effusion', 'normal_or_negative_report']),
            ('hemorrhage and no tumor', ['hemorrhage_or_hematoma', 'normal_or_negative_report']),
            ('tumor without hemorrhage', ['mass_or_lesion', 'normal_or_negative_report']),
            ('resolved hematoma', ['postoperative_or_resolved_state']),
            ('hematoma has resolved', ['postoperative_or_resolved_state']),
            ('resolution of hematoma', ['postoperative_or_resolved_state']),
            ('no resolution of hematoma', ['hemorrhage_or_hematoma']),
            ('partially resolved hematoma', ['hemorrhage_or_hematoma']),
            ('no change in tumor', ['mass_or_lesion']),
            ('treated tumor', ['mass_or_lesion']),
            ('resolved abscess with new tumor', ['mass_or_lesion', 'postoperative_or_resolved_state']),
            ('normal study', ['normal_or_negative_report']),
            ('absence of mass effect', ['normal_or_negative_report']),
            ('postoperative changes', ['postoperative_or_resolved_state']),
            ('inflammatory myofibroblastic tumor', ['mass_or_lesion']),
            ('inflammatory pseudotumor', ['inflammatory_or_autoimmune_disorder']),
            ('retinal detachment lesion', ['eye_or_orbit_disorder']),
            ('hepatic artery pseudoaneurysm', ['aneurysm_or_dissection']),
            ('aneurysmal bone cyst', ['cystic_disorder']),
            ('fibrous dysplasia', ['congenital_or_developmental_disorder']),
            ('unknown diagnosis', ['other']),
            ('liver', ['other']),
            ('CT', ['other']),
            ('stenosis', ['other']),
        ],
        'modality': [
            ('CT', ['ct']), ('CTA', ['ct']), ('CBCT', ['ct']),
            ('computed tomography', ['ct']), ('CT angiography', ['ct']),
            ('MRI/MRA/MRCP', ['mri']), ('magnetic resonance imaging', ['mri']),
            ('magnetic resonance angiography', ['mri']),
            ('X-ray', ['radiography']), ('radiograph', ['radiography']),
            ('panoramic radiograph', ['radiography']), ('mammography', ['radiography']),
            ('ultrasound/sonography/echocardiography/TTE/TEE', ['ultrasound']),
            ('PET/SPECT/scintigraphy', ['nuclear_medicine']),
            ('PET/CT', ['ct', 'nuclear_medicine']),
            ('PET–CT', ['ct', 'nuclear_medicine']),
            ('CT and MRI', ['ct', 'mri']),
            ('fluorescein angiography', ['optical_imaging']),
            ('single photon emission computed tomography', ['nuclear_medicine']),
            ('angiography', ['angiography_fluoroscopy']),
            ('thoracoscopy', ['endoscopy']),
            ('unknown modality', ['other']),
        ],
        'anatomy': [
            ('liver segment VIII', ['hepatobiliary_pancreatic']),
            ('hepatic lobe', ['hepatobiliary_pancreatic']),
            ('biliary tract', ['hepatobiliary_pancreatic']),
            ('pancreas', ['hepatobiliary_pancreatic']),
            ('cervical spine', ['spine_spinal_cord']),
            ('lumbar spine', ['spine_spinal_cord']),
            ('spinal cord', ['spine_spinal_cord']),
            ('femur / knee / shoulder / skeletal muscle', ['musculoskeletal']),
            ('hepatic artery', ['vasculature']),
            ('hepatic arterial branches', ['vasculature']),
            ('hepatic artery and liver', ['hepatobiliary_pancreatic', 'vasculature']),
            ('pulmonary artery', ['vasculature']),
            ('left pulmonary valve', ['heart']),
            ('lateral ventricles', ['brain_intracranial']),
            ('left ventricle', ['heart']),
            ('unknown anatomy', ['other']),
        ],
    }
    failures = []
    count = 0
    for field, examples in cases.items():
        for text, expected in examples:
            actual = mapper.classify(field, text)['labels']
            count += 1
            if actual != sorted(expected):
                failures.append({'field': field, 'text': text, 'expected': expected, 'actual': actual})
    if failures:
        raise AssertionError(json.dumps(failures, indent=2))

    # A compound figure may get disjoint labels from separate incident pairs.
    pairs = [
        {'pair_id': ['figure_a', 'figure_b'], 'modality': 'CT', 'anatomy': 'liver', 'diagnosis': 'tumor'},
        {'pair_id': ['figure_a', 'figure_c'], 'modality': 'MRI', 'anatomy': 'spinal cord', 'diagnosis': 'abscess'},
        {'pair_id': ['figure_a', 'figure_d'], 'modality': 'unknown modality', 'anatomy': 'unknown anatomy', 'diagnosis': 'unknown diagnosis'},
    ]
    cache = {f: {r[f]: mapper.classify(f, r[f]) for r in pairs} for f in cases}
    records = {r['image_id']: r for r in construct(pairs, cache)}
    expected_a = {'image_id': 'figure_a', 'pair_indices': [0, 1, 2],
                  'labels': {'modality': ['ct', 'mri'],
                             'anatomy': ['hepatobiliary_pancreatic', 'spine_spinal_cord'],
                             'diagnosis': ['infection_or_abscess', 'mass_or_lesion']}}
    if records['figure_a'] != expected_a:
        raise AssertionError('Compound-figure union/provenance/other removal failed')
    if any(records['figure_d']['labels'][f] != ['other'] for f in cases):
        raise AssertionError('Unmatched image must retain other')
    if records['figure_b']['labels']['diagnosis'] != ['mass_or_lesion'] or records['figure_c']['labels']['diagnosis'] != ['infection_or_abscess']:
        raise AssertionError('Pair projection crossed unrelated images')
    print(f'{count} phrase examples and compound-figure aggregation checks passed', flush=True)
    if args.checks_only:
        return

    generated = ['labeled_images.jsonl', 'phrase_mappings.json', 'unmatched_phrases.csv', 'summary.json']
    hashes = {f: digest((args.dataset_dir / f).read_bytes()) for f in generated}
    repeats = []
    for seed in ['17', '8191']:
        with tempfile.TemporaryDirectory(prefix='simple_dataset_check_') as temp:
            env = dict(os.environ, PYTHONHASHSEED=seed)
            subprocess.run([sys.executable, str(HERE / 'build_image_level_simple.py'),
                            '--input', str(args.input), '--taxonomy', str(args.taxonomy), '--output', temp],
                           env=env, check=True, stdout=subprocess.DEVNULL)
            rebuilt = {f: digest((Path(temp) / f).read_bytes()) for f in generated}
            if rebuilt != hashes:
                raise AssertionError(f'Complete rebuild differs for hash seed {seed}: {rebuilt}')
            repeats.append({'python_hash_seed': seed, 'all_generated_artifacts_identical': True})
            print(f'Complete independent rebuild with hash seed {seed}: byte-identical', flush=True)
    summary = json.loads((args.dataset_dir / 'summary.json').read_text())
    report = {'passed': True, 'phrase_examples_passed': count,
              'compound_figure_union': True, 'other_removed_when_known': True,
              'unmatched_image_nonempty': True, 'correct_pair_provenance': True,
              'no_cross_image_propagation_beyond_shared_pairs': True,
              'structural_checks': summary['checks'], 'repeated_full_runs': repeats,
              'artifact_sha256': hashes, 'input_sha256': summary['input_sha256'],
              'taxonomy_sha256': summary['taxonomy_sha256'],
              'builder_sha256': summary['script_sha256'],
              'verification_script_sha256': digest(Path(__file__).read_bytes()),
              'scope': 'Software requirements and reproducibility only; no per-image annotation or accuracy review.'}
    (args.dataset_dir / 'checks.json').write_bytes(json_bytes(report))


if __name__ == '__main__':
    main()
