#!/usr/bin/env python3
"""Deterministic, standard-library-only conversion of shared pair descriptions.

Only pair_id, modality, anatomy, and diagnosis are read. A compound figure is
one image: labels from all its pairs are unioned; panel coverage is not required.
Run: python build_image_level_simple.py [--input PATH] [--output DIRECTORY]
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

FIELDS = ('modality', 'anatomy', 'diagnosis')
HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


class Mapper:
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        self.replacements = [(re.compile(x['pattern']), x['replacement'])
                             for x in taxonomy['normalization']['replacements']]
        self.rules = {}
        for field in FIELDS:
            self.rules[field] = []
            for rule in taxonomy['rules'][field]:
                alternatives = [self.alias_pattern(a) for a in rule.get('aliases', [])]
                alternatives.extend(rule.get('patterns', []))
                pattern = re.compile(r'(?<!\w)(?:' + '|'.join(alternatives) + r')(?!\w)')
                self.rules[field].append((rule, pattern))
        matching = taxonomy['matching']
        self.clauses = re.compile(matching['diagnosis_clause_separator'])
        self.negative = re.compile(matching['negation_prefix'])
        self.resolved_before = re.compile(matching['resolution_prefix'])
        self.resolved_after = re.compile(matching['resolution_suffix'])
        self.unresolved = re.compile(matching['unresolved_context'])
        self.negative_after = re.compile(matching['negation_suffix'])
        self.normal = re.compile(matching['normal_report'])
        self.postoperative = re.compile(matching['postoperative_report'])
        self.vessel = re.compile(matching['anatomy_vessel_span'])

    def alias_pattern(self, alias):
        normalized = self.normalize(alias)
        pattern = re.escape(normalized)
        # Controlled alias inflection, not arbitrary stemming of input words.
        if self.taxonomy['normalization'].get('alias_regular_plurals', False):
            if re.search(r'[bcdfghjklmnpqrstvwxz]y$', normalized):
                pattern = pattern[:-1] + '(?:y|ies)'
            elif re.search(r'[a-z]$', normalized) and not normalized.endswith('s'):
                pattern += 's?'
        return pattern

    def normalize(self, text):
        text = unicodedata.normalize('NFKD', text).casefold()
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[‐‑‒–—−-]', ' ', text)
        text = re.sub(r'[^\w\s/,;.]', ' ', text)
        text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)
        text = text.replace('_', ' ')
        for pattern, replacement in self.replacements:
            text = pattern.sub(replacement, text)
        return ' '.join(text.split())

    def hits(self, field, text):
        hits = []
        for rule, pattern in self.rules[field]:
            for match in pattern.finditer(text):
                hits.append({'label': rule['label'], 'rule': rule['id'],
                             'priority': rule['priority'], 'kind': rule.get('kind', 'specific'),
                             'start': match.start(), 'end': match.end(), 'text': match.group()})
        # More specific matches consume overlapping generic/organ terms.
        accepted = []
        for hit in sorted(hits, key=lambda h: (-h['priority'], -(h['end'] - h['start']), h['rule'], h['start'])):
            if any(hit['start'] < h['end'] and h['start'] < hit['end'] for h in accepted):
                continue
            accepted.append(hit)
        return accepted

    def classify(self, field, raw):
        text = self.normalize(raw)
        labels, evidence = set(), []
        if field == 'anatomy':
            # Mask vessel phrases before looking for tissue/organ aliases.
            # Explicitly mentioned organs elsewhere in the string still match.
            spans = list(self.vessel.finditer(text))
            masked = list(text)
            for m in spans:
                labels.add('vasculature')
                evidence.append({'rule': 'anatomy_vessel_span', 'text': m.group(), 'label': 'vasculature'})
                masked[m.start():m.end()] = ' ' * (m.end() - m.start())
            hits = self.hits(field, ''.join(masked))
            labels.update(h['label'] for h in hits)
            evidence.extend(hits)
        elif field == 'modality':
            hits = self.hits(field, text)
            labels.update(h['label'] for h in hits)
            evidence.extend(hits)
        else:
            for clause in filter(None, (c.strip() for c in self.clauses.split(text))):
                hits = self.hits(field, clause)
                specific = [h for h in hits if h['kind'] == 'specific']
                if specific:
                    hits = specific
                elif any(h['kind'] == 'fallback' for h in hits):
                    hits = [h for h in hits if h['kind'] == 'fallback']
                for hit in hits:
                    before, after = clause[:hit['start']], clause[hit['end']:]
                    label, state = hit['label'], 'reported'
                    if self.negative.search(before) or self.negative_after.search(after):
                        label, state = 'normal_or_negative_report', 'negated'
                    elif not self.unresolved.search(clause) and (self.resolved_before.search(before) or self.resolved_after.search(after)):
                        label, state = 'postoperative_or_resolved_state', 'resolved'
                    labels.add(label)
                    evidence.append({**hit, 'label': label, 'state': state, 'clause': clause})
                if not hits:
                    if self.postoperative.search(clause):
                        labels.add('postoperative_or_resolved_state')
                        evidence.append({'rule': 'postoperative_report', 'label': 'postoperative_or_resolved_state', 'clause': clause})
                    elif self.normal.search(clause):
                        labels.add('normal_or_negative_report')
                        evidence.append({'rule': 'normal_report', 'label': 'normal_or_negative_report', 'clause': clause})
        return {'normalized': text, 'labels': sorted(labels or {'other'}),
                'matches': sorted(evidence, key=lambda h: (h.get('clause', ''), h.get('start', -1), h['rule']))}


def construct(pairs, cache):
    images = {}
    for index, pair in enumerate(pairs):
        for image_id in pair['pair_id']:
            rec = images.setdefault(image_id, {'image_id': image_id,
                                    'labels': {f: set() for f in FIELDS}, 'pair_indices': []})
            rec['pair_indices'].append(index)
            for field in FIELDS:
                rec['labels'][field].update(cache[field][pair[field]]['labels'])
    result = []
    for image_id, rec in sorted(images.items()):
        rec['pair_indices'] = sorted(set(rec['pair_indices']))
        for field in FIELDS:
            if len(rec['labels'][field]) > 1:
                rec['labels'][field].discard('other')
            rec['labels'][field] = sorted(rec['labels'][field])
        result.append(rec)
    return result


def dataset_bytes(records):
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in records).encode()


def check(records, pairs, taxonomy):
    ids = [r['image_id'] for r in records]
    endpoints = {i for pair in pairs for i in pair['pair_id']}
    expected = taxonomy['expected_unique_images']
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f'Expected {expected} unique images; got {len(ids)} records, {len(set(ids))} unique')
    if set(ids) != endpoints:
        raise ValueError('Image IDs do not match pair endpoints')
    index_memberships = Counter()
    for r in records:
        for field in FIELDS:
            labels = r['labels'][field]
            if not labels or labels != sorted(set(labels)) or not set(labels) <= taxonomy['categories'][field].keys():
                raise ValueError(f'Invalid labels for {r["image_id"]}: {field}')
            if 'other' in labels and len(labels) != 1:
                raise ValueError('other coexists with known label')
        if r['pair_indices'] != sorted(set(r['pair_indices'])):
            raise ValueError('Invalid pair indices')
        for index in r['pair_indices']:
            if r['image_id'] not in pairs[index]['pair_id']:
                raise ValueError('Incorrect provenance')
            index_memberships[index] += 1
    if any(index_memberships[i] != len(set(pair['pair_id'])) for i, pair in enumerate(pairs)):
        raise ValueError('Missing pair provenance')
    return {'expected_image_count': True, 'unique_image_ids': True, 'all_endpoints': True,
            'declared_labels_only': True, 'nonempty_sorted_unique_labels': True,
            'other_exclusive': True, 'complete_correct_pair_indices': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE.parent / 'pairs_no_pets_categories.json')
    parser.add_argument('--taxonomy', type=Path, default=HERE / 'taxonomy.json')
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    data = args.input.read_bytes()
    pairs = json.loads(data)
    taxonomy_data = args.taxonomy.read_bytes()
    taxonomy = json.loads(taxonomy_data)
    if not isinstance(pairs, list):
        raise ValueError('Input must be a list')
    for index, pair in enumerate(pairs):
        if not isinstance(pair.get('pair_id'), list) or len(pair['pair_id']) != 2:
            raise ValueError(f'Invalid pair_id at {index}')
        if any(not isinstance(i, str) or not i.strip() for i in pair['pair_id']):
            raise ValueError(f'Invalid image ID at {index}')
        if any(not isinstance(pair.get(f), str) or not pair[f].strip() for f in FIELDS):
            raise ValueError(f'Missing description at {index}')
    mapper = Mapper(taxonomy)
    frequencies = {f: Counter(p[f] for p in pairs) for f in FIELDS}
    cache = {f: {} for f in FIELDS}
    for field in FIELDS:
        for phrase, count in sorted(frequencies[field].items()):
            cache[field][phrase] = {**mapper.classify(field, phrase), 'pair_frequency': count}
        print(f'{field}: mapped {len(cache[field]):,} distinct phrases', flush=True)
    records = construct(pairs, cache)
    checks = check(records, pairs, taxonomy)
    output = dataset_bytes(records)
    # Reconstruct to check deterministic aggregation. Separate process reruns
    # additionally verify matching and serialization with different hash seeds.
    repeat = dataset_bytes(construct(pairs, cache))
    if output != repeat:
        raise ValueError('Dataset construction is not reproducible')
    checks['repeated_construction_identical'] = True
    stats = {}
    for field in FIELDS:
        counts = Counter(label for r in records for label in r['labels'][field])
        other = counts['other']
        unmatched = {p: v for p, v in cache[field].items() if v['labels'] == ['other']}
        stats[field] = {'declared_categories': len(taxonomy['categories'][field]),
                        'observed_categories': len(counts), 'category_image_counts': dict(sorted(counts.items())),
                        'other_images': other, 'other_image_percent': round(100 * other / len(records), 6),
                        'other_below_2_percent': other / len(records) < .02,
                        'distinct_input_phrases': len(cache[field]),
                        'unmatched_distinct_phrases': len(unmatched),
                        'unmatched_pair_descriptions': sum(v['pair_frequency'] for v in unmatched.values()),
                        'multilabel_images': sum(len(r['labels'][field]) > 1 for r in records)}
    summary = {'input_path': str(args.input.resolve()), 'input_sha256': digest(data),
               'taxonomy_sha256': digest(taxonomy_data), 'script_sha256': digest(Path(__file__).read_bytes()),
               'dataset_sha256': digest(output), 'pairs': len(pairs), 'unique_images': len(records),
               'fields': stats, 'checks': checks,
               'method': 'Deterministic phrase rules; shared labels assigned to both endpoints; union across all incident pairs.',
               'compound_figures': 'One image ID is one unit; labels need not cover every panel.',
               'scope': 'Only pair_id, modality, anatomy, diagnosis; no captions, pixels, articles or previous annotations.',
               'interpretation': 'Broad text-derived labels. Coverage and structural checks do not measure semantic accuracy. A negative-report label concerns the mentioned condition, not necessarily the whole image.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'labeled_images.jsonl').write_bytes(output)
    (args.output / 'phrase_mappings.json').write_bytes(json_bytes(cache))
    with (args.output / 'unmatched_phrases.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['field', 'phrase', 'normalized', 'pair_frequency'])
        for field in FIELDS:
            for phrase, value in sorted(cache[field].items(), key=lambda pv: (-pv[1]['pair_frequency'], pv[0])):
                if value['labels'] == ['other']:
                    writer.writerow([field, phrase, value['normalized'], value['pair_frequency']])
    (args.output / 'summary.json').write_bytes(json_bytes(summary))
    print(json.dumps({'images': len(records), 'other_percent': {f: stats[f]['other_image_percent'] for f in FIELDS},
                      'dataset_sha256': summary['dataset_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
