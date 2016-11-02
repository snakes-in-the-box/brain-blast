import json

import thunder as td
from extraction import NMF

files = [
    '00.00.test', '00.01.test', '01.00.test',
    '01.01.test', '02.00.test', '02.01.test',
    '03.00.test', '04.00.test', '04.01.test'
]


def learn_dataset(dataset, submission):
    path = '/media/brad/disk2/nf_data_test/neurofinder.' + dataset
    data = td.images.fromtif(path + '/images', ext='tiff')
    algorithm = NMF(k=10, percentile=99, max_iter=50, overlap=0.1)
    model = algorithm.fit(data, chunk_size=(50, 50), padding=(25, 25))
    merged = model.merge(0.1)
    print('found %g regions' % merged.regions.count)
    regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
    result = {'dataset': dataset, 'regions': regions}
    submission.append(result)


def save_submission(submission):
    with open('submission.json', 'w') as f:
        f.write(json.dumps(submission))


if __name__ is "__main__":

    submission = []

    for f in files:
        learn_dataset(f, submission)
