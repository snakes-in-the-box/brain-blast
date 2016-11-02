# built from thunder examples

import json

import thunder as td
from extraction import NMF


class BrainBlast:
    def __init__(self, folder, dataset, submission, submission_path):
        self.folder = folder
        self.dataset = dataset
        self.submission = submission
        self.submission_path = submission_path

    def learn_dataset(self):
        path = self.folder_path + self.dataset
        data = td.images.fromtif(path + '/images', ext='tiff')
        algorithm = NMF(k=11, percentile=99, max_iter=50, overlap=0.1)
        model = algorithm.fit(data, chunk_size=(50, 50), padding=(25, 25))
        merged = model.merge(0.1)
        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
        result = {'dataset': self.dataset, 'regions': regions}
        self.submission.append(result)

    def save_submission(self):
        with open('submission.json', 'w') as f:
            f.write(json.dumps(submission))


if __name__ is "__main__":

    folder_path = '/media/brad/disk2/nf_data_test/neurofinder.'
    files = [
        '00.00.test',
        '00.01.test',
        '01.00.test',
        '01.01.test',
        '02.00.test',
        '02.01.test',
        '03.00.test',
        '04.00.test',
        '04.01.test'
    ]
    submission = []
    bb = BrainBlast(folder_path, files, submission)

    for f in files:
        bb.learn_dataset(f, submission)
