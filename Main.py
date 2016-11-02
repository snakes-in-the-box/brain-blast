# built from thunder examples

import json

import thunder as td
from extraction import NMF


class BrainBlast:
    def __init__(self, f, d, s):
        self.folder = f
        self.data = d
        self.submission = s

    def learn_data(self):
        for d in self.data:
            path = self.folder + d
            print("Analyzing on ", path)
            data = td.images.fromtif(path + '/images', ext='tiff')
            algorithm = NMF(k=10, percentile=99, max_iter=50, overlap=0.1)
            model = algorithm.fit(data, chunk_size=(50, 50), padding=(25, 25))
            merged = model.merge(0.1)
            regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
            result = {'dataset': self.data, 'regions': regions}
            self.submission.append(result)

    def save_submission(self):
        with open('submission.json', 'w') as sf:
            sf.write(json.dumps(submission))


if __name__ == "__main__":

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
    bb.learn_data()
    bb.save_submission()
