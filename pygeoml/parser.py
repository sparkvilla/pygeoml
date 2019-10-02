import os
import re


class Sentinel2:

    PATTERN = re.compile(
            # example pattern to match
            # T37MBN_20170718T075211_B02.jp2
            r'.*_(?P<date>[0-9]{8})'
            r'.*_(?P<band>B0(2|3|4|8))'
            r'.*(?P<fext>jp2$)'
            )

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.lookup = {'B02': 'Sent2_B02', # BLUE
                       'B03': 'Sent2_B03', # GREEN
                       'B04': 'Sent2_B04', # RED
                       'B08': 'Sent2_B08'} # NIR
        self.rfiles = []
        for path_to_f in os.listdir(self.dirpath):
            match = Sentinel2.PATTERN.match(os.path.basename(path_to_f))
            if match:
                self.rfiles.append((os.path.join(self.dirpath,
                                                 path_to_f),
                                    self.lookup[match.group('band')]))
                self.date = match.group('date')
        self._verify_rfiles_not_empty()
        # sort by the Band number
        self.srfiles = sorted(self.rfiles)

    def __repr__(self):
        return 'Sentinel2({})'.format(self.dirpath)

    def _verify_rfiles_not_empty(self):
        assert len(self.rfiles) > 0, "No file matching found at {}".format(self.dirpath)


if __name__ == '__main__':

    basedir = '/home/diego/work/dev/ess_diego/Data_Diego'
    imgdir = os.path.join(
        basedir,
        'S2B_MSIL2A_20190812T073619_N9999_R092_T37MBN_20190919T144441.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555/IMG_DATA/R10m/')

    p = Sentinel2(imgdir)
    print(p.rfiles)
    print("")
    print(p.srfiles)
    print(p.date)
