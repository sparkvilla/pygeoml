import os
import re


class Sentinel2:

    PATTERN = re.compile(
            # example pattern to match
            # T37MBN_20170718T075211_B02.jp2
            r'.*_(?P<date>[0-9]{8})'
            r'.*_(?P<band>B0(2|3|4|8))'
            r'\.(?P<fext>jp2$)'
            )

    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.lookup = {'B02': 'blue',
                       'B03': 'green',
                       'B04': 'red',
                       'B08': 'nir'}
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
        self.fname = '_'.join((self.__class__.__name__, self.date, '.tif'))

    def __repr__(self):
        return 'Sentinel2({})'.format(self.dirpath)

    def _verify_rfiles_not_empty(self):
        assert len(self.rfiles) > 0, "No file matching found at {}".format(self.dirpath)


if __name__ == '__main__':

    basedir = '/home/diego/work/dev/ess_diego/'
    imgdir = os.path.join(
        basedir,
        'Data_Diego/Sentinel/Sentinel_2017/S2B_MSIL1C_20171018T072859_N0205_R049_T37MBN_20171018T074738.SAFE/GRANULE/L1C_T37MBN_A003221_20171018T074738/IMG_DATA')

    p = Sentinel2(imgdir)
    print(p.rfiles)
    print("")
    print(p.srfiles)
    print(p.date)
    print(p.fname)
