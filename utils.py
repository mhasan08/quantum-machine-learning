
'''
    @author Munawar Hasan <munawar.hasan@nist.gov>
'''

from multipledispatch import dispatch


class Logger:
    @dispatch(str)
    def __log__(self, m):
        print(m)

    @dispatch(str, str)
    def __log__(self, x, y):
        print(x, y)

    @dispatch(str, str, str)
    def __log__(self, x, y, z):
        print(x, y, z)